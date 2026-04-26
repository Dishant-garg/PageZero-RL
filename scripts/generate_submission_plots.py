#!/usr/bin/env python3
"""Generate the 5 submission plots from a PageZero training run + eval JSONs.

Reads:
  * ``<run_dir>/reward_log.csv``        (per-episode reward log from train.py)
  * ``<run_dir>/trajectories.jsonl``    (optional; used for termination-reason plot)
  * ``<run_dir>/baseline_eval.json``    (from scripts/eval_checkpoint.py)
  * ``<run_dir>/final_eval.json``       (from scripts/eval_checkpoint.py)

Writes (under ``<run_dir>/plots/``):
  * plot_01_overall_reward_curve.png
  * plot_02_resolved_rate_curve.png
  * plot_03_taskwise_baseline_vs_trained_reward.png
  * plot_04_taskwise_baseline_vs_trained_success.png
  * plot_05_termination_reason_distribution.png

Usage:

    python scripts/generate_submission_plots.py --run-dir outputs/pagezero-...

Plots are intentionally created with explicit axes/legends and consistent
task ordering so the README + judging UI can embed them directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("submission_plots")


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ──────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────


def load_reward_log(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        logger.warning("reward_log not found: %s", csv_path)
        return []
    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["episode"] = int(float(row.get("episode") or 0))
                row["total_reward"] = float(row.get("total_reward") or 0.0)
                row["is_resolved"] = bool(int(row.get("is_resolved") or 0))
                row["num_steps"] = int(float(row.get("num_steps") or 0))
            except Exception:
                continue
            rows.append(row)
    rows.sort(key=lambda r: r["episode"])
    return rows


def load_trajectories(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_eval(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        logger.warning("eval JSON not found: %s", path)
        return None
    with open(path) as f:
        return json.load(f)


def rolling_mean(values: List[float], window: int) -> List[float]:
    out: List[float] = []
    for i, _ in enumerate(values):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        out.append(sum(chunk) / max(1, len(chunk)))
    return out


# ──────────────────────────────────────────────────────────────────────────
# Plot 1: overall reward curve (per episode + rolling avg + trend)
# ──────────────────────────────────────────────────────────────────────────


def plot_overall_reward(rows: List[Dict[str, Any]], out_path: Path) -> None:
    plt = _import_matplotlib()
    if not rows:
        logger.warning("plot_01: no reward rows")
        return
    episodes = [r["episode"] for r in rows]
    rewards = [r["total_reward"] for r in rows]
    window = min(10, max(1, len(rewards) // 4) or 1)
    rolling = rolling_mean(rewards, window)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.3, color="steelblue", marker="o",
            markersize=3, linewidth=1, label="Per episode")
    ax.plot(episodes, rolling, color="steelblue", linewidth=2.5,
            label=f"Rolling mean (w={window})")
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)

    if len(episodes) >= 2:
        try:
            import numpy as np
            z = np.polyfit(episodes, rewards, 1)
            trend = [z[0] * x + z[1] for x in episodes]
            arrow = "up" if z[0] > 0 else "down"
            ax.plot(episodes, trend, color="crimson", linestyle="--",
                    linewidth=1.5, label=f"Trend ({arrow} {abs(z[0]):.3f}/ep)")
        except ImportError:
            pass

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("PageZero — overall reward curve (training)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


# ──────────────────────────────────────────────────────────────────────────
# Plot 2: rolling resolved rate
# ──────────────────────────────────────────────────────────────────────────


def plot_resolved_rate(rows: List[Dict[str, Any]], out_path: Path) -> None:
    plt = _import_matplotlib()
    if not rows:
        logger.warning("plot_02: no reward rows")
        return
    episodes = [r["episode"] for r in rows]
    resolved = [1.0 if r["is_resolved"] else 0.0 for r in rows]
    window = min(10, max(1, len(resolved) // 4) or 1)
    rolling = rolling_mean(resolved, window)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rolling, color="seagreen", linewidth=2.5,
            label=f"Rolling resolved rate (w={window})")
    ax.scatter(episodes, resolved, s=20, color="seagreen", alpha=0.4,
               label="Per-episode resolved (0/1)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Resolved rate")
    ax.set_title("PageZero — resolved rate over training")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


# ──────────────────────────────────────────────────────────────────────────
# Plot 3 / 4: per-task baseline vs trained
# ──────────────────────────────────────────────────────────────────────────


def _collect_task_metric(
    eval_json: Optional[Dict[str, Any]], metric: str
) -> Dict[str, float]:
    if not eval_json:
        return {}
    tasks = eval_json.get("tasks", {}) or {}
    return {tid: float(stats.get(metric, 0.0)) for tid, stats in tasks.items()}


def plot_taskwise_compare(
    baseline: Optional[Dict[str, Any]],
    trained: Optional[Dict[str, Any]],
    metric_key: str,
    metric_label: str,
    title: str,
    out_path: Path,
) -> None:
    plt = _import_matplotlib()
    base_map = _collect_task_metric(baseline, metric_key)
    train_map = _collect_task_metric(trained, metric_key)
    tasks = sorted(set(base_map) | set(train_map))
    if not tasks:
        logger.warning("plot_compare: no eval data for %s", metric_key)
        return
    base_vals = [base_map.get(t, 0.0) for t in tasks]
    train_vals = [train_map.get(t, 0.0) for t in tasks]

    import numpy as np
    x = np.arange(len(tasks))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(tasks)), 5))
    ax.bar(x - width / 2, base_vals, width, label="Baseline", color="lightcoral")
    ax.bar(x + width / 2, train_vals, width, label="Trained", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=15)
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()

    for xi, base, trn in zip(x, base_vals, train_vals):
        ax.text(xi - width / 2, base, f"{base:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + width / 2, trn, f"{trn:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


# ──────────────────────────────────────────────────────────────────────────
# Plot 5: termination-reason distribution
# ──────────────────────────────────────────────────────────────────────────


def _collect_termination_reasons(
    rows: List[Dict[str, Any]],
    trajectories: List[Dict[str, Any]],
) -> Counter:
    counter: Counter = Counter()
    for r in rows:
        cause = (r.get("done_cause") or "").strip()
        if cause:
            counter[cause] += 1
    if counter:
        return counter
    # Fallback: derive from trajectories.jsonl when CSV pre-dates done_cause.
    for traj in trajectories:
        cause = traj.get("done_cause") or ""
        if not cause:
            steps = traj.get("trajectory") or []
            if steps:
                last = steps[-1]
                cause = last.get("done_cause") or ""
        if cause:
            counter[cause] += 1
    return counter


def plot_termination_reasons(
    rows: List[Dict[str, Any]],
    trajectories: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    plt = _import_matplotlib()
    counter = _collect_termination_reasons(rows, trajectories)
    if not counter:
        logger.warning("plot_05: no termination reasons captured (older CSV?)")
        return
    items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(labels)), 5))
    ax.bar(labels, values, color="slateblue")
    ax.set_ylabel("Episode count")
    ax.set_title("PageZero — termination reason distribution")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate PageZero submission plots.")
    p.add_argument("--run-dir", required=True, help="Output directory of a training run.")
    p.add_argument("--reward-log", default=None, help="Override path to reward_log.csv.")
    p.add_argument("--trajectories", default=None, help="Override path to trajectories.jsonl.")
    p.add_argument("--baseline-eval", default=None, help="Override path to baseline_eval.json.")
    p.add_argument("--final-eval", default=None, help="Override path to final_eval.json.")
    p.add_argument("--out-subdir", default="plots", help="Subdir under --run-dir for plot PNGs.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_reward_log(Path(args.reward_log) if args.reward_log else run_dir / "reward_log.csv")
    trajectories = load_trajectories(
        Path(args.trajectories) if args.trajectories else run_dir / "trajectories.jsonl"
    )
    baseline = load_eval(
        Path(args.baseline_eval) if args.baseline_eval else run_dir / "baseline_eval.json"
    )
    final = load_eval(
        Path(args.final_eval) if args.final_eval else run_dir / "final_eval.json"
    )

    plot_overall_reward(rows, out_dir / "plot_01_overall_reward_curve.png")
    plot_resolved_rate(rows, out_dir / "plot_02_resolved_rate_curve.png")
    plot_taskwise_compare(
        baseline, final,
        metric_key="reward_mean",
        metric_label="Mean total reward",
        title="PageZero — per-task reward (baseline vs trained)",
        out_path=out_dir / "plot_03_taskwise_baseline_vs_trained_reward.png",
    )
    plot_taskwise_compare(
        baseline, final,
        metric_key="resolved_rate",
        metric_label="Resolved rate",
        title="PageZero — per-task resolved rate (baseline vs trained)",
        out_path=out_dir / "plot_04_taskwise_baseline_vs_trained_success.png",
    )
    plot_termination_reasons(
        rows, trajectories,
        out_path=out_dir / "plot_05_termination_reason_distribution.png",
    )

    logger.info("Plots saved to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
