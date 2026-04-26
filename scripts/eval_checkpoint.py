#!/usr/bin/env python3
"""Deterministic baseline / final-checkpoint evaluator for PageZero.

Runs a fixed pool of tasks for ``N`` episodes each against a running PageZero
OpenEnv server, using a HuggingFace causal-LM (base or trained adapter) to
choose tools. Emits a JSON summary with overall + per-task metrics so the
notebook / README can show baseline-vs-trained deltas.

Why not just reuse the train-time loggers?
  * Eval uses greedy decoding (``do_sample=False``) for reproducibility.
  * Eval uses a separate seed offset so it never overlaps training prompts.
  * Eval emits a single canonical JSON shape that both ``baseline_eval.json``
    and ``final_eval.json`` follow, which the plotter consumes directly.

Example:

    python scripts/eval_checkpoint.py \\
        --base-model Qwen/Qwen3-0.6B \\
        --output baseline_eval.json \\
        --episodes-per-task 3 \\
        --env-url http://localhost:8000

    python scripts/eval_checkpoint.py \\
        --base-model Qwen/Qwen3-0.6B \\
        --adapter outputs/pagezero-sre-grpo-.../checkpoint-XX \\
        --output final_eval.json \\
        --episodes-per-task 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from client import PageZeroEnvClient
from models import PageZeroAction


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_checkpoint")


# Default eval pool — same pool the trainer uses on the "easy" stage so the
# baseline-vs-trained comparison is apples to apples. Override with --tasks.
DEFAULT_EVAL_TASKS = ["task_1", "task_2", "task_3", "task_4", "task_5"]


SYSTEM_PROMPT = """You are a Staff SRE on-call. Diagnose and fix the cascading incident.

To use a tool, you MUST use this exact format:
<tool_call>
{"name": "check_alerts", "arguments": {}}
</tool_call>

Use triage tools first (check_alerts, pg_stat_activity, redis_info, docker_ps).
Then apply a fix tool (pg_cancel_query, pg_create_index, pg_vacuum, redis_flush_db, docker_restart).
After the stack is healthy, call diagnose_root_cause and write_postmortem, then `done`.
"""


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


# ──────────────────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class EpisodeResult:
    task_id: str
    seed: int
    total_reward: float
    num_steps: int
    is_resolved: bool
    done_cause: str
    diagnose_count: int
    tool_sequence: List[str] = field(default_factory=list)


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first ``<tool_call>`` JSON block from generated text.

    Returns ``None`` when no parseable tool call is present so callers can
    apply the no-tool penalty / format-failure path consistently.
    """
    match = _TOOL_CALL_RE.search(text or "")
    if not match:
        return None
    try:
        obj = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    name = obj.get("name") or obj.get("tool")
    args = obj.get("arguments") or obj.get("args") or {}
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    return {"tool": name, "args": args}


def render_observation(obs: Any, step: int, max_steps: int, last_reward: float) -> str:
    tool_output = getattr(obs, "tool_output", "") or ""
    alerts = getattr(obs, "active_alerts", []) or []
    sla_status = getattr(obs, "sla_status", "OK") or "OK"
    hint = getattr(obs, "judge_feedback", None) or getattr(obs, "hint", "") or ""
    return (
        f"TOOL OUTPUT:\n{tool_output}\n\n"
        f"ACTIVE ALERTS: {alerts}\n"
        f"SLA: {sla_status}\n"
        f"STEP: {step}/{max_steps}\n"
        f"STEP REWARD: {last_reward:+.3f}"
        + (f"\nJUDGE FEEDBACK: {hint}" if hint else "")
    )


def run_episode(
    *,
    client: Any,
    model,
    tokenizer,
    task_id: str,
    seed: int,
    max_turns: int,
    max_new_tokens: int,
    device,
) -> EpisodeResult:
    """Run a single deterministic eval episode.

    ``client`` is expected to be the synchronous wrapper returned by
    ``PageZeroEnvClient(...).sync()`` — OpenEnv's ``EnvClient`` is async-only
    in 0.2.x, so calling ``.reset(...)`` / ``.step(...)`` directly on the bare
    client returns un-awaited coroutines and would silently AttributeError out
    of every episode (visible as ``done_cause="error"`` with empty tool seq).
    """
    import torch  # local import — eval works without torch when --dry-run is used

    reset_result = client.reset(task_id=task_id)
    obs = reset_result.observation
    max_steps = int(getattr(obs, "max_steps", 15) or 15)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Diagnose and fix this incident. task_id={task_id}.\n\n"
                + render_observation(obs, step=0, max_steps=max_steps, last_reward=0.0)
            ),
        },
    ]

    total_reward = 0.0
    tool_sequence: List[str] = []
    done_cause = ""
    diagnose_count = 0
    is_resolved = False
    last_step = 0

    for turn in range(max_turns):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        parsed = parse_tool_call(completion)
        if parsed is None:
            # Format-failure: agent did not produce a valid tool call. Apply
            # the same penalty the trainer uses and stop early.
            total_reward += -0.5
            messages.append({"role": "assistant", "content": completion})
            messages.append(
                {"role": "user", "content": "ERROR: no valid <tool_call> JSON found."}
            )
            done_cause = done_cause or "no_tool"
            break

        try:
            step_result = client.step(
                PageZeroAction(tool=parsed["tool"], args=parsed["args"])
            )
        except Exception as exc:
            logger.warning("step failed for tool=%s: %s", parsed.get("tool"), exc)
            total_reward += -0.1
            done_cause = done_cause or "step_error"
            break

        step_obs = step_result.observation
        reward = float(step_result.reward or 0.0)
        was_done = bool(step_result.done)
        total_reward += reward

        tool_sequence.append(parsed["tool"])
        last_step = int(getattr(step_obs, "step", last_step + 1) or (last_step + 1))
        try:
            diagnose_count = max(
                diagnose_count, int(getattr(step_obs, "diagnose_count", 0) or 0)
            )
        except Exception:
            pass

        messages.append({"role": "assistant", "content": completion})
        messages.append(
            {
                "role": "user",
                "content": render_observation(
                    step_obs, step=last_step, max_steps=max_steps, last_reward=reward
                ),
            }
        )

        if was_done:
            done_cause = str(getattr(step_obs, "done_cause", "") or done_cause or "")
            is_resolved = bool(getattr(step_obs, "stack_healthy", False))
            break

    return EpisodeResult(
        task_id=task_id,
        seed=seed,
        total_reward=total_reward,
        num_steps=last_step,
        is_resolved=is_resolved,
        done_cause=done_cause or "max_turns",
        diagnose_count=diagnose_count,
        tool_sequence=tool_sequence,
    )


# ──────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────


def aggregate(results: List[EpisodeResult]) -> Dict[str, Any]:
    per_task: Dict[str, Dict[str, Any]] = {}
    for r in results:
        per_task.setdefault(
            r.task_id,
            {
                "n_episodes": 0,
                "rewards": [],
                "resolved_count": 0,
                "steps": [],
                "diagnose_counts": [],
                "done_causes": {},
            },
        )
        bucket = per_task[r.task_id]
        bucket["n_episodes"] += 1
        bucket["rewards"].append(r.total_reward)
        bucket["resolved_count"] += int(bool(r.is_resolved))
        bucket["steps"].append(r.num_steps)
        bucket["diagnose_counts"].append(r.diagnose_count)
        cause = r.done_cause or "unknown"
        bucket["done_causes"][cause] = bucket["done_causes"].get(cause, 0) + 1

    summary_per_task: Dict[str, Dict[str, Any]] = {}
    overall_rewards: List[float] = []
    overall_resolved = 0
    overall_n = 0
    for tid, b in sorted(per_task.items()):
        n = max(b["n_episodes"], 1)
        rewards = b["rewards"]
        mean = sum(rewards) / n
        var = sum((r - mean) ** 2 for r in rewards) / n
        std = var ** 0.5
        summary_per_task[tid] = {
            "n_episodes": b["n_episodes"],
            "reward_mean": round(mean, 4),
            "reward_std": round(std, 4),
            "resolved_rate": round(b["resolved_count"] / n, 4),
            "mean_steps": round(sum(b["steps"]) / n, 4),
            "mean_diagnose_count": round(sum(b["diagnose_counts"]) / n, 4),
            "done_causes": b["done_causes"],
        }
        overall_rewards.extend(rewards)
        overall_resolved += b["resolved_count"]
        overall_n += b["n_episodes"]

    n = max(overall_n, 1)
    mean = sum(overall_rewards) / n
    var = sum((r - mean) ** 2 for r in overall_rewards) / n
    return {
        "tasks": summary_per_task,
        "overall": {
            "n_episodes": overall_n,
            "reward_mean": round(mean, 4),
            "reward_std": round(var ** 0.5, 4),
            "resolved_rate": round(overall_resolved / n, 4),
        },
        "raw_episodes": [
            {
                "task_id": r.task_id,
                "seed": r.seed,
                "total_reward": r.total_reward,
                "num_steps": r.num_steps,
                "is_resolved": r.is_resolved,
                "done_cause": r.done_cause,
                "diagnose_count": r.diagnose_count,
                "tool_sequence": r.tool_sequence,
            }
            for r in results
        ],
    }


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic eval for PageZero checkpoints.")
    p.add_argument("--env-url", default=os.environ.get("PAGEZERO_ENV_URL", "http://localhost:8000"))
    p.add_argument("--base-model", required=True, help="HF model id (e.g. Qwen/Qwen3-0.6B)")
    p.add_argument("--adapter", default=None, help="Optional path/repo to a LoRA adapter; omit for baseline.")
    p.add_argument("--tasks", nargs="*", default=DEFAULT_EVAL_TASKS)
    p.add_argument("--episodes-per-task", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--base-seed", type=int, default=10_000,
                   help="Seed offset; kept disjoint from training seeds.")
    p.add_argument("--output", default="eval.json")
    p.add_argument("--label", default=None, help="Optional label stored in JSON (e.g. 'baseline').")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip model load — just exercise env reachability and dataset shape.")
    return p.parse_args()


def load_model(base_model: str, adapter: Optional[str]):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, trust_remote_code=True
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        try:
            model = model.merge_and_unload()
        except Exception:
            logger.warning("merge_and_unload failed — running with PEFT wrapper.")

    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def _write_stub_output(path: str, label: str, tasks: List[str],
                       episodes_per_task: int, base_seed: int,
                       base_model: str, adapter: Optional[str],
                       env_url: str, error: str) -> None:
    """Write a structurally-valid eval JSON when startup blows up.

    Plot generator depends on the JSON existing with the canonical shape. We
    fill in zero metrics + a single ``startup_error`` done_cause per task so
    the failure is obvious in plot 5 (termination reason distribution) but
    nothing downstream crashes on a missing file.
    """
    per_task: Dict[str, Dict[str, Any]] = {}
    raw_eps: List[Dict[str, Any]] = []
    for tid in tasks:
        per_task[tid] = {
            "n_episodes": int(episodes_per_task),
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "resolved_rate": 0.0,
            "mean_steps": 0.0,
            "mean_diagnose_count": 0.0,
            "done_causes": {"startup_error": int(episodes_per_task)},
        }
        for ep in range(int(episodes_per_task)):
            raw_eps.append({
                "task_id": tid,
                "seed": int(base_seed + ep * 7919 + (abs(hash(tid)) % 9999)),
                "total_reward": 0.0,
                "num_steps": 0,
                "is_resolved": False,
                "done_cause": "startup_error",
                "diagnose_count": 0,
                "tool_sequence": [],
            })
    overall_n = int(episodes_per_task) * len(tasks)
    payload = {
        "tasks": per_task,
        "overall": {
            "n_episodes": overall_n,
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "resolved_rate": 0.0,
        },
        "raw_episodes": raw_eps,
        "label": label,
        "base_model": base_model,
        "adapter": adapter,
        "env_url": env_url,
        "episodes_per_task": int(episodes_per_task),
        "base_seed": int(base_seed),
        "startup_error": str(error)[:2000],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def main() -> int:
    args = parse_args()
    label = args.label or ("baseline" if not args.adapter else "trained")

    if args.dry_run:
        logger.info("[dry-run] tasks=%s episodes=%s url=%s", args.tasks,
                    args.episodes_per_task, args.env_url)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"label": label, "tasks": {}, "overall": {}}, f, indent=2)
        return 0

    # ── Startup: model load + env connect. Wrap with detailed error reporting
    # so a silent ``exit=1`` from the notebook subprocess still leaves an
    # actionable trail (full traceback to stderr + stub JSON for plotting).
    try:
        logger.info("Loading %s%s", args.base_model,
                    f" + adapter {args.adapter}" if args.adapter else "")
        model, tokenizer = load_model(args.base_model, args.adapter)
        import torch
        device = next(model.parameters()).device
    except Exception as exc:
        import traceback
        logger.error("[eval] FATAL during model load: %s", exc)
        traceback.print_exc()
        _write_stub_output(
            args.output, label, args.tasks, args.episodes_per_task,
            args.base_seed, args.base_model, args.adapter, args.env_url,
            error=f"model_load: {exc}",
        )
        return 2

    try:
        async_client = PageZeroEnvClient(base_url=args.env_url)
        sync_client = async_client.sync()
        sync_client.__enter__()
    except Exception as exc:
        import traceback
        logger.error(
            "[eval] FATAL during env connect to %s: %s",
            args.env_url, exc,
        )
        traceback.print_exc()
        _write_stub_output(
            args.output, label, args.tasks, args.episodes_per_task,
            args.base_seed, args.base_model, args.adapter, args.env_url,
            error=f"env_connect: {exc}",
        )
        return 3

    client = sync_client
    try:
        results: List[EpisodeResult] = []
        for tid in args.tasks:
            for ep in range(int(args.episodes_per_task)):
                seed = args.base_seed + ep * 7919 + (abs(hash(tid)) % 9999)
                logger.info("[eval] task=%s ep=%d seed=%d", tid, ep, seed)
                t0 = time.time()
                try:
                    res = run_episode(
                        client=client,
                        model=model,
                        tokenizer=tokenizer,
                        task_id=tid,
                        seed=seed,
                        max_turns=args.max_turns,
                        max_new_tokens=args.max_new_tokens,
                        device=device,
                    )
                except Exception as exc:
                    logger.warning("[eval] failed task=%s ep=%d: %s", tid, ep, exc)
                    res = EpisodeResult(
                        task_id=tid, seed=seed, total_reward=0.0, num_steps=0,
                        is_resolved=False, done_cause="error", diagnose_count=0,
                    )
                logger.info(
                    "[eval]   total_reward=%+.2f steps=%d resolved=%s cause=%s (%.1fs)",
                    res.total_reward, res.num_steps, res.is_resolved,
                    res.done_cause, time.time() - t0,
                )
                results.append(res)

        agg = aggregate(results)
        agg["label"] = label
        agg["base_model"] = args.base_model
        agg["adapter"] = args.adapter
        agg["env_url"] = args.env_url
        agg["episodes_per_task"] = int(args.episodes_per_task)
        agg["base_seed"] = int(args.base_seed)

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(agg, f, indent=2, default=str)

        ov = agg["overall"]
        logger.info(
            "[eval] DONE label=%s n=%s reward_mean=%.3f resolved_rate=%.3f → %s",
            label, ov.get("n_episodes"), ov.get("reward_mean", 0.0),
            ov.get("resolved_rate", 0.0), args.output,
        )
    # SyncEnvClient.__exit__ already closes the websocket and stops the loop.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
