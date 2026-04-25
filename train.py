"""
GRPO Training Script — PageZero SRE Agent

Migrated to TRL OpenEnv's environment_factory flow:
- No manual generate_rollout_completions loop
- No custom rollout_func token plumbing
- Trainer drives multi-turn tool calling automatically

Setup (2 terminals on H100):

  # Install
  pip install -e ".[train]"

  # Terminal 1: OpenEnv server (adversarial mode with Claude judge)
  GYM_MODE=adversarial LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-ant-... uv run server

  # Terminal 2: GRPO training
  python train.py --vllm-mode colocate
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Help PyTorch reuse fragmented GPU memory (critical for TRL+vLLM colocate on 80GB)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

try:
    from .client import PageZeroEnvClient
    from .models import PageZeroAction
except (ImportError, ValueError):
    from client import PageZeroEnvClient
    from models import PageZeroAction


# ---- Optional TRL/vLLM compatibility patch for older stacks ----
_orig_vllm_gen = None


def _patch_vllm_generate(trainer: GRPOTrainer) -> None:
    """Wrap vLLM generate to normalize logprobs shape on older TRL stacks."""
    global _orig_vllm_gen
    if _orig_vllm_gen is not None or not hasattr(trainer, "vllm_generation"):
        return

    _orig_vllm_gen = trainer.vllm_generation.generate

    def _wrapped_generate(**kwargs):
        result = _orig_vllm_gen(**kwargs)
        prompt_ids, completion_ids, logprobs, *rest = result
        if logprobs and logprobs[0] and isinstance(logprobs[0][0], float):
            logprobs = [[[lp] for lp in seq] for seq in logprobs]
        return (prompt_ids, completion_ids, logprobs, *rest)

    trainer.vllm_generation.generate = _wrapped_generate


def patch_trl_vllm_compat() -> None:
    """Apply TRL/vLLM compatibility patches if needed by this TRL build."""
    _orig_train = GRPOTrainer.train

    def _patched_train(self, *args, **kwargs):
        _patch_vllm_generate(self)
        return _orig_train(self, *args, **kwargs)

    GRPOTrainer.train = _patched_train


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a Staff SRE on-call. Diagnose and fix the cascading incident across the Application, PostgreSQL, and Redis cache.

Use tools to investigate and remediate.
Follow this workflow:
1) Triage alerts and service health.
2) Investigate app/db/cache signals.
3) Apply precise fixes.
4) Verify recovery.
5) Call done when fully restored.

Prefer precise arguments for tools requiring params.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for PageZero SRE agent")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B", help="Agent model to fine-tune")
    parser.add_argument("--env-url", default="http://localhost:8000", help="OpenEnv server URL")
    parser.add_argument("--dataset-size", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--max-turns", type=int, default=15, help="Max tool-calling turns per episode")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max tokens across one episode")
    parser.add_argument("--num-generations", type=int, default=8, help="G for GRPO")
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1, help="Max GRPO training steps (-1 = auto)")
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--push-to-hub", action="store_true", help="Push model to HF Hub after training")
    parser.add_argument("--hub-repo", default=None, help="HF Hub repo, e.g. your-name/pagezero-sre-agent")
    parser.add_argument(
        "--vllm-mode",
        choices=("colocate", "server"),
        default="colocate",
        help="vLLM mode: colocate (1 GPU) or server (separate vLLM process)",
    )
    parser.add_argument("--vllm-server-url", default="http://localhost:8001", help="vLLM server URL (server mode)")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5, help="vLLM GPU memory fraction (0.0-1.0)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--report-to", default="none", choices=("tensorboard", "wandb", "none"))
    parser.add_argument("--reward-log", default="reward_log.csv", help="CSV file for per-episode reward logging")
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return name.replace("/", "-")


class PageZeroToolEnv:
    """Environment wrapper for GRPOTrainer(environment_factory=...).

    Public methods (except reset) are exposed as tools automatically.
    """

    BASE_URL: str = "http://localhost:8000"
    _instances: list["PageZeroToolEnv"] = []

    def __init__(self):
        self.client = PageZeroEnvClient(base_url=self.BASE_URL)
        self.total_reward = 0.0
        self.diagnosis_reward = 0.0
        self.fix_reward = 0.0
        self.is_done = False
        self._episode_logged = False
        self._last_hint = ""
        self._last_observation_text = ""
        self.__class__._instances.append(self)

    @classmethod
    def close_all(cls) -> None:
        for env in cls._instances:
            try:
                env.client.close()
            except Exception:
                pass
        cls._instances.clear()

    def reset(self, **kwargs) -> str | None:
        """Reset environment state for a new episode.

        Returns:
            Initial observation text appended to the user prompt.
        """
        result = self.client.reset()
        self.total_reward = 0.0
        self.diagnosis_reward = 0.0
        self.fix_reward = 0.0
        self.is_done = bool(result.done)
        self._episode_logged = False

        obs = result.observation
        self._last_hint = getattr(obs, "hint", "") or ""
        self._last_observation_text = self._format_observation(obs, reward=0.0)
        return self._last_observation_text

    def _format_observation(self, obs, reward: float) -> str:
        tool_output = getattr(obs, "tool_output", "") or ""
        alerts = getattr(obs, "active_alerts", []) or []
        alert_text = "\n".join(alerts) if alerts else "None"
        sla_status = getattr(obs, "sla_status", "OK")
        revenue_loss = getattr(obs, "revenue_loss_usd", 0.0)
        downtime_minutes = getattr(obs, "downtime_minutes", 0.0)
        step = getattr(obs, "step", 0)
        max_steps = getattr(obs, "max_steps", 15)
        hint = getattr(obs, "hint", "") or ""

        text = (
            f"{tool_output}\n\n"
            f"CURRENT ALERTS:\n{alert_text}\n\n"
            f"SLA STATUS: {sla_status}\n"
            f"REVENUE LOST: ${revenue_loss}\n"
            f"DOWNTIME: {downtime_minutes} minutes\n"
            f"STEP REWARD: {reward:.4f}\n"
            f"STEP: {step}/{max_steps}"
        )
        if hint:
            text += f"\nHINT: {hint}"
        return text

    def _run_tool(self, tool: str, args: Dict[str, Any]) -> str:
        if self.is_done and tool != "done":
            raise ValueError("Episode already done. No further tools are allowed.")

        result = self.client.step(PageZeroAction(tool=tool, args=args))
        reward = float(result.reward or 0.0)
        self.total_reward += reward
        self.is_done = bool(result.done)

        if tool.startswith(("pg_cancel", "pg_create", "pg_vacuum", "docker_restart", "rollback_deploy", "redis_flush_db")):
            self.fix_reward = reward
        else:
            self.diagnosis_reward = reward

        obs = result.observation
        self._last_hint = getattr(obs, "hint", "") or ""
        self._last_observation_text = self._format_observation(obs, reward=reward)
        return self._last_observation_text

    # --- Triage ---
    def check_alerts(self) -> str:
        """Check active incident alerts.

        Returns:
            Current alert information.
        """
        return self._run_tool("check_alerts", {})

    def get_service_metrics(self, service: str = "app") -> str:
        """Get service metrics.

        Args:
            service: Service name such as app, redis, or postgres.

        Returns:
            Service metrics.
        """
        return self._run_tool("get_service_metrics", {"service": service})

    def get_error_rate(self) -> str:
        """Get aggregate application error rate.

        Returns:
            Error rate summary.
        """
        return self._run_tool("get_error_rate", {})

    # --- Application ---
    def read_app_logs(self, lines: int = 200) -> str:
        """Read recent application logs.

        Args:
            lines: Number of log lines to fetch.

        Returns:
            Log output.
        """
        return self._run_tool("read_app_logs", {"lines": lines})

    def search_logs(self, pattern: str) -> str:
        """Search logs for a text pattern.

        Args:
            pattern: Search pattern.

        Returns:
            Matching log lines.
        """
        return self._run_tool("search_logs", {"pattern": pattern})

    def get_recent_deploys(self) -> str:
        """List recent deployments.

        Returns:
            Deployment history.
        """
        return self._run_tool("get_recent_deploys", {})

    def rollback_deploy(self) -> str:
        """Rollback latest deployment.

        Returns:
            Rollback status.
        """
        return self._run_tool("rollback_deploy", {})

    def curl_endpoint(self, url: str) -> str:
        """Curl an endpoint for health/behavior check.

        Args:
            url: Endpoint URL.

        Returns:
            HTTP response summary.
        """
        return self._run_tool("curl_endpoint", {"url": url})

    # --- PostgreSQL ---
    def pg_stat_activity(self) -> str:
        """Inspect PostgreSQL active sessions.

        Returns:
            Active query/session information.
        """
        return self._run_tool("pg_stat_activity", {})

    def pg_locks(self) -> str:
        """Inspect PostgreSQL lock state.

        Returns:
            Lock diagnostics.
        """
        return self._run_tool("pg_locks", {})

    def pg_explain_analyze(self, query: str) -> str:
        """Run EXPLAIN ANALYZE on a SQL query.

        Args:
            query: SQL query text.

        Returns:
            Query plan and timing.
        """
        return self._run_tool("pg_explain_analyze", {"query": query})

    def pg_stat_statements(self) -> str:
        """Inspect pg_stat_statements.

        Returns:
            Statement-level performance stats.
        """
        return self._run_tool("pg_stat_statements", {})

    def pg_cancel_query(self, pid: int) -> str:
        """Cancel a PostgreSQL backend query.

        Args:
            pid: Process id to cancel.

        Returns:
            Cancellation result.
        """
        return self._run_tool("pg_cancel_query", {"pid": pid})

    def pg_create_index(self, table: str, column: str) -> str:
        """Create an index on table(column).

        Args:
            table: Table name.
            column: Column name.

        Returns:
            Index creation result.
        """
        return self._run_tool("pg_create_index", {"table": table, "column": column})

    def pg_vacuum(self, table: str) -> str:
        """Run VACUUM on a table.

        Args:
            table: Table name.

        Returns:
            Vacuum status.
        """
        return self._run_tool("pg_vacuum", {"table": table})

    def pg_show_tables(self) -> str:
        """List PostgreSQL tables.

        Returns:
            Table list.
        """
        return self._run_tool("pg_show_tables", {})

    # --- Redis ---
    def redis_info(self) -> str:
        """Get Redis INFO diagnostics.

        Returns:
            Redis INFO output.
        """
        return self._run_tool("redis_info", {})

    def redis_slowlog(self) -> str:
        """Inspect Redis slowlog entries.

        Returns:
            Slowlog output.
        """
        return self._run_tool("redis_slowlog", {})

    def redis_keys(self, pattern: str = "*") -> str:
        """List Redis keys by pattern.

        Args:
            pattern: Redis key pattern.

        Returns:
            Matching keys.
        """
        return self._run_tool("redis_keys", {"pattern": pattern})

    def redis_flush_db(self) -> str:
        """Flush Redis DB.

        Returns:
            Flush result.
        """
        return self._run_tool("redis_flush_db", {})

    def redis_get_key(self, key: str) -> str:
        """Get value of a Redis key.

        Args:
            key: Redis key.

        Returns:
            Key value.
        """
        return self._run_tool("redis_get_key", {"key": key})

    # --- Infrastructure ---
    def docker_ps(self) -> str:
        """List Docker containers.

        Returns:
            Container list.
        """
        return self._run_tool("docker_ps", {})

    def docker_stats(self, container: str) -> str:
        """Get Docker resource stats for a container.

        Args:
            container: Container name.

        Returns:
            Stats output.
        """
        return self._run_tool("docker_stats", {"container": container})

    def docker_restart(self, container: str) -> str:
        """Restart a container.

        Args:
            container: Container name.

        Returns:
            Restart result.
        """
        return self._run_tool("docker_restart", {"container": container})

    def docker_logs(self, container: str) -> str:
        """Read logs for a container.

        Args:
            container: Container name.

        Returns:
            Container logs.
        """
        return self._run_tool("docker_logs", {"container": container})

    def check_disk_usage(self) -> str:
        """Check disk usage on host/container runtime.

        Returns:
            Disk usage summary.
        """
        return self._run_tool("check_disk_usage", {})

    # --- Resolution ---
    def diagnose_root_cause(self, root_cause: str) -> str:
        """Record a root-cause diagnosis.

        Args:
            root_cause: One-sentence root-cause summary.

        Returns:
            Acknowledgement from environment.
        """
        return self._run_tool("diagnose_root_cause", {"root_cause": root_cause})

    def done(self) -> str:
        """Mark incident handling as complete.

        Returns:
            Final environment message.
        """
        return self._run_tool("done", {})


def plot_rewards(csv_path: Path, out_path: Path | None = None) -> None:
    """Plot reward curves from the CSV log."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes, totals, diags, fixes = [], [], [], []
    with open(csv_path) as f:
        reader = __import__("csv").reader(f)
        next(reader)
        for row in reader:
            episodes.append(int(row[0]))
            totals.append(float(row[1]))
            diags.append(float(row[2]))
            fixes.append(float(row[3]))

    if not episodes:
        logger.warning("No episodes to plot")
        return

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    window = min(10, len(episodes))

    def rolling_avg(vals):
        return [sum(vals[max(0, i - window) : i + 1]) / min(i + 1, window) for i in range(len(vals))]

    rolling = rolling_avg(totals)

    ax1.plot(episodes, totals, alpha=0.25, color="blue", marker="o", markersize=3, label="Per episode")
    ax1.plot(episodes, rolling, color="blue", linewidth=2.5, label=f"Rolling avg ({window})")

    import numpy as np

    z = np.polyfit(episodes, totals, 1)
    trend = np.poly1d(z)
    ax1.plot(
        episodes,
        trend(episodes),
        color="red",
        linewidth=1.5,
        linestyle="--",
        label=f"Trend ({'↑' if z[0] > 0 else '↓'} {abs(z[0]):.3f}/ep)",
    )

    ax1.set_ylabel("Total Reward")
    ax1.set_xlabel("Episode")
    ax1.set_title("PageZero SRE Agent — GRPO Training Reward Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax1.text(
        0.02,
        0.02,
        f"Episodes: {len(episodes)} | Final avg: {rolling[-1]:.2f} | Best: {max(totals):.2f}",
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    save_path = out_path or csv_path.with_suffix(".png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Reward plot saved to {save_path}")


def main() -> None:
    patch_trl_vllm_compat()
    args = parse_args()

    logger.info("=" * 60)
    logger.info("PageZero SRE Agent — GRPO Training (OpenEnv + TRL)")
    logger.info("=" * 60)
    logger.info(f"Agent model:    {args.model_id}")
    logger.info(f"Env URL:        {args.env_url}")
    logger.info(f"Episodes:       {args.dataset_size}")
    logger.info(f"Generations/G:  {args.num_generations}")
    logger.info(f"vLLM mode:      {args.vllm_mode}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    PageZeroToolEnv.BASE_URL = args.env_url

    # Conversational format works best for tool-calling agent training.
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Diagnose and fix this production incident."},
    ]
    dataset = Dataset.from_dict({"prompt": [prompt_messages for _ in range(args.dataset_size)]})

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_output_dir = Path("outputs") / f"pagezero-sre-grpo-{sanitize_name(args.model_id)}-{timestamp}"
    output_dir = Path(args.output_dir or default_output_dir)

    grpo_config = GRPOConfig(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        max_grad_norm=1.0,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=1,
        generation_batch_size=args.num_generations,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        max_tool_calling_iterations=args.max_turns,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        temperature=args.temperature,
        report_to=args.report_to,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_repo if args.push_to_hub else None,
        hub_strategy="every_save",
        save_total_limit=3,
        loss_type="dapo",
        mask_truncated_completions=True,
        beta=0.01,
        chat_template_kwargs={"enable_thinking": False},
    )

    import csv

    reward_log_path = output_dir / args.reward_log
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_counter = [0]
    all_rewards = []

    with open(reward_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "diagnosis_reward", "fix_reward", "timestamp"])

    def _log_episode(total_r: float, diag_r: float, fix_r: float) -> None:
        episode_counter[0] += 1
        all_rewards.append(total_r)
        with open(reward_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode_counter[0], total_r, diag_r, fix_r, datetime.now().isoformat()])

        n = len(all_rewards)
        mean_all = sum(all_rewards) / n
        last10 = all_rewards[-10:]
        mean_10 = sum(last10) / len(last10)
        best = max(all_rewards)

        logger.info(
            f"Episode {episode_counter[0]}: reward={total_r:.2f} "
            f"(diag={diag_r:.2f}, fix={fix_r:.2f}) | "
            f"mean={mean_all:.2f}, mean(10)={mean_10:.2f}, best={best:.2f}"
        )

    def reward_total(completions=None, environments=None, **kwargs) -> list[float]:
        if not environments:
            count = len(completions) if completions is not None else 0
            return [0.0 for _ in range(count)]

        values: list[float] = []
        for env in environments:
            total = float(getattr(env, "total_reward", 0.0))
            diag = float(getattr(env, "diagnosis_reward", 0.0))
            fix = float(getattr(env, "fix_reward", 0.0))
            values.append(total)

            if not getattr(env, "_episode_logged", False):
                _log_episode(total, diag, fix)
                env._episode_logged = True

        return values

    def reward_diagnosis(completions=None, environments=None, **kwargs) -> list[float]:
        if not environments:
            count = len(completions) if completions is not None else 0
            return [0.0 for _ in range(count)]
        return [float(getattr(env, "diagnosis_reward", 0.0)) for env in environments]

    def reward_fix(completions=None, environments=None, **kwargs) -> list[float]:
        if not environments:
            count = len(completions) if completions is not None else 0
            return [0.0 for _ in range(count)]
        return [float(getattr(env, "fix_reward", 0.0)) for env in environments]

    try:
        from peft import LoraConfig
    except Exception as e:
        raise ImportError(
            "peft is required to run train.py with LoRA. "
            "Install a compatible peft/transformers combination."
        ) from e

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        reward_funcs=[reward_total, reward_diagnosis, reward_fix],
        train_dataset=dataset,
        args=grpo_config,
        peft_config=peft_config,
        environment_factory=PageZeroToolEnv,
    )

    logger.info("Starting GRPO training...")
    logger.info(f"Using {args.num_generations} environment rollouts per prompt")

    try:
        trainer.train()
    finally:
        PageZeroToolEnv.close_all()
        try:
            plot_rewards(reward_log_path, output_dir / "reward_plot.png")
        except Exception as e:
            logger.warning(f"Could not generate reward plot: {e}")

    trainer.save_model(str(output_dir))
    logger.info(f"Model saved to {output_dir}")
    logger.info(f"Reward log: {reward_log_path}")

    if args.push_to_hub and args.hub_repo:
        trainer.push_to_hub()
        logger.info(f"Model pushed to https://huggingface.co/{args.hub_repo}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
