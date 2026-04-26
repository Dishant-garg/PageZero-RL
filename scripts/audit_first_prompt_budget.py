#!/usr/bin/env python3
"""
Estimate size of the *first* model turn context (no TRL / no torch).

TRL may add template overhead; this script only measures the strings we control
in ``train.py`` so you can see whether alert text is likely to be truncated by
a small ``max_prompt_length`` if your stack sets one.

Usage (repo root):

  python scripts/audit_first_prompt_budget.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_system_prompt() -> str:
    src = (ROOT / "train.py").read_text(encoding="utf-8")
    m = re.search(r'SYSTEM_PROMPT = """(.*?)"""', src, re.DOTALL)
    if not m:
        raise RuntimeError("Could not parse SYSTEM_PROMPT from train.py")
    return m.group(1)


def _fake_format_observation(*, tool_output: str, alert: str) -> str:
    """Mirror the non-prior part of PageZeroToolEnv._format_observation (approx)."""
    incident_header = (
        "INCIDENT CONTEXT:\n"
        "  task_id: task_3\n"
        "  scenario: 🚨 ALERT: CRITICAL: Redis memory usage > 95%…\n\n"
    )
    return (
        f"{incident_header}"
        f"TOOL OUTPUT:\n{tool_output}\n\n"
        f"CURRENT ALERTS:\n{alert}\n\n"
        "SLA STATUS: OK\n"
        "REVENUE LOST: $0.0\n"
        "DOWNTIME: 0.0 minutes\n"
        "PHASE: None (fix_step=False, repeat=1)\n"
        "STEP REWARD: +0.0000\n"
        "STEP: 0/15"
    )


def main() -> None:
    system = _load_system_prompt()
    user = "Diagnose and fix this production incident."
    long_alert = (
        "CRITICAL: Redis memory usage > 95%, OOM errors in app logs — "
        "check redis_info and flush orphaned keys"
    )
    tool_out = f"🚨 ALERT: {long_alert}"
    obs_block = _fake_format_observation(tool_output=tool_out, alert=long_alert)

    # Rough tokenizer-free budget: Qwen-ish English often ~3–4 chars / token.
    def approx_tokens(chars: int) -> tuple[float, float]:
        return chars / 4.0, chars / 3.0

    parts = [
        ("SYSTEM_PROMPT", system),
        ("user (static)", user),
        ("first observation block (approx)", obs_block),
    ]
    total_chars = sum(len(t) for _, t in parts)
    lo, hi = approx_tokens(total_chars)

    print("First-turn string budget (chars, approx token range @ 3–4 chars/token)")
    print("=" * 72)
    for name, t in parts:
        a, b = approx_tokens(len(t))
        print(f"{name:42s} chars={len(t):5d}  ~tokens {a:5.0f}–{b:5.0f}")
    print("-" * 72)
    print(f"{'TOTAL (concat)':42s} chars={total_chars:5d}  ~tokens {lo:5.0f}–{hi:5.0f}")
    print()
    print(
        "Interpretation: if your TRL/HF stack uses a small max_prompt_length "
        "(e.g. 2k) *including* chat-template special tokens, you could still "
        "truncate on later turns when tool outputs grow — but a typical first "
        "reset observation is usually sub‑1k tokens unless logs are huge."
    )


if __name__ == "__main__":
    main()
