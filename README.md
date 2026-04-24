---
title: Pagezero Environment Server
emoji: ⌨️
colorFrom: green
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

<div align="center">
  <h1>🚨 PageZero</h1>
  <h3>An Autonomous SRE Reinforcement Learning Benchmark</h3>
  <br />
</div>

**PageZero** is an advanced, fully dockerized Reinforcement Learning (RL) benchmark environment built on top of [OpenEnv](https://github.com/openenv-core). It is designed to train and evaluate AI agents on real-world Site Reliability Engineering (SRE) tasks. 

Unlike simple "text-only" reasoning benchmarks, PageZero spins up a live, multi-container production stack. Agents must write and execute real bash commands, PostgreSQL queries, and Redis commands to triage, diagnose, and resolve escalating production incidents.

---

## 🏗️ Architecture

The environment simulates a standard three-tier web application architecture:
1.  **Flask Application** (`pagezero-app-1`)
2.  **PostgreSQL Database** (`pagezero-postgres-1`)
3.  **Redis Cache** (`pagezero-redis-1`)

The Gym Environment evaluates agents using two signals:
*   **Terminal Health Check**: A programmatic, deterministic verification script that tests if the application stack is actually functional and SLA violations have ceased.
*   **LLM Judge via Gemini**: A deterministic "Principal SRE" evaluator that provides dense, per-step reward shaping based on the agent's adherence to the proper SRE workflow (Triage → Investigate → Diagnose → Fix → Verify).

---

## 🚀 Quick Start

### 1. Requirements
*   Docker & Docker Compose
*   Python 3.10+
*   `uv` package manager (recommended)
*   A Gemini API Key (only required for LLM Judge/Scenario generation)

### 2. Setup the Environment

Configure your environment variables:
```bash
cp .env.example .env
# Open .env and add your GEMINI_API_KEY
```

Start the live production infrastructure:
```bash
docker compose up -d
```

Verify everything is working natively without an LLM:
```bash
uv run python verify.py --verbose
```
*This will run a 16-step smoke test that injects faults and verifies the Python backend can catch them.*

---

## 🤖 Running Inference (The SRE Agent)

To watch an autonomous agent respond to an incident in real-time, you can use the built-in inference script. This script hooks up `gemini-2.5-flash` to the environment and manages the tool-loop.

```bash
uv run python play.py
```

**What it does:**
1. Triggers an incident based on the Curriculum logic (e.g., Cache drops, CPU spikes, DB Locks).
2. The agent attempts to diagnose and fix it in under 15 steps.
3. If the agent gets stuck in a loop, `play.py` automatically implements "Loop-Breaking" interventions.
4. Outputs an evaluation summary string `[END] success=true score=0.85...`

---

## 🧠 Training 

PageZero is designed out-of-the-box for **GRPO (Group Relative Policy Optimization)** training via `trl` and `vllm`. 

To train your own SRE reasoning model locally:
```bash
uv run python train.py --model-id Qwen/Qwen2.5-Coder-7B-Instruct --dataset-size 500
```
*(Ensure you have adequate GPU VRAM before running local unsloth/VLLM GRPO pipelines).*

---

## 📚 Built-in RL Mechanisms

PageZero contains multiple advanced RL environment features that prevent overfitting and guarantee agent robustness:

*   **Curriculum Learning (`server/curriculum.py`)**: Agents start by solving obvious "Warmup" tasks (e.g., heavy query taking up CPU). As they succeed, the environment unlocks "Medium" and "Hard" scenarios (e.g., Cascading Redis OOM causing PostgreSQL connection exhaustion).
*   **Schema Drift (`server/schema_drift.py`)**: To prevent agents from memorizing database schemas, the environment randomly renames tables or keys (e.g., mutating `orders` table to `tx_log`) requiring the agent to defensively execute `\dt` or `redis_keys *` first.
*   **Programmatic SLA Signal**: Agents are penalized mathematically simply based on simulated "Downtime USD" per minute, enforcing speed.

---

## 📂 Project Structure

```text
PageZero/
├── docker-compose.yml           # Live infrastructure definition
├── play.py                      # Multi-turn autonomous agent inference
├── train.py                     # GRPO RL training pipeline
├── verify.py                    # Deterministic backend testing
├── models.py                    # Pydantic schemas for Action/Observation
├── .env                         # Ports, DB Auth, and Gemini Keys
└── server/
    ├── PageZero_environment.py  # Core OpenEnv Gym implementation
    ├── stack_backend.py         # Subprocess bridge to docker exec
    ├── executor.py              # Router for agent tool calls
    ├── llm_judge.py             # Trajectory Evaluator
    ├── llm_designer.py          # Scenario generation pools
    └── schema_drift.py          # Adversarial workspace mutations
```
