---
title: "Building PageZero: Training AI to be an Elite SRE"
date: "2026-04-26"
author: "PageZero Engineering Team"
---

# Building PageZero: Training AI to be an Elite SRE

Have you ever wondered what happens when a live production server catches fire at 3 AM? The alarms go off, CPUs max out, databases lock up, and an on-call engineer has to blindly wake up and dive into a terminal to save the day.

We asked ourselves: **Can we train an autonomous AI agent to do this instead?**

Say hello to **PageZero** — our fully dockerized, reinforcement learning benchmark that simulates real-world Site Reliability Engineering (SRE) incidents and trains LLM agents to fix them using actual bash commands, SQL queries, and diagnostic tools.

Here’s the story of how we built it, what we used, the bumps along the way, and where we landed!

---

## The Tech Stack (What We Used)

We didn't just want a "text-only" reasoning puzzle. We wanted our agent to face a real app. So we gave it exactly that.
- **Infrastructure:** Docker & Docker Compose to spin up a live three-tier web application (Flask, PostgreSQL, Redis).
- **Training Environment:** [OpenEnv](https://github.com/openenv-core) wrapped around a standard Gym environment.
- **Evaluation Engine:** Google's Gemini-2.5-Flash acting as our "Principal SRE" Judge to grade the agent step-by-step.
- **RL Framework:** Group Relative Policy Optimization (GRPO) using `trl`, `vllm`, and `unsloth` for high-efficiency local and cloud training.
- **Language/Package Management:** Modern Python 3.10+ powered by `uv` for lightning-fast lockfiles and dependencies.

---

## The Timeline: How It All Came Together

Building a sandbox where an AI can safely but realistically battle outages was an intense ride. Here’s how our timeline played out:

### 1. Setting Up the Matrix (The Live Infrastructure)
We started by constructing the core environment using Docker to build a robust Flask backend communicating with Redis and Postgres. The goal here was straightforward: inject chaos (like dropping caches or spiking CPU) and expose the agent to the actual containerized services. We used OpenEnv to expose this infrastructure to the agent cleanly.

### 2. Enter the Judge (Hybrid Reward System)
We quickly learned that traditional binary reinforcement learning (1 for fix, 0 for failure) wasn't enough. Production incidents require *process*. Therefore, we wired up a **Hybrid Reward System**:
* **Deterministic Reward**: Core Python scripts that programmatically verify the application stack SLA recovery and uptime.
* **LLM Judge**: We used Gemini to hand out dense rewards step-by-step based on whether the agent followed proper SRE workflows (e.g., *Check the logs before blindly restarting services!*).

### 3. The `write_postmortem` Era
To enforce realistic documentation over raw fixing hacks, we introduced the Post-Mortem Incident Protocol. We mandated that agents couldn't just "fix it and bounce"; they had to formally write down the root cause. 

### 4. Training, Troubleshooting, & Scaling Up
We hooked our models up via local-to-tunnel endpoints to Google Colab and began training via GRPO. 

---

## What Worked Beautifully

*   **Dense LLM Judging:** Letting Gemini validate intermediate steps practically solved the sparse-reward RL problem. By rewarding a great SQL diagnostic query even if the ultimate fix failed, the agent incrementally improved at an incredible rate.
*   **Dockerized Realism:** Running actual Docker daemon processes over mock environments proved to be a masterstroke. The agent literally learns to `docker exec -it` and read real stack traces natively!
*   **Curriculum Learning Task Tiers:** Starting the agent on basic web server restarts and progressively opening the gates stopped the model from collapsing early in training. The tasks are heavily curated into 3 tiers:
    *   **Easy Tier (e.g., task_1)**: Simple point failures or latency bumps.
    *   **Medium Tier (e.g., task_6)**: Cascading failures needing connection diagnoses.
    *   **Hard Tier (e.g., task_11)**: Distributed system meltdowns spanning DB/Cache/Flask that need deep multi-stage analysis.

---

## What DID NOT Work (Our Pain Points)

*   **The "Guessing" Agent Loophole:** We originally found the agent realized it could earn points by repeatedly guessing the root cause via the `diagnose_root_cause` tool without doing any actual investigation first! We had to hard-code a "Diagnose-Overuse Cutoff" that immediately penalized the agent if it spammed the documentation tool.
*   **Connection and Timeout Nightmares:** Moving from local training to scaled-up GRPO on Colab GPUs using tunneling caused enormous environment timeout failures due to concurrency ceilings. Managing session limits reliably took massive engineering effort.

---

## The Results

We ran exhaustive baselines using deterministic SRE test cases offline. 

When observing the agent's behavior post-training, the **Learning Curve** showed a tremendous improvement. The agent evolved from blindly retrying dead commands to strategically issuing system checks (`top`, `ps aux`, `\dt`), interpreting the real stdout/stderr strings, and issuing precise rectifications. 

**Wait, what about Adversarial Training?**
We also successfully implemented background chaos networks that randomly disrupt latency and simulate race conditions while the agent learns. However, as noted below, due to infrastructure constraints, we could not train this to more than **4 episodes, 4 generations, and a 512 token length limit**. Therefore, the metrics you see below are *preliminary results* to show that the environment logic works! For fully rigorous training of the agent, we need significantly more resources.

*(Here are some screenshots from our metrics and terminal logs)*

![Reward Curve](plots/plot_01_overall_reward_curve.png)
*A steep upward climb! The agent rapidly stopped timing out and started scoring successful terminal gates.*

![Termination Reasons](plots/plot_05_termination_reason_distribution.png)
*Notice how early episodes ended in "diagnose_overuse" or "timeout" while later steps heavily skewed into successful accepted closures.*

---

## Wrapping Up 

Building PageZero was an experiment in giving AI actual skin in the game. We’ve come away with an incredibly capable pipeline for teaching models technical SRE workflows safely. With our fully stable GRPO loops, the ceiling for creating an elite 'On-Call Copilot' just skyrocketed. 

Next time the server drops dead at 3 AM... maybe PageZero can handle it.
