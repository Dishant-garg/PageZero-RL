# PageZero Makefile — orchestrates stack, OpenEnv server, and training

.PHONY: up down rebuild verify server train logs ps clean

# ── Docker Stack ──────────────────────────────────────────────────────────────

up:
	@echo "╔══════════════════════════════════╗"
	@echo "║  Starting PageZero Docker Stack  ║"
	@echo "╚══════════════════════════════════╝"
	docker compose up -d --build
	@echo "Waiting for healthchecks..."
	@sleep 5
	docker compose ps

down:
	docker compose down

rebuild:
	docker compose down
	docker compose up -d --build --force-recreate
	@sleep 5
	docker compose ps

logs:
	docker compose logs -f

ps:
	docker compose ps

# ── Verification ─────────────────────────────────────────────────────────────

verify:
	@echo "Running stack verification..."
	python verify.py --verbose

verify-quiet:
	python verify.py

# ── OpenEnv Server ───────────────────────────────────────────────────────────

server:
	@echo "Starting OpenEnv server on :8000..."
	uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

server-prod:
	uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 2

# ── Training ─────────────────────────────────────────────────────────────────

train:
	python train.py --model-id Qwen/Qwen3-0.6B --dataset-size 50 --max-turns 15

train-quick:
	python train.py --model-id Qwen/Qwen3-0.6B --dataset-size 10 --max-turns 5 --vllm-mode server

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f agent_transcripts.jsonl reward_log.csv
