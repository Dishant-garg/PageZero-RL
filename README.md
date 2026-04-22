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

# Pagezero Environment

A simple test environment that echoes back messages. Perfect for testing the env APIs as well as demonstrating environment usage patterns.

## Quick Start

The simplest way to use the Pagezero environment is through the `PagezeroEnv` class:

```python
from PageZero import PagezeroAction, PagezeroEnv

try:
    # Create environment from Docker image
    PageZeroenv = PagezeroEnv.from_docker_image("PageZero-env:latest")

    # Reset
    result = PageZeroenv.reset()
    print(f"Reset: {result.observation.echoed_message}")

    # Send multiple messages
    messages = ["Hello, World!", "Testing echo", "Final message"]

    for msg in messages:
        result = PageZeroenv.step(PagezeroAction(message=msg))
        print(f"Sent: '{msg}'")
        print(f"  → Echoed: '{result.observation.echoed_message}'")
        print(f"  → Length: {result.observation.message_length}")
        print(f"  → Reward: {result.reward}")

finally:
    # Always clean up
    PageZeroenv.close()
```

That's it! The `PagezeroEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t PageZero-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
**PagezeroAction**: Contains a single field
- `message` (str) - The message to echo back

### Observation
**PagezeroObservation**: Contains the echo response and metadata
- `echoed_message` (str) - The message echoed back
- `message_length` (int) - Length of the message
- `reward` (float) - Reward based on message length (length × 0.1)
- `done` (bool) - Always False for echo environment
- `metadata` (dict) - Additional info like step count

### Reward
The reward is calculated as: `message_length × 0.1`
- "Hi" → reward: 0.2
- "Hello, World!" → reward: 1.3
- Empty message → reward: 0.0

## Advanced Usage

### Connecting to an Existing Server

If you already have a Pagezero environment server running, you can connect directly:

```python
from PageZero import PagezeroEnv

# Connect to existing server
PageZeroenv = PagezeroEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = PageZeroenv.reset()
result = PageZeroenv.step(PagezeroAction(message="Hello!"))
```

Note: When connecting to an existing server, `PageZeroenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from PageZero import PagezeroAction, PagezeroEnv

# Connect with context manager (auto-connects and closes)
with PagezeroEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(PagezeroAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    PageZeroEnvironment,  # Pass class, not instance
    PageZeroAction,
    PageZeroObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from PageZero import PagezeroAction, PagezeroEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with PagezeroEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(PagezeroAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/PageZero_environment.py
```

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## How to Run PageZero SRE Environment

This codebase has been updated to support a full-stack SRE training environment. Follow these steps to set up the infrastructure and run the Gym environment.

### 1. Infrastructure Setup (Docker)

The environment requires a running Docker stack (PostgreSQL, Redis, and a Flask App).

```bash
# Start the full-stack infrastructure
# Note: Requires sudo if user is not in the 'docker' group
sudo docker compose up -d --build

# Wait ~15 seconds for seeding to complete
sleep 15
```

### 3. Start the OpenEnv Server

Expose the Gym interface over HTTP/WebSocket for the agent.

```bash
# Run from project root
source .venv/bin/activate
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Verify the Connection

Run the smoke-test to ensure the server logic correctly interfaces with the Docker containers.

```bash
python3 verify.py --verbose
```

### 5. Training with GRPO

To train an agent using the Group Relative Policy Optimization (GRPO) algorithm:

```bash
python train.py \
  --model-id Qwen/Qwen3-0.6B \
  --dataset-size 50 \
  --max-turns 15 \
  --vllm-mode colocate
```

## Project Structure

```
PageZero-RL/
├── docker/                # Dockerfiles and seed scripts
├── server/
│   ├── PageZero_environment.py  # Core Gym logic
│   ├── stack_backend.py   # Docker interface
│   ├── executor.py        # Tool execution router
│   ├── app.py             # FastAPI entry point
│   └── ...                # Judge, Designer, Curriculum
├── models.py              # Action/Observation schemas
├── train.py               # GRPO Training script
├── verify.py              # Backend smoke tests
└── docker-compose.yml     # Infrastructure orchestration
```
