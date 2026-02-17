# microGPT Cerebral Lab

A real-time visualizer for understanding how a tiny GPT learns, token by token.

This project wraps the original single-file algorithm in `/reference/microgpt_original.py` with:
- an instrumented engine (`/backend/engine.py`),
- a FastAPI + WebSocket backend (`/backend/app.py`),
- a browser UI (`/frontend`) for live computation graph, attention, probabilities, and loss.

## Why this exists

The original `microgpt.py` is perfect for conceptual clarity, but everything happens in text logs.
This UI makes forward pass, attention, loss, backprop, and optimizer steps observable while training runs.

## Quickstart

```bash
cd /Users/sarvesh/code/microgpt-ui
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## CI on self-hosted runner

This repository ships a CI workflow in `.github/workflows/ci.yml` that runs on
`self-hosted` runners only. It runs a backend smoke check and Python module compile.

### Runner setup (GitHub UI)

1. In GitHub, open **Settings → Actions → Runners**.
2. Click **New self-hosted runner**.
3. Select the runner operating system (for this project, Linux is recommended).
4. Copy the generated **Configure** commands and run them on your machine inside a fresh folder.
5. Install service mode and start it so it remains online:
   ```bash
   sudo ./svc.sh install
   sudo ./svc.sh start
   ```
6. Go back to the repository actions page and confirm the runner appears as online.

### Required software on the runner

- `git`
- `python3` and `pip` (3.11+ recommended)
- `bash`
- Optional: `docker` (workflow skips docker checks if absent)
- Optional: `gcc`/`build-essential` for platform-specific dependency builds

### Linux setup example

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip curl ca-certificates
```

### macOS setup example

```bash
xcode-select --install
brew install git python@3.11 curl
```

### Registering the runner manually

After downloading the runner package and configuring, validate with:

```bash
./config.sh --url <repo_url> --token <token>
./svc.sh install
./svc.sh start
./run.sh
```

### Local CI verification on self-hosted host

Before pushing, replicate the workflow environment manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python - <<'PY'
from backend.app import app
from fastapi.testclient import TestClient

client = TestClient(app)
start = client.post("/api/session/start", json={"n_embd":16, "n_head":4, "n_layer":1, "block_size":8, "learning_rate":0.01, "num_steps":50, "temperature":0.5, "seed":42})
start.raise_for_status()
sid = start.json()["session_id"]
client.post(f"/api/session/{sid}/step").raise_for_status()
sample = client.post(f"/api/session/{sid}/sample", json={"num_samples":1, "temperature":0.6})
sample.raise_for_status()
print("Local self-hosted smoke test passed")
PY
python3 -m compileall -q backend
```

## What to explore

1. Start a session with tiny defaults and run single steps.
2. Watch the computation graph pulse for each train step.
3. Inspect attention weights (layer 0, head 0) for the latest token.
4. Compare top token probabilities vs. target token.
5. Generate inference samples at different temperatures.
6. Reset a session to return to step 0.
7. Use keyboard shortcuts for fast iteration.
8. Export training traces as JSON for offline analysis.

## API surface

- `POST /api/session/start` create a new model session
- `POST /api/session/{id}/step` run one training step
- `POST /api/session/{id}/run` run N steps asynchronously
- `POST /api/session/{id}/pause` pause an active run
- `POST /api/session/{id}/reset` reset the model state to initial weights for the current session config
- `POST /api/session/{id}/sample` sample names from current model state
- `GET /api/sessions` list active sessions, steps, and websocket client counts
- `WS /ws/{id}` live event stream for train events and run status

## UI features

- Graph Pulse: shows forward/backward stages per training step.
- Loss Timeline: rolling chart + rolling min/max/latest annotations.
- Training Progress: progress bar and remaining steps.
- Attention Constellation: latest token attention heatmap.
- Probability Panel: top-k token distribution on current token.
- Trace Export: download session traces as JSON from the current browser session.

## Keyboard shortcuts

- `N`: start a new session
- `S`: run a single training step
- `R`: run multiple steps
- `P`: pause running session
- `C`: generate samples
- `Enter`: start a new session

## Notes

- Data defaults to `input.txt`; if missing, it auto-downloads the same names dataset used by the original script.
- This code is intentionally small and educational, not optimized for speed.
