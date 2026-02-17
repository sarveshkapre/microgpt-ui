"""FastAPI service exposing instrumented microGPT training sessions."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.engine import MicroGPT, ModelConfig


ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"


class StartSessionRequest(BaseModel):
    n_embd: int = 16
    n_head: int = 4
    n_layer: int = 1
    block_size: int = 8
    learning_rate: float = 1e-2
    num_steps: int = 500
    temperature: float = 0.5
    seed: int = 42


class RunRequest(BaseModel):
    steps: int = Field(default=100, ge=1, le=2000)
    delay_ms: int = Field(default=150, ge=0, le=2000)


class SampleRequest(BaseModel):
    num_samples: int = Field(default=10, ge=1, le=100)
    temperature: float = Field(default=0.5, gt=0.0, le=2.0)


class SessionState:
    def __init__(self, engine: MicroGPT):
        self.engine = engine
        self.lock = asyncio.Lock()
        self.clients: set[WebSocket] = set()
        self.run_task: asyncio.Task[Any] | None = None
        self.running = False

    async def broadcast(self, payload: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for client in self.clients:
            try:
                await client.send_json(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            self.clients.discard(client)

    async def stop_run(self) -> None:
        self.running = False
        if self.run_task and not self.run_task.done():
            self.run_task.cancel()
            try:
                await self.run_task
            except Exception:
                pass
        self.run_task = None


app = FastAPI(title="microGPT Visualizer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, SessionState] = {}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/session/start")
async def start_session(req: StartSessionRequest) -> dict[str, Any]:
    if req.n_embd <= 0 or req.n_head <= 0:
        raise HTTPException(status_code=400, detail="n_embd and n_head must be positive integers")
    if req.n_head > req.n_embd:
        raise HTTPException(status_code=400, detail="n_head cannot be greater than n_embd")
    if req.n_embd % req.n_head != 0:
        raise HTTPException(
            status_code=400,
            detail="n_embd must be divisible by n_head",
        )

    cfg = replace(
        ModelConfig(),
        n_embd=req.n_embd,
        n_head=req.n_head,
        n_layer=req.n_layer,
        block_size=req.block_size,
        learning_rate=req.learning_rate,
        num_steps=req.num_steps,
        temperature=req.temperature,
        seed=req.seed,
    )
    engine = MicroGPT(cfg)
    session_id = str(uuid.uuid4())
    sessions[session_id] = SessionState(engine)
    return {"session_id": session_id, "metadata": engine.metadata()}


def _get_session(session_id: str) -> SessionState:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return session


@app.get("/api/session/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    session = _get_session(session_id)
    return {
        "session_id": session_id,
        "running": session.running,
        "metadata": session.engine.metadata(),
    }


@app.post("/api/session/{session_id}/step")
async def step_session(session_id: str) -> dict[str, Any]:
    session = _get_session(session_id)
    async with session.lock:
        event = session.engine.train_step()
    await session.broadcast(event)
    return event


async def _run_steps(session_id: str, steps: int, delay_ms: int) -> None:
    session = _get_session(session_id)
    try:
        for _ in range(steps):
            if not session.running:
                break
            if session.engine.step >= session.engine.cfg.num_steps:
                break
            async with session.lock:
                event = session.engine.train_step()
            await session.broadcast(event)
            if delay_ms:
                await asyncio.sleep(delay_ms / 1000.0)
    finally:
        session.running = False
        session.run_task = None
        await session.broadcast(
            {"type": "run_status", "running": False, "completed": session.engine.step >= session.engine.cfg.num_steps}
        )


@app.post("/api/session/{session_id}/run")
async def run_session(session_id: str, req: RunRequest) -> dict[str, Any]:
    session = _get_session(session_id)
    if session.running:
        raise HTTPException(status_code=409, detail="Session is already running")
    if session.engine.step >= session.engine.cfg.num_steps:
        return {"running": False, "completed": True, "reason": "session training complete"}

    session.running = True
    session.run_task = asyncio.create_task(_run_steps(session_id, req.steps, req.delay_ms))
    await session.broadcast({"type": "run_status", "running": True})
    return {"running": True, "steps": req.steps, "delay_ms": req.delay_ms}


@app.post("/api/session/{session_id}/pause")
async def pause_session(session_id: str) -> dict[str, Any]:
    session = _get_session(session_id)
    await session.stop_run()
    await session.broadcast({"type": "run_status", "running": False})
    return {"running": False}


@app.post("/api/session/{session_id}/reset")
async def reset_session(session_id: str) -> dict[str, Any]:
    session = _get_session(session_id)
    await session.stop_run()
    session.engine = MicroGPT(replace(ModelConfig(), **session.engine.cfg.__dict__))
    payload = {"session_id": session_id, "running": False, "metadata": session.engine.metadata()}
    await session.broadcast({"type": "session", **payload})
    return payload


@app.post("/api/session/{session_id}/sample")
async def sample_session(session_id: str, req: SampleRequest) -> dict[str, Any]:
    session = _get_session(session_id)
    async with session.lock:
        samples = session.engine.sample(req.num_samples, req.temperature)
    payload = {
        "type": "samples",
        "samples": samples,
        "temperature": req.temperature,
        "num_samples": req.num_samples,
    }
    await session.broadcast(payload)
    return payload


@app.websocket("/ws/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str) -> None:
    session = _get_session(session_id)
    await websocket.accept()
    session.clients.add(websocket)
    await websocket.send_json({"type": "session", "session_id": session_id, "metadata": session.engine.metadata()})

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        session.clients.discard(websocket)


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))
