"""FastAPI application — REST + WebSocket for the evolution simulator."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from api.sim_manager import SimManager
from ai.factory import create_ai_provider

app = FastAPI(title="Alien Evolution Simulator")
sim = SimManager()

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


class StartRequest(BaseModel):
    planet: str = "titan"
    producers: int = 2
    consumers: int = 1
    grid_size: int = 50
    ai: str | None = None
    ai_model: str | None = None
    ai_key: str | None = None
    ai_base_url: str | None = None
    ai_interval: int = 60


# ── REST endpoints ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = FRONTEND_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/simulation/start")
async def start_sim(req: StartRequest):
    ai_provider = None
    if req.ai:
        ai_provider = create_ai_provider(
            provider=req.ai,
            api_key=req.ai_key,
            model=req.ai_model,
            base_url=req.ai_base_url,
        )
    sim.start(
        planet=req.planet,
        producers=req.producers,
        consumers=req.consumers,
        grid_size=req.grid_size,
        ai_provider=ai_provider,
        ai_interval=req.ai_interval,
    )
    return {"status": "started", "planet": req.planet}


@app.post("/api/simulation/pause")
async def pause_sim():
    sim.pause()
    return {"status": "paused"}


@app.post("/api/simulation/resume")
async def resume_sim():
    sim.resume()
    return {"status": "resumed"}


@app.post("/api/simulation/stop")
async def stop_sim():
    sim.stop()
    return {"status": "stopped"}


@app.post("/api/simulation/speed")
async def set_speed(tps: int = 10):
    sim.set_speed(tps)
    return {"speed": tps}


@app.get("/api/simulation/state")
async def get_state():
    return sim.get_state()


@app.get("/api/planets")
async def list_planets():
    from simulation.templates import list_planets as _list
    return _list()


# ── WebSocket (concurrent send/receive) ────────────────────────

@app.websocket("/ws/simulation")
async def ws_simulation(ws: WebSocket):
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    sim.register_ws(queue)
    sim.set_event_loop(asyncio.get_event_loop())

    # Send initial state + grid
    state = sim.get_state()
    if state:
        await ws.send_json(state)

    async def sender():
        """Forward simulation broadcasts to the client."""
        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=2.0)
                await ws.send_text(payload)
            except asyncio.TimeoutError:
                try:
                    await ws.send_json({"type": "ping"})
                except Exception:
                    break
            except Exception:
                break

    async def receiver():
        """Handle client control messages."""
        while True:
            try:
                msg = await ws.receive_text()
                data = json.loads(msg)
                action = data.get("action")
                if action == "pause":
                    sim.pause()
                elif action == "resume":
                    sim.resume()
                elif action == "speed":
                    sim.set_speed(data.get("tps", 10))
                elif action == "step":
                    # Single step while paused
                    if sim.engine and not sim._running:
                        sim.engine.step()
                        state = sim._build_broadcast()
                        state["running"] = False
                        await ws.send_json(state)
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                pass
            except Exception:
                break

    try:
        # Run sender and receiver concurrently
        done, pending = await asyncio.wait(
            [asyncio.create_task(sender()), asyncio.create_task(receiver())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
    except Exception:
        pass
    finally:
        sim.unregister_ws(queue)
