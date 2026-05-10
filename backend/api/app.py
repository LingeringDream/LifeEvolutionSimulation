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


class CustomPlanet(BaseModel):
    name: str = "Custom"
    gravity: float = 9.8
    surface_temp: float = 15.0
    albedo: float = 0.3
    axial_tilt: float = 23.5
    orbital_distance: float = 1.0
    atmospheric_pressure: float = 1.0
    co2_ratio: float = 0.0004
    ch4_ratio: float = 0.0
    magnetic_field: float = 1.0
    season_period: int = 365

class StartRequest(BaseModel):
    planet: str = "titan"
    custom_planet: CustomPlanet | None = None
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

@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import Response
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y="80" font-size="80">🧬</text></svg>'
    return Response(content=svg, media_type="image/svg+xml")


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
    custom_cfg = req.custom_planet.model_dump() if req.custom_planet else None
    sim.start(
        planet=req.planet,
        producers=req.producers,
        consumers=req.consumers,
        grid_size=req.grid_size,
        ai_provider=ai_provider,
        ai_interval=req.ai_interval,
        custom_planet=custom_cfg,
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


@app.get("/api/config")
async def get_config():
    """Return AI config from .env (key masked)."""
    import os
    def mask(key):
        if not key: return ""
        return key[:6] + "..." + key[-4:] if len(key) > 12 else "***"

    return {
        "provider": os.environ.get("AI_PROVIDER", ""),
        "openai_model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        "openai_key_preview": mask(os.environ.get("OPENAI_API_KEY", "")),
        "claude_model": os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
        "claude_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "claude_key_preview": mask(os.environ.get("ANTHROPIC_API_KEY", "")),
        "custom_base_url": os.environ.get("CUSTOM_API_BASE_URL", ""),
        "custom_model": os.environ.get("CUSTOM_MODEL", ""),
        "custom_key_set": bool(os.environ.get("CUSTOM_API_KEY")),
        "custom_key_preview": mask(os.environ.get("CUSTOM_API_KEY", "")),
        "ai_interval": int(os.environ.get("AI_INTERVAL", "60")),
    }


# ── Data export endpoints ──────────────────────────────────────

@app.get("/api/runs")
async def api_list_runs():
    return sim.list_runs()


@app.get("/api/runs/{run_id}")
async def api_get_run(run_id: int):
    run = sim.get_run(run_id)
    if not run:
        return {"error": "Run not found"}
    return run


@app.get("/api/runs/{run_id}/snapshots")
async def api_get_snapshots(run_id: int):
    return sim.get_snapshots(run_id)


@app.get("/api/runs/{run_id}/species")
async def api_get_species_history(run_id: int, species_id: str | None = None):
    return sim.get_species_history(run_id, species_id)


@app.get("/api/runs/{run_id}/events")
async def api_get_events(run_id: int):
    return sim.get_events(run_id)


@app.post("/api/runs/{run_id}/export/csv")
async def api_export_csv(run_id: int):
    import tempfile
    out_dir = tempfile.mkdtemp(prefix="evo_export_")
    files = sim.export_csv(run_id, out_dir)
    return {"status": "ok", "files": files, "dir": out_dir}


@app.post("/api/runs/{run_id}/export/json")
async def api_export_json(run_id: int):
    import tempfile
    out_path = tempfile.mktemp(prefix=f"run_{run_id}_", suffix=".json")
    path = sim.export_json(run_id, out_path)
    return {"status": "ok", "file": path}


# ── Save / Load endpoints ──────────────────────────────────────

class SaveRequest(BaseModel):
    name: str = ""

class LoadRequest(BaseModel):
    save_id: str
    ai: str | None = None
    ai_model: str | None = None
    ai_key: str | None = None
    ai_base_url: str | None = None

class AutoSaveRequest(BaseModel):
    enabled: bool = True
    interval: int = 500

@app.get("/api/saves")
async def api_list_saves():
    return sim.list_saves()

@app.post("/api/saves")
async def api_save(req: SaveRequest):
    save_id = sim.save(name=req.name)
    if save_id:
        return {"status": "ok", "save_id": save_id}
    return {"status": "error", "message": "No running simulation"}

@app.post("/api/saves/load")
async def api_load_save(req: LoadRequest):
    ai_provider = None
    if req.ai:
        ai_provider = create_ai_provider(provider=req.ai, api_key=req.ai_key, model=req.ai_model, base_url=req.ai_base_url)
    ok = sim.load(req.save_id, ai_provider=ai_provider)
    if ok:
        return {"status": "ok"}
    return {"status": "error", "message": "Failed to load save"}

@app.delete("/api/saves/{save_id}")
async def api_delete_save(save_id: str):
    ok = sim.delete_save(save_id)
    return {"status": "ok" if ok else "error"}

@app.post("/api/simulation/auto-save")
async def api_set_auto_save(req: AutoSaveRequest):
    sim.set_auto_save(req.enabled, req.interval)
    return {"enabled": req.enabled, "interval": req.interval}


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
