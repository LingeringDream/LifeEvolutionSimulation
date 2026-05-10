"""SQLite database for persisting simulation data."""
from __future__ import annotations
import sqlite3
import json
import time
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "sim_data.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL DEFAULT '',
            planet      TEXT NOT NULL,
            grid_size   INTEGER NOT NULL,
            producers   INTEGER NOT NULL,
            consumers   INTEGER NOT NULL,
            ai_provider TEXT,
            ai_model    TEXT,
            ai_interval INTEGER,
            config_json TEXT,
            created_at  REAL NOT NULL,
            finished_at REAL,
            total_ticks INTEGER DEFAULT 0,
            status      TEXT DEFAULT 'running'
        );

        CREATE TABLE IF NOT EXISTS snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      INTEGER NOT NULL REFERENCES runs(id),
            tick        INTEGER NOT NULL,
            timestamp   REAL NOT NULL,
            env_json    TEXT NOT NULL,
            species_json TEXT NOT NULL,
            events_json TEXT
        );

        CREATE TABLE IF NOT EXISTS species_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      INTEGER NOT NULL REFERENCES runs(id),
            tick        INTEGER NOT NULL,
            species_id  TEXT NOT NULL,
            species_name TEXT NOT NULL,
            biomass     REAL NOT NULL,
            metabolic_type TEXT,
            genes_json  TEXT
        );

        CREATE TABLE IF NOT EXISTS event_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      INTEGER NOT NULL REFERENCES runs(id),
            tick        INTEGER NOT NULL,
            event_type  TEXT NOT NULL,
            description TEXT,
            details_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_snap_run ON snapshots(run_id, tick);
        CREATE INDEX IF NOT EXISTS idx_sp_run ON species_log(run_id, tick);
        CREATE INDEX IF NOT EXISTS idx_evt_run ON event_log(run_id, tick);
    """)
    conn.close()


# ── Run management ──────────────────────────────────────────────

def create_run(planet: str, grid_size: int, producers: int, consumers: int,
               ai_provider: str | None = None, ai_model: str | None = None,
               ai_interval: int | None = None, config_json: str = "{}") -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO runs (planet, grid_size, producers, consumers, ai_provider, ai_model, ai_interval, config_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (planet, grid_size, producers, consumers, ai_provider, ai_model, ai_interval, config_json, time.time()),
    )
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def finish_run(run_id: int, total_ticks: int):
    conn = get_conn()
    conn.execute(
        "UPDATE runs SET finished_at=?, total_ticks=?, status='completed' WHERE id=?",
        (time.time(), total_ticks, run_id),
    )
    conn.commit()
    conn.close()


def list_runs() -> list[dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_run(run_id: int) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Snapshot recording ──────────────────────────────────────────

def save_snapshot(run_id: int, tick: int, env: dict, species: list[dict], events: list[dict]):
    conn = get_conn()
    conn.execute(
        "INSERT INTO snapshots (run_id, tick, timestamp, env_json, species_json, events_json) VALUES (?,?,?,?,?,?)",
        (run_id, tick, time.time(), json.dumps(env), json.dumps(species), json.dumps(events)),
    )
    # Also log individual species
    for sp in species:
        conn.execute(
            "INSERT INTO species_log (run_id, tick, species_id, species_name, biomass, metabolic_type, genes_json) VALUES (?,?,?,?,?,?,?)",
            (run_id, tick, sp.get("id", ""), sp.get("name", ""), sp.get("biomass", 0),
             sp.get("metabolic_type", ""), json.dumps(sp.get("genes", {}))),
        )
    conn.commit()
    conn.close()


def save_events(run_id: int, events: list[dict]):
    if not events:
        return
    conn = get_conn()
    for e in events:
        conn.execute(
            "INSERT INTO event_log (run_id, tick, event_type, description, details_json) VALUES (?,?,?,?,?)",
            (run_id, e.get("tick", 0), e.get("type", ""), e.get("desc", ""), json.dumps(e)),
        )
    conn.commit()
    conn.close()


# ── Query ───────────────────────────────────────────────────────

def get_snapshots(run_id: int, limit: int = 1000) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM snapshots WHERE run_id=? ORDER BY tick ASC LIMIT ?",
        (run_id, limit),
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["env"] = json.loads(d["env_json"])
        d["species"] = json.loads(d["species_json"])
        d["events"] = json.loads(d["events_json"] or "[]")
        del d["env_json"], d["species_json"], d["events_json"]
        result.append(d)
    return result


def get_species_history(run_id: int, species_id: str | None = None, limit: int = 5000) -> list[dict]:
    conn = get_conn()
    if species_id:
        rows = conn.execute(
            "SELECT * FROM species_log WHERE run_id=? AND species_id=? ORDER BY tick ASC LIMIT ?",
            (run_id, species_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM species_log WHERE run_id=? ORDER BY tick ASC LIMIT ?",
            (run_id, limit),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_events(run_id: int, limit: int = 500) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM event_log WHERE run_id=? ORDER BY tick ASC LIMIT ?",
        (run_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Export ──────────────────────────────────────────────────────

def export_run_csv(run_id: int, output_dir: str) -> dict[str, str]:
    """Export a run's data to CSV files. Returns dict of {name: path}."""
    import csv
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}

    # Species history
    sp_path = out / f"run_{run_id}_species.csv"
    rows = get_species_history(run_id, limit=100000)
    if rows:
        with open(sp_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        files["species"] = str(sp_path)

    # Events
    ev_path = out / f"run_{run_id}_events.csv"
    rows = get_events(run_id, limit=100000)
    if rows:
        with open(ev_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        files["events"] = str(ev_path)

    # Snapshots summary
    sn_path = out / f"run_{run_id}_snapshots.csv"
    snaps = get_snapshots(run_id, limit=100000)
    if snaps:
        flat = []
        for s in snaps:
            row = {"tick": s["tick"], "timestamp": s["timestamp"]}
            row["temp_mean"] = s["env"].get("temperature_mean", 0)
            row["res_mean"] = s["env"].get("resources_mean", 0)
            row["o2"] = s["env"].get("atmosphere", {}).get("O2", 0)
            row["co2"] = s["env"].get("atmosphere", {}).get("CO2", 0)
            row["n_species"] = len(s["species"])
            row["total_biomass"] = sum(sp.get("biomass", 0) for sp in s["species"])
            flat.append(row)
        with open(sn_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=flat[0].keys())
            w.writeheader()
            w.writerows(flat)
        files["snapshots"] = str(sn_path)

    return files


def export_run_json(run_id: int, output_path: str) -> str:
    """Export full run data to a single JSON file."""
    run = get_run(run_id)
    snaps = get_snapshots(run_id, limit=100000)
    events = get_events(run_id, limit=100000)
    data = {"run": run, "snapshots": snaps, "events": events}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(path)
