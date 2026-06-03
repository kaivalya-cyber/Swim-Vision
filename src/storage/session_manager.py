# This file manages persistent session storage for SwimVision analysis results.
"""Session storage manager — SQLite-based persistent tracking for SwimVision."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.reference.optimal_ranges import SwimmerProfile


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "swimvision.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS swimmers (
    id TEXT PRIMARY KEY,
    name TEXT,
    height_cm REAL,
    experience TEXT DEFAULT 'intermediate',
    gender TEXT,
    age INTEGER,
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    swimmer_id TEXT,
    clip_id TEXT NOT NULL,
    analysis_mode TEXT NOT NULL DEFAULT 'dive',
    original_filename TEXT,
    crop_x INTEGER,
    crop_y INTEGER,
    crop_w INTEGER,
    crop_h INTEGER,
    overall_severity TEXT,
    reaction_time_ms REAL,
    num_cycles INTEGER DEFAULT 0,
    stroke_type TEXT,
    status TEXT DEFAULT 'completed',
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (swimmer_id) REFERENCES swimmers(id)
);

CREATE TABLE IF NOT EXISTS session_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    phase TEXT,
    flag TEXT,
    category TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS session_risk (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    risk_level TEXT,
    risk_score REAL,
    num_factors INTEGER,
    flagged_factors_json TEXT,
    recommendations_json TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS session_symmetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    pair_name TEXT NOT NULL,
    left_value REAL,
    right_value REAL,
    symmetry_index_pct REAL,
    classification TEXT,
    phase TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_swimmer ON sessions(swimmer_id);
CREATE INDEX IF NOT EXISTS idx_sessions_mode ON sessions(analysis_mode);
CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_metrics_session ON session_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_risk_session ON session_risk(session_id);
CREATE INDEX IF NOT EXISTS idx_symmetry_session ON session_symmetry(session_id);
"""


def _get_db_path() -> Path:
    """Get the SQLite database path, respecting environment overrides."""
    import os
    env_path = os.environ.get("SWIMVISION_DB_PATH")
    return Path(env_path) if env_path else DB_PATH


def init_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Initialize the SQLite database and return a connection."""
    path = db_path or _get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    except Exception as exc:
        LOGGER.error("Failed to initialize database schema: %s", exc)
    return conn


def upsert_swimmer(
    conn: sqlite3.Connection,
    swimmer_id: str,
    name: str = "",
    profile: Optional[SwimmerProfile] = None,
    notes: str = "",
) -> bool:
    """Insert or update a swimmer record."""
    try:
        now = datetime.now().isoformat(timespec="seconds")
        conn.execute(
            """INSERT INTO swimmers (id, name, height_cm, experience, gender, age, notes, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
               name=excluded.name, height_cm=excluded.height_cm,
               experience=excluded.experience, gender=excluded.gender,
               age=excluded.age, notes=excluded.notes, updated_at=excluded.updated_at""",
            (
                swimmer_id,
                name,
                profile.height_cm if profile else None,
                profile.experience if profile else "intermediate",
                profile.gender if profile else None,
                profile.age if profile else None,
                notes,
                now,
            ),
        )
        conn.commit()
        return True
    except Exception as exc:
        LOGGER.error("Failed to upsert swimmer %s: %s", swimmer_id, exc)
        return False


def record_session(
    conn: sqlite3.Connection,
    session_id: str,
    clip_id: str,
    swimmer_id: str = "",
    analysis_mode: str = "dive",
    original_filename: str = "",
    crop: Optional[List[int]] = None,
    overall_severity: Optional[str] = None,
    reaction_time_ms: Optional[float] = None,
    num_cycles: int = 0,
    stroke_type: Optional[str] = None,
    status: str = "completed",
) -> bool:
    """Record a completed analysis session."""
    try:
        existing = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if existing:
            conn.execute(
                """UPDATE sessions SET
                   swimmer_id=?, clip_id=?, analysis_mode=?, original_filename=?,
                   crop_x=COALESCE(?, crop_x), crop_y=COALESCE(?, crop_y),
                   crop_w=COALESCE(?, crop_w), crop_h=COALESCE(?, crop_h),
                   overall_severity=COALESCE(?, overall_severity),
                   reaction_time_ms=COALESCE(?, reaction_time_ms),
                   num_cycles=?, stroke_type=COALESCE(?, stroke_type),
                   status=?
                   WHERE id=?""",
                (
                    swimmer_id or "",
                    clip_id,
                    analysis_mode,
                    original_filename,
                    crop[0] if crop else None, crop[1] if crop else None,
                    crop[2] if crop else None, crop[3] if crop else None,
                    overall_severity,
                    reaction_time_ms,
                    num_cycles,
                    stroke_type,
                    status,
                    session_id,
                ),
            )
        else:
            conn.execute(
                """INSERT INTO sessions
                   (id, swimmer_id, clip_id, analysis_mode, original_filename,
                    crop_x, crop_y, crop_w, crop_h, overall_severity,
                    reaction_time_ms, num_cycles, stroke_type, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    swimmer_id or "",
                    clip_id,
                    analysis_mode,
                    original_filename,
                    crop[0] if crop else None,
                    crop[1] if crop else None,
                    crop[2] if crop else None,
                    crop[3] if crop else None,
                    overall_severity,
                    reaction_time_ms,
                    num_cycles,
                    stroke_type,
                    status,
                ),
            )
        conn.commit()
        return True
    except Exception as exc:
        LOGGER.error("Failed to record session %s: %s", session_id, exc)
        return False


def record_metrics(
    conn: sqlite3.Connection,
    session_id: str,
    metrics: Dict[str, Any],
) -> int:
    """Record session metric values."""
    count = 0
    try:
        for phase_name in ("block_phase", "flight_phase", "entry_phase", "stroke_cycle"):
            phase_metrics = metrics.get(phase_name, [])
            if isinstance(phase_metrics, list):
                for row in phase_metrics:
                    if isinstance(row, dict):
                        conn.execute(
                            """INSERT INTO session_metrics (session_id, metric_name, metric_value, phase, flag)
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                session_id,
                                str(row.get("metric", "")),
                                float(row.get("measured", 0)) if row.get("measured") is not None else None,
                                phase_name,
                                str(row.get("flag", "")),
                            ),
                        )
                        count += 1
        conn.commit()
    except Exception as exc:
        LOGGER.error("Failed to record metrics for %s: %s", session_id, exc)
    return count


def record_risk(
    conn: sqlite3.Connection,
    session_id: str,
    risk_data: Dict[str, Any],
) -> bool:
    """Record injury risk assessment results."""
    try:
        existing = conn.execute("SELECT 1 FROM session_risk WHERE session_id = ?", (session_id,)).fetchone()
        if existing:
            conn.execute(
                """UPDATE session_risk SET
                   risk_level=?, risk_score=?, num_factors=?,
                   flagged_factors_json=?, recommendations_json=?
                   WHERE session_id=?""",
                (
                    risk_data.get("overall_risk_level"),
                    risk_data.get("total_risk_score"),
                    risk_data.get("num_flagged_factors"),
                    json.dumps(risk_data.get("flagged_factors", [])),
                    json.dumps(risk_data.get("preventive_recommendations", [])),
                    session_id,
                ),
            )
        else:
            conn.execute(
                """INSERT INTO session_risk
                   (session_id, risk_level, risk_score, num_factors, flagged_factors_json, recommendations_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    risk_data.get("overall_risk_level"),
                    risk_data.get("total_risk_score"),
                    risk_data.get("num_flagged_factors"),
                    json.dumps(risk_data.get("flagged_factors", [])),
                    json.dumps(risk_data.get("preventive_recommendations", [])),
                ),
            )
        conn.commit()
        return True
    except Exception as exc:
        LOGGER.error("Failed to record risk for %s: %s", session_id, exc)
        return False


def record_symmetry(
    conn: sqlite3.Connection,
    session_id: str,
    symmetry_data: Dict[str, Any],
) -> int:
    """Record symmetry analysis results."""
    count = 0
    try:
        phases = symmetry_data.get("phases", {})
        for phase_name, pairs in phases.items():
            for pair_name, data in pairs.items():
                conn.execute(
                    """INSERT INTO session_symmetry
                       (session_id, pair_name, left_value, right_value, symmetry_index_pct, classification, phase)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        pair_name,
                        data.get("left_mean") or data.get("left_value"),
                        data.get("right_mean") or data.get("right_value"),
                        data.get("symmetry_index_pct"),
                        data.get("classification"),
                        phase_name,
                    ),
                )
                count += 1
        conn.commit()
    except Exception as exc:
        LOGGER.error("Failed to record symmetry for %s: %s", session_id, exc)
    return count


def get_swimmer_history(
    conn: sqlite3.Connection,
    swimmer_id: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Get session history for a swimmer."""
    try:
        cursor = conn.execute(
            """SELECT id, clip_id, analysis_mode, overall_severity, reaction_time_ms,
                      num_cycles, stroke_type, status, created_at
               FROM sessions
               WHERE swimmer_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (swimmer_id, limit),
        )
        return [dict(zip([d[0] for d in cursor.description], row)) for row in cursor.fetchall()]
    except Exception as exc:
        LOGGER.error("Failed to get history for %s: %s", swimmer_id, exc)
        return []


def get_all_swimmers(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Get all registered swimmers."""
    try:
        cursor = conn.execute("SELECT id, name, height_cm, experience, gender, age FROM swimmers ORDER BY name")
        return [dict(zip([d[0] for d in cursor.description], row)) for row in cursor.fetchall()]
    except Exception as exc:
        LOGGER.error("Failed to get swimmers: %s", exc)
        return []
