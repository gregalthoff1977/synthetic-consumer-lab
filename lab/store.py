"""Minimal durable SQLite study storage; one independent DB per deployment."""
import json
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4


def connect(path=None):
    db = Path(path or os.environ.get("LAB_DB_PATH", "data/studies.sqlite3"))
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS studies (id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS responses (study_id TEXT NOT NULL, concept_id TEXT NOT NULL, respondent_id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(study_id,concept_id,respondent_id))")
    conn.commit()
    return conn


def save_study(conn, study):
    if not study.get("id"):
        study["id"] = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("INSERT INTO studies VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,payload=excluded.payload", (study["id"], study["name"], study.get("created_at", now), json.dumps(study)))
    conn.commit()
    return study["id"]


def list_studies(conn):
    return [dict(r) for r in conn.execute("SELECT id,name,created_at FROM studies ORDER BY created_at DESC")]


def load_study(conn, study_id):
    row = conn.execute("SELECT payload FROM studies WHERE id=?", (study_id,)).fetchone()
    return json.loads(row[0]) if row else None


def save_response(conn, study_id, record):
    conn.execute("INSERT INTO responses VALUES(?,?,?,?) ON CONFLICT(study_id,concept_id,respondent_id) DO UPDATE SET payload=excluded.payload", (study_id, record["concept_id"], record["respondent_id"], json.dumps(record)))
    conn.commit()


def load_responses(conn, study_id):
    return [json.loads(r[0]) for r in conn.execute("SELECT payload FROM responses WHERE study_id=? ORDER BY concept_id,respondent_id", (study_id,))]


def clear_responses(conn, study_id):
    conn.execute("DELETE FROM responses WHERE study_id=?", (study_id,))
    conn.commit()
