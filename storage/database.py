"""
storage/database.py – Phase 3 SQLite persistence layer.

Three tables:
  jobs              – one row per job search session
  research          – key/value research fragments attached to a job
  candidate_analysis – resume parse + job-match data per job
  final_responses   – final LLM analysis JSON per job

Connection handling
  - Every public function accepts an optional `db_path` argument so tests can
    use temporary databases without touching real files.
  - All queries use parameterised (?) placeholders – no string interpolation.
  - Connections are opened/closed inside each function (no global state).

Environment variable
  JOB_AGENT_DB_PATH – path to the SQLite file.  Defaults to data/agent.db
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "agent.db")


def _get_db_path(db_path: Optional[str] = None) -> str:
    """Return the resolved database path."""
    if db_path:
        return db_path
    env_path = os.getenv("JOB_AGENT_DB_PATH")
    if env_path:
        return env_path
    return os.path.abspath(DEFAULT_DB_PATH)


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open a SQLite connection with foreign-key enforcement enabled."""
    path = _get_db_path(db_path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

_CREATE_JOBS = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    title           TEXT,
    company         TEXT,
    job_description TEXT NOT NULL,
    resume_text     TEXT,
    status          TEXT NOT NULL DEFAULT 'new',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
"""

_CREATE_RESEARCH = """
CREATE TABLE IF NOT EXISTS research (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    source      TEXT,
    created_at  TEXT NOT NULL,
    UNIQUE(job_id, key)
);
"""

_CREATE_CANDIDATE_ANALYSIS = """
CREATE TABLE IF NOT EXISTS candidate_analysis (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    resume_data     TEXT,
    job_match       TEXT,
    tool_results    TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
"""

_CREATE_FINAL_RESPONSES = """
CREATE TABLE IF NOT EXISTS final_responses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    response_json   TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
"""


def init_db(db_path: Optional[str] = None) -> None:
    """Create all tables if they do not already exist."""
    with _connect(db_path) as conn:
        conn.executescript(
            _CREATE_JOBS
            + _CREATE_RESEARCH
            + _CREATE_CANDIDATE_ANALYSIS
            + _CREATE_FINAL_RESPONSES
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Jobs table helpers
# ---------------------------------------------------------------------------

def create_job(
    job_id: str,
    job_description: str,
    title: str = "",
    company: str = "",
    resume_text: str = "",
    db_path: Optional[str] = None,
) -> str:
    """
    Insert a new job record and return its job_id.
    Idempotent: if the job_id already exists the existing record is returned unchanged.
    """
    now = _utcnow()
    init_db(db_path)
    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT job_id FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if existing:
            return job_id
        conn.execute(
            """
            INSERT INTO jobs (job_id, title, company, job_description, resume_text,
                              status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'new', ?, ?)
            """,
            (job_id, title, company, job_description, resume_text, now, now),
        )
        conn.commit()
    return job_id


def get_job(job_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return the job row as a dict, or None if not found."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    return dict(row) if row else None


def list_jobs(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return all job records ordered by creation time (newest first)."""
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def update_job_status(
    job_id: str,
    status: str,
    db_path: Optional[str] = None,
) -> None:
    """
    Update the status field of an existing job.
    Valid statuses: 'new', 'running', 'interrupted', 'complete', 'error'
    """
    now = _utcnow()
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
            (status, now, job_id),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Research table helpers
# ---------------------------------------------------------------------------

def save_research(
    job_id: str,
    key: str,
    value: Any,
    source: str = "",
    db_path: Optional[str] = None,
) -> None:
    """
    Persist a research fragment.  Uses INSERT OR REPLACE so re-saving the same
    key updates the existing row rather than inserting a duplicate.
    """
    now = _utcnow()
    init_db(db_path)
    value_str = json.dumps(value) if not isinstance(value, str) else value
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO research (job_id, key, value, source, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(job_id, key) DO UPDATE SET value = excluded.value,
                                                    source = excluded.source
            """,
            (job_id, key, value_str, source, now),
        )
        conn.commit()


def load_research(
    job_id: str, db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Return all research fragments for a job as {key: value}.
    Attempts JSON deserialisation of each value; falls back to raw string.
    """
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT key, value FROM research WHERE job_id = ?", (job_id,)
        ).fetchall()
    result: Dict[str, Any] = {}
    for row in rows:
        try:
            result[row["key"]] = json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            result[row["key"]] = row["value"]
    return result


# ---------------------------------------------------------------------------
# Candidate analysis table helpers
# ---------------------------------------------------------------------------

def save_candidate_analysis(
    job_id: str,
    resume_data: Optional[Dict[str, Any]] = None,
    job_match: Optional[Dict[str, Any]] = None,
    tool_results: Optional[List[Any]] = None,
    db_path: Optional[str] = None,
) -> None:
    """
    Upsert candidate analysis for a job.  Only one row per job_id is kept;
    subsequent calls overwrite the previous record.
    """
    now = _utcnow()
    init_db(db_path)
    resume_json = json.dumps(resume_data or {})
    match_json = json.dumps(job_match or {})
    tools_json = json.dumps(tool_results or [])
    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM candidate_analysis WHERE job_id = ?", (job_id,)
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE candidate_analysis
                SET resume_data = ?, job_match = ?, tool_results = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (resume_json, match_json, tools_json, now, job_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO candidate_analysis
                    (job_id, resume_data, job_match, tool_results, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, resume_json, match_json, tools_json, now, now),
            )
        conn.commit()


def load_candidate_analysis(
    job_id: str, db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Return the latest candidate analysis for a job, or None.
    Returns a dict with keys: resume_data, job_match, tool_results.
    """
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM candidate_analysis WHERE job_id = ?", (job_id,)
        ).fetchone()
    if row is None:
        return None
    return {
        "resume_data": _safe_json(row["resume_data"]),
        "job_match": _safe_json(row["job_match"]),
        "tool_results": _safe_json(row["tool_results"]),
    }


# ---------------------------------------------------------------------------
# Final responses table helpers
# ---------------------------------------------------------------------------

def save_final_response(
    job_id: str,
    response: Dict[str, Any],
    db_path: Optional[str] = None,
) -> None:
    """Persist the final LLM-generated analysis for a job."""
    now = _utcnow()
    init_db(db_path)
    response_json = json.dumps(response)
    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM final_responses WHERE job_id = ?", (job_id,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE final_responses SET response_json = ?, updated_at = ? WHERE job_id = ?",
                (response_json, now, job_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO final_responses (job_id, response_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (job_id, response_json, now, now),
            )
        conn.commit()


def load_final_response(
    job_id: str, db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Return the saved final response for a job, or None."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT response_json FROM final_responses WHERE job_id = ?", (job_id,)
        ).fetchone()
    if row is None:
        return None
    return _safe_json(row["response_json"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(value: Optional[str]) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
