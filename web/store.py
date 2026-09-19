"""Local JSON-file storage. Phase 1 has no database: sessions and reports
are one JSON file each under data/sessions and data/reports."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
REPORTS_DIR = DATA_DIR / "reports"
UPLOADS_DIR = DATA_DIR / "uploads"

for d in (SESSIONS_DIR, REPORTS_DIR, UPLOADS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- sessions (upload + pipeline status) ---

def create_session(meta: dict) -> dict:
    session_id = meta["session_id"]
    record = {
        "session_id": session_id,
        "status": "queued",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "error": None,
        **meta,
    }
    _write_json(SESSIONS_DIR / f"{session_id}.json", record)
    return record


def get_session(session_id: str) -> Optional[dict]:
    return _read_json(SESSIONS_DIR / f"{session_id}.json")


def update_session(session_id: str, **fields: Any) -> Optional[dict]:
    record = get_session(session_id)
    if record is None:
        return None
    record.update(fields)
    record["updated_at"] = now_iso()
    _write_json(SESSIONS_DIR / f"{session_id}.json", record)
    return record


def list_sessions() -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(SESSIONS_DIR.glob("*.json"))
    ]


# --- reports (pipeline output, the report.html data shape) ---

def save_report(session_id: str, report: dict) -> None:
    _write_json(REPORTS_DIR / f"{session_id}.json", report)


def get_report(session_id: str) -> Optional[dict]:
    return _read_json(REPORTS_DIR / f"{session_id}.json")


def list_reports() -> list[dict]:
    reports = []
    for p in sorted(REPORTS_DIR.glob("*.json")):
        try:
            reports.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return reports


# --- static reference config (faculty roster, syllabus) ---

def load_faculty() -> list[dict]:
    data = _read_json(DATA_DIR / "faculty.json") or {"faculty": []}
    return data["faculty"]


def get_faculty(faculty_id: str) -> Optional[dict]:
    for f in load_faculty():
        if f["id"] == faculty_id:
            return f
    return None


def load_syllabus() -> dict:
    return _read_json(DATA_DIR / "syllabus.json") or {"subjects": {}}
