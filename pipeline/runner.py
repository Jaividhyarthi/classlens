"""Orchestrates the analysis pipeline for one uploaded lecture.

Phase 0: probes the uploaded file for real duration/stream info and stops
there -- asr_diarize / video_analysis / aggregate / relevance / report are
built in the following steps and wired in here once they exist, instead of
faking their output.
"""
import json
import subprocess
from pathlib import Path

from web import store


def probe_media(path: Path) -> dict:
    """Real ffprobe call -- duration and stream presence, nothing invented."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    has_video = any(s.get("codec_type") == "video" for s in streams)
    duration_sec = float(info.get("format", {}).get("duration", 0.0))
    return {"duration_sec": duration_sec, "has_audio": has_audio, "has_video": has_video}


def run_pipeline(session_id: str) -> None:
    session = store.get_session(session_id)
    if session is None:
        return

    store.update_session(session_id, status="processing", stage="probing_media")
    try:
        media_path = Path(session["upload_path"])
        media_info = probe_media(media_path)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the UI, not swallowed
        store.update_session(session_id, status="error", error=str(exc), stage="probing_media")
        return

    store.update_session(
        session_id,
        status="pending_pipeline",
        stage="probed",
        media_info=media_info,
        error=None,
    )
