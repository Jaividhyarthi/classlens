"""Orchestrates the full analysis pipeline for one uploaded lecture:
probe -> ASR+diarization -> video pose analysis -> aggregate -> relevance
classification -> report assembly -> store. Runs as a FastAPI background
task so POST /api/analyze returns immediately.
"""
import json
import subprocess
from pathlib import Path

from pipeline import aggregate, asr_diarize, relevance, report, video_analysis
from pipeline.sarvam_client import SarvamConfigError
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


def _lookup_unit(subject: str, unit_id: str) -> dict:
    syllabus = store.load_syllabus()
    subject_cfg = syllabus.get("subjects", {}).get(subject, {})
    for unit in subject_cfg.get("units", []):
        if unit.get("id") == unit_id:
            return unit
    units = subject_cfg.get("units", [])
    return units[0] if units else {}


def run_pipeline(session_id: str) -> None:
    session = store.get_session(session_id)
    if session is None:
        return
    media_path = Path(session["upload_path"])

    store.update_session(session_id, status="processing", stage="probing_media")
    try:
        media_info = probe_media(media_path)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the UI, not swallowed
        store.update_session(session_id, status="error", error=str(exc), stage="probing_media")
        return
    store.update_session(session_id, stage="probed", media_info=media_info, error=None)

    if not media_info["has_audio"]:
        store.update_session(session_id, status="error", stage="probed",
                              error="Uploaded file has no audio track -- cannot transcribe.")
        return

    try:
        store.update_session(session_id, status="processing", stage="transcribing_diarizing")
        transcript = asr_diarize.transcribe_and_diarize(media_path)
    except SarvamConfigError as exc:
        store.update_session(session_id, status="error", stage="transcribing_diarizing", error=str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        store.update_session(session_id, status="error", stage="transcribing_diarizing",
                              error=f"ASR/diarization failed: {exc}")
        return

    video_windows = []
    if media_info["has_video"]:
        try:
            store.update_session(session_id, status="processing", stage="analyzing_video")
            video_windows = video_analysis.analyze_video(media_path)
        except Exception as exc:  # noqa: BLE001
            # Video analysis failing shouldn't block an audio-only report --
            # windows just get zeroed gesture/board-facing values downstream.
            store.update_session(session_id, stage="analyzing_video", error=f"Video analysis failed (continuing audio-only): {exc}")
            video_windows = []

    try:
        store.update_session(session_id, status="processing", stage="aggregating")
        timeline = aggregate.build_timeline(transcript, video_windows, duration_sec=media_info["duration_sec"])

        store.update_session(session_id, stage="classifying_relevance")
        unit = _lookup_unit(session["subject"], session.get("unit", ""))
        syllabus_topics = unit.get("topics", [])
        relevance_results = relevance.classify_segments(transcript.segments, session["subject"], syllabus_topics)

        store.update_session(session_id, stage="building_report")
        session_meta = dict(session)
        session_meta["duration_sec"] = media_info["duration_sec"]
        final_report = report.build_report(
            session_id=session_id,
            session_meta=session_meta,
            transcript=transcript,
            timeline=timeline,
            relevance_results=relevance_results,
            unit_config=unit,
        )
    except Exception as exc:  # noqa: BLE001
        store.update_session(session_id, status="error", stage="building_report", error=f"Report assembly failed: {exc}")
        return

    store.save_report(session_id, final_report)
    store.update_session(session_id, status="ready", stage="done", error=None)
