"""Aggregates real stored reports (data/reports/*.json) into the shapes the
dashboard templates need. Everything here reads only from actual pipeline
output + the faculty/syllabus config -- nothing is fabricated. Faculty or
subjects with zero analyzed sessions get honest zero/empty values.
"""
from collections import defaultdict
from typing import Optional

from . import store


def _avg(values: list[float]) -> float:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 1) if values else 0.0


def reports_for_faculty(faculty_id: str) -> list[dict]:
    return [r for r in store.list_reports() if r.get("faculty_id") == faculty_id and r.get("status") == "ready"]


def latest_processing_session(faculty_id: Optional[str] = None, department: Optional[str] = None) -> Optional[dict]:
    """Most recent session still mid-pipeline, for the 'live' status banners.
    Honest substitute for a live camera feed we don't have: shows real
    pipeline progress instead of a fabricated 'live engagement %'.
    """
    roster_ids = None
    if department:
        roster_ids = {f["id"] for f in store.load_faculty() if f["department"] == department}

    candidates = [
        s for s in store.list_sessions()
        if s.get("status") in ("queued", "processing", "pending_pipeline")
    ]
    if faculty_id:
        candidates = [s for s in candidates if s.get("faculty_id") == faculty_id]
    elif roster_ids is not None:
        candidates = [s for s in candidates if s.get("faculty_id") in roster_ids]

    if not candidates:
        return None
    candidates.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return candidates[0]


def staff_dashboard_context(faculty_id: str) -> dict:
    faculty = store.get_faculty(faculty_id)
    reports = sorted(
        reports_for_faculty(faculty_id),
        key=lambda r: r.get("session_datetime", ""),
        reverse=True,
    )

    talk_ratios = [r["metrics"]["talk_ratio_teacher_pct"] for r in reports if r.get("metrics")]
    syllabus_pcts = [r["syllabus"]["pct_covered"] for r in reports if r.get("syllabus")]
    wait_times = [r["metrics"]["avg_wait_time_sec"] for r in reports if r.get("metrics")]
    quality_scores = [r["quality_index"] for r in reports if r.get("quality_index") is not None]

    subjects: dict[str, dict] = {}
    for f in store.load_faculty():
        pass
    subject_chips = faculty["subjects"] if faculty else []

    return {
        "faculty": faculty,
        "session_count": len(reports),
        "talk_ratio_teacher": _avg(talk_ratios),
        "talk_ratio_student": round(100 - _avg(talk_ratios), 1) if talk_ratios else 0.0,
        "syllabus_coverage": _avg(syllabus_pcts),
        "avg_wait_time": _avg(wait_times),
        "quality_index": _avg(quality_scores),
        "subject_chips": subject_chips,
        "recent_reports": reports[:6],
        "processing_session": latest_processing_session(faculty_id=faculty_id),
    }


def hod_dashboard_context(department: str = "Mechanical Engineering") -> dict:
    roster = [f for f in store.load_faculty() if f["department"] == department]
    all_reports = [r for r in store.list_reports() if r.get("status") == "ready"]
    dept_reports = [r for r in all_reports if any(f["id"] == r.get("faculty_id") for f in roster)]

    talk_ratios = [r["metrics"]["talk_ratio_teacher_pct"] for r in dept_reports if r.get("metrics")]

    flagged = []
    for r in dept_reports:
        rel = r.get("topic_relevance") or {}
        if rel.get("off_topic_count", 0) > 0 or rel.get("low_confidence_count", 0) > 0:
            flagged.append(r)

    leaderboard = []
    for f in roster:
        f_reports = [r for r in dept_reports if r.get("faculty_id") == f["id"]]
        if not f_reports:
            continue
        latest = sorted(f_reports, key=lambda r: r.get("session_datetime", ""), reverse=True)[0]
        leaderboard.append({
            "faculty": f,
            "session_count": len(f_reports),
            "latest_session_id": latest["session_id"],
            "quality_index": _avg([r["quality_index"] for r in f_reports if r.get("quality_index") is not None]),
            "punctuality_pct": _avg([r["metrics"].get("punctuality_pct", 100.0) for r in f_reports if r.get("metrics")]),
            "syllabus_coverage": _avg([r["syllabus"]["pct_covered"] for r in f_reports if r.get("syllabus")]),
            "question_count": sum(sum(w.get("question_count", 0) for w in r.get("windows", [])) for r in f_reports),
            "last_note": (f_reports[0].get("coaching") or [{}])[0].get("text", ""),
        })
    leaderboard.sort(key=lambda x: x["quality_index"], reverse=True)

    year_labels = {1: "1st Yr", 2: "2nd Yr", 3: "3rd Yr", 4: "4th Yr"}
    year_bars = []
    for i, year in enumerate([1, 2, 3, 4]):
        year_reports = [r for r in dept_reports if r.get("year") == year and r.get("syllabus")]
        pct = _avg([r["syllabus"]["pct_covered"] for r in year_reports])
        bar_h = round(pct / 100 * 70, 1)
        year_bars.append({
            "year": year,
            "label": year_labels[year],
            "pct": pct,
            "x": 25 + i * 80,
            "y": round(94 - 14 - bar_h, 1),
            "h": bar_h,
            "text_x": 25 + i * 80 + 17,
        })

    return {
        "department": department,
        "faculty_count": len(roster),
        "sessions_audited": len(dept_reports),
        "dept_talk_ratio": _avg(talk_ratios),
        "flagged_reports": flagged,
        "leaderboard": leaderboard,
        "processing_session": latest_processing_session(department=department),
        "year_bars": year_bars,
    }


def faculty_directory_context(department: str = "Mechanical Engineering") -> dict:
    roster = [f for f in store.load_faculty() if f["department"] == department]
    entries = []
    for f in roster:
        reports = reports_for_faculty(f["id"])
        talk_ratios = [r["metrics"]["talk_ratio_teacher_pct"] for r in reports if r.get("metrics")]
        syllabus_pcts = [r["syllabus"]["pct_covered"] for r in reports if r.get("syllabus")]
        latest_note = None
        latest_session_id = None
        if reports:
            reports_sorted = sorted(reports, key=lambda r: r.get("session_datetime", ""), reverse=True)
            latest_session_id = reports_sorted[0]["session_id"]
            coaching = reports_sorted[0].get("coaching") or []
            latest_note = coaching[0]["text"] if coaching else None
        entries.append({
            "faculty": f,
            "session_count": len(reports),
            "quality_index": _avg([r["quality_index"] for r in reports if r.get("quality_index") is not None]),
            "talk_ratio": _avg(talk_ratios),
            "syllabus_coverage": _avg(syllabus_pcts),
            "latest_note": latest_note or "No lectures analyzed yet for this faculty member.",
            "latest_session_id": latest_session_id,
        })
    entries.sort(key=lambda e: e["quality_index"], reverse=True)
    return {"department": department, "entries": entries, "count": len(entries)}


def analytics_context(subject: Optional[str] = None) -> dict:
    reports = [r for r in store.list_reports() if r.get("status") == "ready"]
    if subject:
        reports = [r for r in reports if r.get("subject") == subject]

    by_faculty: dict[str, list[dict]] = defaultdict(list)
    for r in reports:
        by_faculty[r["faculty_id"]].append(r)

    rows = []
    for faculty_id, f_reports in by_faculty.items():
        faculty = store.get_faculty(faculty_id)
        if not faculty:
            continue
        talk_ratios = [r["metrics"]["talk_ratio_teacher_pct"] for r in f_reports if r.get("metrics")]
        syllabus_pcts = [r["syllabus"]["pct_covered"] for r in f_reports if r.get("syllabus")]
        latest = sorted(f_reports, key=lambda r: r.get("session_datetime", ""), reverse=True)[0]
        rows.append({
            "faculty": faculty,
            "session_count": len(f_reports),
            "talk_ratio_teacher": _avg(talk_ratios),
            "talk_ratio_student": round(100 - _avg(talk_ratios), 1) if talk_ratios else 0.0,
            "syllabus_coverage": _avg(syllabus_pcts),
            "cadence_score": _avg([r["quality_index"] for r in f_reports if r.get("quality_index") is not None]) * 10,
            "latest_session_id": latest["session_id"],
        })
    rows.sort(key=lambda r: r["cadence_score"], reverse=True)

    dept_mean_talk = _avg([r["talk_ratio_teacher"] for r in rows]) if rows else 0.0
    return {
        "subject": subject,
        "rows": rows,
        "dept_mean_talk_ratio": dept_mean_talk,
        "session_total": len(reports),
    }
