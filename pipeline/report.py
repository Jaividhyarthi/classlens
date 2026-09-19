"""Turns a timeline (aggregate.py) + relevance classification (relevance.py)
into exactly the data shape web/templates/report.html and the dashboards
read, plus LLM-generated coaching insights that cite the specific metric
behind each observation.

Every number here traces back to a real computed value -- there is no
literal magic constant standing in for a metric. The two places this
module makes an editorial call rather than reading a raw number are
documented inline: the quality_index formula, and the "instructional
focus" proxy used for syllabus pct_covered (fine-grained per-topic
coverage isn't implemented -- relevance.py classifies into 5 broad
categories, not "which of these 7 syllabus topics").
"""
import json
from dataclasses import asdict
from typing import Optional

from pipeline.aggregate import TimelineWindow
from pipeline.asr_diarize import DiarizedSegment, TranscriptResult
from pipeline.relevance import RelevanceResult, flagged_off_topic, scorable_results
from pipeline.sarvam_client import SarvamAPIError, SarvamConfigError, chat_completion

TIMELINE_COLORS = {
    "teacher_monologue": "bg-tertiary-container/70",
    "board_derivation": "bg-outline-variant/80",
    "student_interaction": "bg-primary-container",
    "idle_transition": "bg-amber-500/70",
}
TIMELINE_LABELS = {
    "teacher_monologue": "Teacher Monologue",
    "board_derivation": "Board Derivation",
    "student_interaction": "Student Interaction",
    "idle_transition": "Idle / Transition",
}


def _fmt_time(sec: float) -> str:
    sec = max(int(sec), 0)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _classify_window_category(window: TimelineWindow) -> str:
    if window.pause_pct > 60:
        return "idle_transition"
    if window.question_count > 0 or window.teacher_talk_ratio < 60:
        return "student_interaction"
    if window.board_facing_pct > 55:
        return "board_derivation"
    return "teacher_monologue"


def _weighted_avg(pairs: list[tuple[float, float]]) -> float:
    """pairs of (value, weight)."""
    total_weight = sum(w for _, w in pairs)
    if total_weight <= 0:
        return 0.0
    return sum(v * w for v, w in pairs) / total_weight


def build_timeline_segments(timeline: list[TimelineWindow], duration_sec: float) -> tuple[list[dict], list[dict]]:
    if duration_sec <= 0 or not timeline:
        return [], []

    segments = []
    category_duration: dict[str, float] = {}
    for w in timeline:
        category = _classify_window_category(w)
        dur = w.end_sec - w.start_sec
        category_duration[category] = category_duration.get(category, 0.0) + dur
        segments.append({
            "start_pct": round(100.0 * w.start_sec / duration_sec, 2),
            "width_pct": round(100.0 * dur / duration_sec, 2),
            "color_class": TIMELINE_COLORS[category],
            "label": TIMELINE_LABELS[category],
        })

    legend = [
        {
            "label": TIMELINE_LABELS[cat],
            "pct": round(100.0 * dur / duration_sec, 0),
            "swatch_class": TIMELINE_COLORS[cat],
        }
        for cat, dur in sorted(category_duration.items(), key=lambda kv: -kv[1])
    ]
    return segments, legend


def compute_wait_times(transcript: TranscriptResult, max_wait: float = 20.0) -> list[dict]:
    """For each teacher question, the gap before the next speaker segment
    starts. Returns [{start_sec, wait_sec, question_text}], sorted by time."""
    events = []
    for seg in transcript.segments:
        if not seg.text.strip().endswith("?"):
            continue
        next_start = None
        for other in transcript.segments:
            if other.start_sec > seg.end_sec and (next_start is None or other.start_sec < next_start):
                next_start = other.start_sec
        if next_start is None:
            continue
        wait = next_start - seg.end_sec
        if 0 < wait < max_wait:
            events.append({"start_sec": seg.start_sec, "wait_sec": round(wait, 1), "question_text": seg.text.strip()})
    events.sort(key=lambda e: e["start_sec"])
    return events


def build_cues(transcript: TranscriptResult, relevance_results: list[RelevanceResult], duration_sec: float, max_cues: int = 5) -> list[dict]:
    cues = []

    for event in compute_wait_times(transcript):
        cues.append({
            "pct": round(100.0 * event["start_sec"] / duration_sec, 2) if duration_sec else 0,
            "time_label": _fmt_time(event["start_sec"]),
            "tag": "Question & Wait Time" if event["wait_sec"] < 3 else "Cognitive Anchor Detected",
            "text": f'"{event["question_text"]}" -- {event["wait_sec"]}s pause before the next response.',
        })

    for r in flagged_off_topic(relevance_results):
        cues.append({
            "pct": round(100.0 * r.start_sec / duration_sec, 2) if duration_sec else 0,
            "time_label": _fmt_time(r.start_sec),
            "tag": "Off-Topic Deviation Flagged",
            "text": f'"{r.text.strip()}" -- classified genuine_off_topic (confidence {r.confidence:.2f}).',
        })

    cues.sort(key=lambda c: c["pct"])
    if not cues:
        cues = [{
            "pct": 2, "time_label": "00:00",
            "tag": "No notable moments detected",
            "text": "No question-and-wait moments or off-topic deviations were flagged in this session.",
        }]
    return cues[:max_cues]


def build_metrics(timeline: list[TimelineWindow], wait_events: list[dict]) -> dict:
    weighted = [(w.teacher_talk_ratio, w.end_sec - w.start_sec) for w in timeline]
    talk_ratio_teacher = round(_weighted_avg(weighted), 1)
    talk_ratio_student = round(100 - talk_ratio_teacher, 1) if timeline else 0.0

    wpm_pairs = [(w.wpm, w.end_sec - w.start_sec) for w in timeline if w.wpm > 0]
    wpm = round(_weighted_avg(wpm_pairs), 0) if wpm_pairs else 0.0

    wait_times = [e["wait_sec"] for e in wait_events]
    avg_wait_time = round(sum(wait_times) / len(wait_times), 1) if wait_times else 0.0

    board_pairs = [(w.board_facing_pct, w.end_sec - w.start_sec) for w in timeline]
    board_presence = round(_weighted_avg(board_pairs), 0)

    return {
        "talk_ratio_teacher_pct": talk_ratio_teacher,
        "talk_ratio_student_pct": talk_ratio_student,
        "wpm": wpm,
        "wpm_note": "Well-paced articulation" if 110 <= wpm <= 160 else ("Slow pace" if wpm < 110 and wpm > 0 else "Fast pace" if wpm > 160 else "Insufficient teacher speech"),
        "wpm_bar_pct": min(round(wpm / 200 * 100, 0), 100) if wpm else 0,
        "avg_wait_time_sec": avg_wait_time,
        "avg_wait_time_note": f"From {len(wait_times)} observed question(s)" if wait_times else "No question-and-wait moments detected",
        "avg_wait_time_bar_pct": min(round(avg_wait_time / 10 * 100, 0), 100),
        "board_presence_pct": board_presence,
        "board_presence_note": "High visual engagement" if board_presence >= 50 else "Mostly facing the class",
    }


def compute_quality_index(metrics: dict, syllabus: dict, off_topic_count: int, question_count: int) -> tuple[float, str]:
    """0-10 composite. Weights and bands are a documented editorial choice,
    not a validated psychometric instrument -- shown here so the number in
    the UI is reconstructable, not a black box.

    - talk balance (30%): 10 at teacher_talk_ratio in [55,70], decaying
      linearly to 0 at the edges (0% or 100%)
    - syllabus pace (30%): pct_covered / scheduled_pct, capped at 10
    - engagement (25%): 10 at >= 1 question per 10 minutes of session, else
      scaled down linearly
    - off-topic penalty (15%): 10 if zero flagged off-topic segments, -2.5
      per flagged segment down to 0
    """
    ratio = metrics["talk_ratio_teacher_pct"]
    if 55 <= ratio <= 70:
        talk_score = 10.0
    elif ratio < 55:
        talk_score = max(0.0, 10.0 * ratio / 55.0)
    else:
        talk_score = max(0.0, 10.0 * (100 - ratio) / 30.0)

    scheduled = syllabus.get("scheduled_pct", 1) or 1
    syllabus_score = min(10.0, 10.0 * syllabus["pct_covered"] / scheduled)

    duration_min = syllabus.get("_duration_min", 50)
    questions_per_10min = question_count / max(duration_min / 10.0, 1e-6)
    engagement_score = min(10.0, 10.0 * questions_per_10min)

    off_topic_score = max(0.0, 10.0 - 2.5 * off_topic_count)

    composite = 0.30 * talk_score + 0.30 * syllabus_score + 0.25 * engagement_score + 0.15 * off_topic_score
    composite = round(max(0.0, min(10.0, composite)), 1)

    if composite >= 9.0:
        grade = "Exemplary"
    elif composite >= 7.5:
        grade = "Strong"
    elif composite >= 6.0:
        grade = "Satisfactory"
    else:
        grade = "Needs Support"
    return composite, grade


def _fallback_coaching(metrics: dict, syllabus: dict, off_topic_count: int) -> list[dict]:
    """Used only if the Sarvam chat call fails -- still evidence-cited,
    just deterministic rather than model-written."""
    notes = []
    ratio = metrics["talk_ratio_teacher_pct"]
    if ratio > 75:
        notes.append({
            "icon": "record_voice_over",
            "title": "High Teacher Talk Ratio",
            "text": f"Teacher talk ratio was {ratio}% this session, above the 55-70% band associated with balanced dialogue. Consider more open questions to draw out student voice.",
        })
    elif ratio < 45:
        notes.append({
            "icon": "record_voice_over",
            "title": "Low Instructional Talk Share",
            "text": f"Teacher talk ratio was {ratio}%, which may indicate long silences or classroom management overhead rather than instruction.",
        })
    if syllabus["pct_covered"] < syllabus.get("scheduled_pct", 100):
        notes.append({
            "icon": "menu_book",
            "title": "Behind Scheduled Pace",
            "text": f"Instructional focus covered {syllabus['pct_covered']}% of scorable content against a {syllabus.get('scheduled_pct')}% scheduled target for this unit.",
        })
    if off_topic_count:
        notes.append({
            "icon": "flag",
            "title": "Off-Topic Segments Flagged",
            "text": f"{off_topic_count} segment(s) were classified genuine_off_topic by the relevance model this session.",
        })
    if not notes:
        notes.append({
            "icon": "verified",
            "title": "No Notable Deviations",
            "text": f"Talk ratio ({ratio}%), syllabus pace, and topic relevance were all within expected bands this session.",
        })
    return notes


def generate_coaching(metrics: dict, syllabus: dict, off_topic_count: int, model: str = "sarvam-m") -> list[dict]:
    schema = {
        "type": "object",
        "properties": {
            "insights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "icon": {"type": "string"},
                        "title": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["icon", "title", "text"],
                },
            }
        },
        "required": ["insights"],
    }
    prompt = (
        "You are an instructional coach for engineering faculty. Given these real computed "
        "metrics from one lecture, write 2-4 coaching observations. Each observation's `text` "
        "MUST cite the specific number that supports it -- never a bare judgement with no "
        "evidence. Be constructive, not punitive.\n\n"
        f"Teacher talk ratio: {metrics['talk_ratio_teacher_pct']}% (student: {metrics['talk_ratio_student_pct']}%)\n"
        f"Speech pace: {metrics['wpm']} wpm\n"
        f"Average question wait time: {metrics['avg_wait_time_sec']}s\n"
        f"Board-facing presence: {metrics['board_presence_pct']}%\n"
        f"Syllabus coverage (instructional focus): {syllabus['pct_covered']}% vs {syllabus.get('scheduled_pct')}% scheduled target\n"
        f"Off-topic segments flagged: {off_topic_count}\n"
    )
    try:
        content = chat_completion(
            messages=[
                {"role": "system", "content": "Respond with JSON only, matching the given schema."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.4,
            response_format={"type": "json_schema", "json_schema": {"name": "coaching_insights", "schema": schema, "strict": True}},
        )
        parsed = json.loads(content)
        insights = parsed.get("insights", [])
        if insights:
            return insights
    except (SarvamAPIError, SarvamConfigError, json.JSONDecodeError, KeyError):
        pass
    return _fallback_coaching(metrics, syllabus, off_topic_count)


def build_report(
    session_id: str,
    session_meta: dict,
    transcript: TranscriptResult,
    timeline: list[TimelineWindow],
    relevance_results: list[RelevanceResult],
    unit_config: Optional[dict] = None,
    coaching_model: str = "sarvam-m",
) -> dict:
    duration_sec = session_meta.get("duration_sec") or (timeline[-1].end_sec if timeline else 0.0)
    unit_config = unit_config or {}
    unit_topics = set(unit_config.get("topics", []))

    scorable = scorable_results(relevance_results)
    on_topic_sec = sum(
        r.end_sec - r.start_sec for r in scorable if r.category in ("on_topic_instructional", "analogy_example")
    )
    scorable_sec = sum(r.end_sec - r.start_sec for r in scorable) or 1e-6
    pct_covered = round(100.0 * on_topic_sec / scorable_sec, 1)
    scheduled_pct = unit_config.get("scheduled_pct_target", 75)
    delta = round(pct_covered - scheduled_pct, 1)
    syllabus = {
        "unit_id": unit_config.get("id"),
        "unit_name": unit_config.get("name", session_meta.get("subject", "Unit")),
        "pct_covered": pct_covered,
        "scheduled_pct": scheduled_pct,
        "delta_note": f"{'+' if delta >= 0 else ''}{delta}% {'ahead of' if delta >= 0 else 'behind'} academic calendar",
        "_duration_min": duration_sec / 60.0,
    }

    # fill in active_syllabus_topic per window now that relevance results exist
    windows = []
    for w in timeline:
        window_relevance = [r for r in scorable if r.start_sec < w.end_sec and r.end_sec > w.start_sec]
        on_topic_here = any(r.category in ("on_topic_instructional", "analogy_example") for r in window_relevance)
        w_dict = asdict(w)
        w_dict["active_syllabus_topic"] = syllabus["unit_name"] if on_topic_here else None
        windows.append(w_dict)

    off_topic = flagged_off_topic(relevance_results)
    off_topic_count = len(off_topic)
    low_confidence_count = sum(1 for r in relevance_results if r.category == "low_confidence")

    cues = build_cues(transcript, relevance_results, duration_sec)
    metrics = build_metrics(timeline, compute_wait_times(transcript))
    timeline_segments, legend = build_timeline_segments(timeline, duration_sec)
    total_questions = sum(w["question_count"] for w in windows)

    quality_index, quality_grade = compute_quality_index(metrics, syllabus, off_topic_count, total_questions)
    coaching = generate_coaching(metrics, syllabus, off_topic_count, model=coaching_model)

    topic_relevance = {
        "off_topic_count": off_topic_count,
        "low_confidence_count": low_confidence_count,
        "note": (
            "No off-topic deviations detected" if off_topic_count == 0
            else f"{off_topic_count} off-topic deviation(s) detected"
        ),
    }

    return {
        "session_id": session_id,
        "faculty_id": session_meta.get("faculty_id"),
        "faculty_name": session_meta.get("faculty_name"),
        "subject": session_meta.get("subject"),
        "year": session_meta.get("year"),
        "hall": session_meta.get("hall"),
        "session_datetime": session_meta.get("session_datetime") or session_meta.get("created_at"),
        "duration_sec": duration_sec,
        "status": "ready",
        "windows": windows,
        "timeline_segments": timeline_segments,
        "legend": legend,
        "cues": cues,
        "metrics": metrics,
        "coaching": coaching,
        "syllabus": syllabus,
        "topic_relevance": topic_relevance,
        "quality_index": quality_index,
        "quality_grade": quality_grade,
    }
