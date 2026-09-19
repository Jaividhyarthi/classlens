"""Classifies transcript segments against the syllabus into exactly five
categories, via a Sarvam chat model applied directly to the (possibly
code-mixed Tamil/English) text -- no translation step, per the brief.

Only "genuine_off_topic" and confident classifications are ever surfaced
as flags; "low_confidence" segments are excluded from scoring, never
guessed into one of the other four categories.
"""
import json
from dataclasses import dataclass

from pipeline.asr_diarize import DiarizedSegment
from pipeline.sarvam_client import SarvamAPIError, chat_completion

CATEGORIES = [
    "on_topic_instructional",
    "analogy_example",
    "admin_management",
    "genuine_off_topic",
    "low_confidence",
]

BATCH_SIZE = 25
LOW_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MODEL = "sarvam-m"

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string"},
                },
                "required": ["index", "category", "confidence", "rationale"],
            },
        }
    },
    "required": ["classifications"],
}


@dataclass
class RelevanceResult:
    start_sec: float
    end_sec: float
    speaker_id: str
    text: str
    category: str
    confidence: float
    rationale: str


def _build_prompt(batch: list[DiarizedSegment], subject: str, syllabus_topics: list[str]) -> str:
    topics_str = ", ".join(syllabus_topics) if syllabus_topics else "(no syllabus topics configured)"
    lines = [
        f"Subject: {subject}",
        f"Syllabus topics for this unit: {topics_str}",
        "",
        "Classify each numbered classroom transcript segment below into exactly one category:",
        "- on_topic_instructional: directly teaching syllabus content",
        "- analogy_example: an analogy, real-world example, or illustration used to explain syllabus content",
        "- admin_management: classroom administration (attendance, announcements, discipline, scheduling)",
        "- genuine_off_topic: content unrelated to the subject and not administrative",
        "- low_confidence: transcript is too garbled, short, or ambiguous to classify confidently",
        "",
        "The transcript may be code-mixed Tamil and English. Classify it as-is -- do not translate.",
        "If a segment sounds off-topic but could plausibly be a classroom analogy or example, "
        "prefer analogy_example over genuine_off_topic unless it's clearly unrelated.",
        "When unsure, use low_confidence rather than guessing.",
        "",
        "Segments:",
    ]
    for i, seg in enumerate(batch):
        lines.append(f"[{i}] ({seg.speaker_id}): {seg.text}")
    return "\n".join(lines)


def _classify_batch(batch: list[DiarizedSegment], subject: str, syllabus_topics: list[str], model: str) -> list[RelevanceResult]:
    prompt = _build_prompt(batch, subject, syllabus_topics)
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "segment_classifications", "schema": _RESPONSE_SCHEMA, "strict": True},
    }
    try:
        content = chat_completion(
            messages=[
                {"role": "system", "content": "You are an academic quality reviewer classifying classroom transcript segments. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.0,
            response_format=response_format,
        )
        parsed = json.loads(content)
        by_index = {c["index"]: c for c in parsed.get("classifications", [])}
    except (SarvamAPIError, json.JSONDecodeError, KeyError):
        # A failed or malformed classification call must not silently score
        # segments as on-topic -- mark the whole batch low_confidence instead.
        by_index = {}

    results = []
    for i, seg in enumerate(batch):
        c = by_index.get(i)
        if c and c.get("category") in CATEGORIES:
            category = c["category"]
            confidence = float(c.get("confidence", 0.0))
            rationale = c.get("rationale", "")
        else:
            category = "low_confidence"
            confidence = 0.0
            rationale = "Classifier did not return a result for this segment."
        results.append(RelevanceResult(
            start_sec=seg.start_sec, end_sec=seg.end_sec, speaker_id=seg.speaker_id,
            text=seg.text, category=category, confidence=confidence, rationale=rationale,
        ))
    return results


def classify_segments(
    segments: list[DiarizedSegment],
    subject: str,
    syllabus_topics: list[str],
    model: str = DEFAULT_MODEL,
    batch_size: int = BATCH_SIZE,
) -> list[RelevanceResult]:
    teaching_segments = [s for s in segments if s.text.strip()]
    results: list[RelevanceResult] = []
    for start in range(0, len(teaching_segments), batch_size):
        batch = teaching_segments[start:start + batch_size]
        results.extend(_classify_batch(batch, subject, syllabus_topics, model))
    return results


def flagged_off_topic(results: list[RelevanceResult], min_confidence: float = LOW_CONFIDENCE_THRESHOLD) -> list[RelevanceResult]:
    """Only genuine_off_topic segments the model was reasonably confident
    about are ever surfaced as flags."""
    return [r for r in results if r.category == "genuine_off_topic" and r.confidence >= min_confidence]


def scorable_results(results: list[RelevanceResult]) -> list[RelevanceResult]:
    """low_confidence segments are excluded from any scoring/aggregation,
    never guessed into one of the other four categories."""
    return [r for r in results if r.category != "low_confidence"]
