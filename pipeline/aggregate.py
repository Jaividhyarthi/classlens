"""Merges diarized audio segments (asr_diarize.py) and pose-derived video
windows (video_analysis.py) into one 60-second-window timeline -- the
single source of truth every report/dashboard number reads from.

active_syllabus_topic is left as None here: aggregate.py doesn't know the
syllabus, that's relevance.py's job (step 4 in the build order, which runs
after this). report.py fills it in per window once relevance classification
exists.
"""
import re
from dataclasses import asdict, dataclass
from typing import Optional

from pipeline.asr_diarize import TranscriptResult
from pipeline.video_analysis import VideoWindow

WINDOW_SEC = 60
_QUESTION_RE = re.compile(r"\?")


@dataclass
class TimelineWindow:
    start_sec: float
    end_sec: float
    teacher_talk_ratio: float  # % of spoken time in this window that was the teacher (0-100)
    wpm: float  # teacher words per minute in this window
    pause_pct: float  # % of window with nobody speaking
    gesture_rate_per_min: float
    board_facing_pct: float
    question_count: int
    active_syllabus_topic: Optional[str]
    transcript_confidence: Optional[float]


def identify_teacher_speaker(transcript: TranscriptResult) -> Optional[str]:
    """The teacher is the speaker with the most total talk time -- one
    instructor talks continuously, students each speak briefly. This
    breaks down for whole-class choral responses or a co-taught session,
    which Phase 1 doesn't try to handle."""
    talk_time: dict[str, float] = {}
    for seg in transcript.segments:
        talk_time[seg.speaker_id] = talk_time.get(seg.speaker_id, 0.0) + max(seg.end_sec - seg.start_sec, 0.0)
    if not talk_time:
        return None
    return max(talk_time, key=talk_time.get)


def _overlap_sec(seg_start: float, seg_end: float, win_start: float, win_end: float) -> float:
    return max(0.0, min(seg_end, win_end) - max(seg_start, win_start))


def build_timeline(
    transcript: TranscriptResult,
    video_windows: list[VideoWindow],
    duration_sec: Optional[float] = None,
    window_sec: int = WINDOW_SEC,
) -> list[TimelineWindow]:
    teacher_id = identify_teacher_speaker(transcript)

    if duration_sec is None:
        transcript_end = max((s.end_sec for s in transcript.segments), default=0.0)
        video_end = max((w.end_sec for w in video_windows), default=0.0)
        duration_sec = max(transcript_end, video_end)

    n_windows = max(int((duration_sec + window_sec - 1) // window_sec), 1)
    video_by_start = {round(w.start_sec): w for w in video_windows}

    timeline: list[TimelineWindow] = []
    for i in range(n_windows):
        w_start = i * window_sec
        w_end = min((i + 1) * window_sec, duration_sec) or float(window_sec)
        win_duration = max(w_end - w_start, 1e-6)

        teacher_talk_sec = 0.0
        total_talk_sec = 0.0
        teacher_words = 0
        question_count = 0

        for seg in transcript.segments:
            overlap = _overlap_sec(seg.start_sec, seg.end_sec, w_start, w_end)
            if overlap <= 0:
                continue
            total_talk_sec += overlap
            if seg.speaker_id == teacher_id:
                teacher_talk_sec += overlap
                seg_duration = max(seg.end_sec - seg.start_sec, 1e-6)
                word_count = len(seg.text.split())
                teacher_words += word_count * (overlap / seg_duration)
            question_count += len(_QUESTION_RE.findall(seg.text)) if overlap > 0 else 0

        teacher_talk_ratio = round(100.0 * teacher_talk_sec / total_talk_sec, 1) if total_talk_sec > 0 else 0.0
        pause_pct = round(100.0 * max(win_duration - total_talk_sec, 0.0) / win_duration, 1)
        wpm = round(teacher_words / (win_duration / 60.0), 1) if teacher_talk_sec > 0 else 0.0

        video_window = video_by_start.get(w_start)
        gesture_rate = video_window.gesture_rate_per_min if video_window else 0.0
        board_facing = video_window.board_facing_pct if video_window else 0.0

        timeline.append(TimelineWindow(
            start_sec=float(w_start),
            end_sec=float(w_end),
            teacher_talk_ratio=teacher_talk_ratio,
            wpm=wpm,
            pause_pct=pause_pct,
            gesture_rate_per_min=gesture_rate,
            board_facing_pct=board_facing,
            question_count=question_count,
            active_syllabus_topic=None,
            transcript_confidence=transcript.language_probability,
        ))

    return timeline


def timeline_to_dicts(timeline: list[TimelineWindow]) -> list[dict]:
    return [asdict(w) for w in timeline]
