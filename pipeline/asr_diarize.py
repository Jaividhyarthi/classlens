"""Audio extraction + Sarvam batch ASR with speaker diarization.

Produces a flat list of diarized segments for a lecture recording:
[{start_sec, end_sec, speaker_id, text}, ...]

speaker_id is Sarvam's raw label (e.g. "SPEAKER_00"); pipeline/aggregate.py
decides which speaker_id is "the teacher" (the speaker with the most total
talk time is almost always the teacher in a classroom recording -- one
instructor, many students who each speak briefly).
"""
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pipeline.sarvam_client import SpeechToTextBatchJob


@dataclass
class DiarizedSegment:
    start_sec: float
    end_sec: float
    speaker_id: str
    text: str


@dataclass
class TranscriptResult:
    full_transcript: str
    language_code: Optional[str]
    segments: list[DiarizedSegment]


def extract_audio(video_path: Path, out_path: Optional[Path] = None) -> Path:
    """Mono 16kHz PCM WAV -- the format Sarvam's docs say to pass
    input_audio_codec for explicitly, and the smallest reliable upload."""
    if out_path is None:
        out_path = video_path.with_suffix(".wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr.strip()}")
    return out_path


def _parse_batch_output(raw: dict) -> TranscriptResult:
    """Parses the batch job's downloaded output JSON.

    request_id/transcript/timestamps/language_code/language_probability are
    confirmed fields (sarvamai/types/speech_to_text_response.py). The
    diarized_transcript.entries nesting -- {transcript, start_time_seconds,
    end_time_seconds, speaker_id} -- is taken from sarvamai/types/
    diarized_transcript.py and diarized_entry.py, which exist specifically
    for this purpose, but I could not confirm the exact top-level key name
    against a live batch response (no API key in this environment, and
    docs.sarvam.ai is network-blocked here). Parsing is defensive: if the
    expected shape isn't there, this raises with the raw payload rather than
    silently returning empty segments -- run this once against a real key
    and fix the key name here if Sarvam's actual wire format differs.
    """
    full_transcript = raw.get("transcript", "")
    language_code = raw.get("language_code")

    diarized = raw.get("diarized_transcript")
    if not diarized or "entries" not in diarized:
        raise ValueError(
            "Batch output has no diarized_transcript.entries -- either "
            "with_diarization wasn't honored, or Sarvam's real field name "
            f"differs from what's documented in the SDK types. Raw keys: {list(raw.keys())}"
        )

    segments = [
        DiarizedSegment(
            start_sec=float(e["start_time_seconds"]),
            end_sec=float(e["end_time_seconds"]),
            speaker_id=str(e["speaker_id"]),
            text=e.get("transcript", ""),
        )
        for e in diarized["entries"]
    ]
    return TranscriptResult(full_transcript=full_transcript, language_code=language_code, segments=segments)


def transcribe_and_diarize(
    video_or_audio_path: Path,
    num_speakers: int = 2,
    language_code: str = "unknown",
    model: str = "saaras:v3",
    poll_interval: float = 5.0,
    overall_timeout: float = 900.0,
) -> TranscriptResult:
    path = Path(video_or_audio_path)
    audio_path = path
    cleanup = False
    if path.suffix.lower() not in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"):
        audio_path = extract_audio(path)
        cleanup = True

    try:
        job = SpeechToTextBatchJob(
            audio_path=audio_path,
            language_code=language_code,
            model=model,
            num_speakers=num_speakers,
            input_audio_codec="pcm_s16le" if audio_path.suffix.lower() == ".wav" else None,
        )
        raw_output = job.run(poll_interval=poll_interval, overall_timeout=overall_timeout)
        return _parse_batch_output(raw_output)
    finally:
        if cleanup and audio_path.exists():
            audio_path.unlink()


if __name__ == "__main__":
    import sys

    result = transcribe_and_diarize(Path(sys.argv[1]))
    print(json.dumps({
        "language_code": result.language_code,
        "transcript": result.full_transcript,
        "segments": [s.__dict__ for s in result.segments],
    }, indent=2))
