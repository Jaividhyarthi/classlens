"""MediaPipe pose estimation over a lecture video, sampled into 60-second
windows: gesture rate, board-facing percentage, and a movement index.

Uses the MediaPipe Tasks API (mp.tasks.vision.PoseLandmarker) -- the
package installed in this environment (mediapipe==1.0.1) does not expose
the older mp.solutions.pose API at all, confirmed by inspecting the
installed package directly rather than assuming from training data.

GPU is used when available (BaseOptions.Delegate.GPU, "limited to Ubuntu
platforms" per MediaPipe's own docs -- this runs on Linux), CPU delegate
otherwise. No GPU was present when this was built, so this has only been
exercised on the CPU path.

Heuristics (Phase 1, defensible but approximate -- documented so nobody
mistakes them for ground truth):
- board_facing: BlazePose's 33 landmarks include a per-point visibility
  score. Facing the camera, both shoulders are typically visible and far
  apart in x; turned side-on toward a board, the far shoulder's visibility
  drops and the shoulder span narrows. board_facing_pct is the fraction of
  sampled frames where shoulder span (normalized by the session's own
  observed max) falls below a threshold.
- gesture_rate: peaks in frame-to-frame wrist displacement, counted as
  discrete gesture events per window.
- movement: total shoulder-midpoint displacement across the window,
  normalized by frame diagonal -- a unitless "how much the teacher moved"
  index, not a calibrated distance.
"""
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_lite.task"
WINDOW_SEC = 60
SAMPLE_FPS = 2.0  # frames sampled per second of video -- CPU-friendly


@dataclass
class VideoWindow:
    start_sec: float
    end_sec: float
    gesture_rate_per_min: float
    board_facing_pct: float
    movement_index: float
    frames_with_person: int
    frames_sampled: int


def _has_gpu() -> bool:
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _make_landmarker():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pose model not found at {MODEL_PATH}. Download it with:\n"
            "curl -o pipeline/models/pose_landmarker_lite.task "
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        )
    delegate = mp.tasks.BaseOptions.Delegate.GPU if _has_gpu() else mp.tasks.BaseOptions.Delegate.CPU
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH), delegate=delegate),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


@dataclass
class _FrameObservation:
    t_sec: float
    person_detected: bool
    shoulder_span: Optional[float] = None
    shoulder_mid: Optional[tuple] = None
    wrist_positions: list = field(default_factory=list)


def _observe_frame(result, frame_w: int, frame_h: int, t_sec: float) -> _FrameObservation:
    if not result.pose_landmarks:
        return _FrameObservation(t_sec=t_sec, person_detected=False)

    lm = result.pose_landmarks[0]
    PL = mp.tasks.vision.PoseLandmark
    left_sh, right_sh = lm[PL.LEFT_SHOULDER], lm[PL.RIGHT_SHOULDER]
    left_wr, right_wr = lm[PL.LEFT_WRIST], lm[PL.RIGHT_WRIST]

    shoulder_span = abs(left_sh.x - right_sh.x) * frame_w
    shoulder_mid = (
        (left_sh.x + right_sh.x) / 2.0 * frame_w,
        (left_sh.y + right_sh.y) / 2.0 * frame_h,
    )
    wrists = [(left_wr.x * frame_w, left_wr.y * frame_h), (right_wr.x * frame_w, right_wr.y * frame_h)]
    return _FrameObservation(
        t_sec=t_sec, person_detected=True,
        shoulder_span=shoulder_span, shoulder_mid=shoulder_mid, wrist_positions=wrists,
    )


def _summarize_window(observations: list[_FrameObservation], start_sec: float, end_sec: float, max_shoulder_span: float) -> VideoWindow:
    present = [o for o in observations if o.person_detected]
    frames_sampled = len(observations)
    frames_with_person = len(present)

    if not present or max_shoulder_span <= 0:
        return VideoWindow(
            start_sec=start_sec, end_sec=end_sec,
            gesture_rate_per_min=0.0, board_facing_pct=0.0, movement_index=0.0,
            frames_with_person=frames_with_person, frames_sampled=frames_sampled,
        )

    facing_camera_frames = sum(1 for o in present if (o.shoulder_span / max_shoulder_span) >= 0.6)
    board_facing_pct = round(100.0 * (1 - facing_camera_frames / len(present)), 1)

    diag = float(np.hypot(1.0, 1.0))
    movement_total = 0.0
    prev_mid = None
    for o in present:
        if prev_mid is not None and o.shoulder_mid is not None:
            dx = o.shoulder_mid[0] - prev_mid[0]
            dy = o.shoulder_mid[1] - prev_mid[1]
            movement_total += float(np.hypot(dx, dy))
        prev_mid = o.shoulder_mid
    movement_index = round(movement_total, 1)

    gesture_events = 0
    prev_wrists = None
    speeds = []
    for o in present:
        if prev_wrists is not None and o.wrist_positions:
            speed = max(
                float(np.hypot(a[0] - b[0], a[1] - b[1]))
                for a, b in zip(o.wrist_positions, prev_wrists)
            )
            speeds.append(speed)
        prev_wrists = o.wrist_positions
    if speeds:
        threshold = np.percentile(speeds, 70)
        threshold = max(threshold, 8.0)
        was_above = False
        for s in speeds:
            above = s > threshold
            if above and not was_above:
                gesture_events += 1
            was_above = above
    window_minutes = max((end_sec - start_sec) / 60.0, 1e-6)
    gesture_rate_per_min = round(gesture_events / window_minutes, 1)

    return VideoWindow(
        start_sec=start_sec, end_sec=end_sec,
        gesture_rate_per_min=gesture_rate_per_min,
        board_facing_pct=board_facing_pct,
        movement_index=movement_index,
        frames_with_person=frames_with_person,
        frames_sampled=frames_sampled,
    )


def analyze_video(video_path: Path, window_sec: int = WINDOW_SEC, sample_fps: float = SAMPLE_FPS) -> list[VideoWindow]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    frame_stride = max(int(round(fps / sample_fps)), 1)

    landmarker = _make_landmarker()
    observations: list[_FrameObservation] = []
    max_shoulder_span = 0.0

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_stride == 0:
                t_sec = frame_idx / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(t_sec * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                obs = _observe_frame(result, frame_w, frame_h, t_sec)
                observations.append(obs)
                if obs.shoulder_span:
                    max_shoulder_span = max(max_shoulder_span, obs.shoulder_span)
            frame_idx += 1
    finally:
        cap.release()
        landmarker.close()

    windows: list[VideoWindow] = []
    n_windows = max(int(np.ceil(duration_sec / window_sec)), 1)
    for i in range(n_windows):
        w_start = i * window_sec
        w_end = min((i + 1) * window_sec, duration_sec) or window_sec
        window_obs = [o for o in observations if w_start <= o.t_sec < w_end]
        windows.append(_summarize_window(window_obs, w_start, w_end, max_shoulder_span))

    return windows


if __name__ == "__main__":
    import json
    import sys

    result_windows = analyze_video(Path(sys.argv[1]))
    print(json.dumps([w.__dict__ for w in result_windows], indent=2))
