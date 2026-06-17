"""
Real-Time Hand Gesture Recognition System
==========================================
Compatible with mediapipe >= 0.10.30 (new Tasks API)

Supported gestures:
  - Thumbs Up
  - Thumbs Down
  - Fist

Setup:
  pip install mediapipe opencv-python numpy requests

On first run, the hand landmarker model is auto-downloaded (~8 MB).
"""

import cv2
import numpy as np
import time
import sys
import os
import urllib.request

# ── Auto-download the MediaPipe hand landmarker model ──────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print(f"[INFO] Downloading hand landmarker model (~8 MB) ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[INFO] Model saved to '{MODEL_PATH}'")
    except Exception as e:
        sys.exit(f"[ERROR] Failed to download model: {e}\n"
                 f"        Download manually from:\n        {MODEL_URL}\n"
                 f"        and place it next to this script as '{MODEL_PATH}'")

download_model()

# ── MediaPipe Tasks import ─────────────────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp

# ── Landmark indices ───────────────────────────────────────────────────────
WRIST       = 0
THUMB_CMC   = 1
THUMB_MCP   = 2
THUMB_IP    = 3
THUMB_TIP   = 4
INDEX_MCP   = 5
INDEX_PIP   = 6
INDEX_TIP   = 8
MIDDLE_MCP  = 9
MIDDLE_PIP  = 10
MIDDLE_TIP  = 12
RING_MCP    = 13
RING_PIP    = 14
RING_TIP    = 16
PINKY_MCP   = 17
PINKY_PIP   = 18
PINKY_TIP   = 20

# ── Visual config ──────────────────────────────────────────────────────────
GESTURE_COLORS = {
    "Thumbs Up":   (50,  205,  50),
    "Thumbs Down": (30,  30,  220),
    "Fist":        (0,   165, 255),
    "Unknown":     (180, 180, 180),
}
FONT       = cv2.FONT_HERSHEY_DUPLEX
THICKNESS  = 2

# ── Gesture classifier ─────────────────────────────────────────────────────
def classify_gesture(landmarks) -> str:
    """
    landmarks: list of 21 NormalizedLandmark objects (x, y, z all 0-1)
    """
    def y(idx): return landmarks[idx].y   # y increases downward

    # Finger fold check (tip y > pip y means folded, since y grows downward)
    index_folded  = y(INDEX_TIP)  > y(INDEX_PIP)
    middle_folded = y(MIDDLE_TIP) > y(MIDDLE_PIP)
    ring_folded   = y(RING_TIP)   > y(RING_PIP)
    pinky_folded  = y(PINKY_TIP)  > y(PINKY_PIP)
    all_folded    = index_folded and middle_folded and ring_folded and pinky_folded

    # Hand height for threshold calibration
    hand_height = abs(y(WRIST) - y(MIDDLE_MCP)) + 1e-6
    threshold   = 0.20 * hand_height

    thumb_up   = (y(THUMB_MCP) - y(THUMB_TIP)) > threshold  # tip above MCP
    thumb_down = (y(THUMB_TIP) - y(WRIST))     > threshold  # tip below wrist
    thumb_folded = y(THUMB_TIP) > y(THUMB_IP)

    if all_folded and thumb_up:
        return "Thumbs Up"
    if all_folded and thumb_down:
        return "Thumbs Down"
    if all_folded and thumb_folded:
        return "Fist"
    return "Unknown"

# ── Drawing helpers ────────────────────────────────────────────────────────
mp_drawing       = mp.solutions.drawing_utils
mp_drawing_styles= mp.solutions.drawing_styles
mp_hands_conn    = mp.solutions.hands

def draw_landmarks_on_frame(frame, landmarks):
    """Draw MediaPipe hand skeleton using the classic drawing utils."""
    proto = landmark_pb2.NormalizedLandmarkList()
    proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
        for lm in landmarks
    ])
    mp_drawing.draw_landmarks(
        frame, proto,
        mp_hands_conn.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )

def draw_gesture_label(frame, gesture: str, landmarks, img_w: int, img_h: int):
    color = GESTURE_COLORS.get(gesture, GESTURE_COLORS["Unknown"])
    xs = [lm.x * img_w for lm in landmarks]
    ys = [lm.y * img_h for lm in landmarks]
    x_min, y_min = int(min(xs)), int(min(ys))

    (tw, th), baseline = cv2.getTextSize(gesture, FONT, 1.0, THICKNESS)
    pad = 8
    cv2.rectangle(frame,
                  (x_min - pad,      y_min - th - 2 * pad),
                  (x_min + tw + pad, y_min - pad),
                  color, cv2.FILLED)
    cv2.putText(frame, gesture,
                (x_min, y_min - pad - baseline // 2),
                FONT, 1.0, (255, 255, 255), THICKNESS, cv2.LINE_AA)

def draw_hud(frame, fps: float, gesture: str, frame_idx: int):
    h, w = frame.shape[:2]
    color = GESTURE_COLORS.get(gesture, GESTURE_COLORS["Unknown"])
    banner_h = 54
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"FPS: {fps:5.1f}",
                (14, h - 18), FONT, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_idx}",
                (14, h - 36), FONT, 0.55, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.putText(frame, gesture,
                (w // 2 - 80, h - 14), FONT, 0.85, color, 2, cv2.LINE_AA)
    cv2.putText(frame, "Hand Gesture Recognition",
                (10, 30), FONT, 0.7, (220, 220, 220), 1, cv2.LINE_AA)

# ── Video writer ───────────────────────────────────────────────────────────
def make_writer(path, fps, width, height):
    for codec in ("avc1", "mp4v", "XVID"):
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            print(f"[INFO] VideoWriter: codec='{codec}' → {path}")
            return writer
    raise RuntimeError("No working video codec found.")

# ── Main loop ──────────────────────────────────────────────────────────────
def run(
    camera_index : int   = 0,
    output_path  : str   = "gesture_output.mp4",
    max_hands    : int   = 2,
    min_det_conf : float = 0.7,
    min_trk_conf : float = 0.6,
    display      : bool  = True,
):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {camera_index}.")

    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Camera: {width}×{height} @ {src_fps:.1f} FPS")

    writer = make_writer(output_path, src_fps, width, height)

    # Build the HandLandmarker (new Tasks API)
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options   = HandLandmarkerOptions(
        base_options            = base_opts,
        running_mode            = RunningMode.VIDEO,
        num_hands               = max_hands,
        min_hand_detection_confidence = min_det_conf,
        min_tracking_confidence       = min_trk_conf,
    )

    fps_window  = []
    frame_idx   = 0
    last_ts     = time.perf_counter()
    current_fps = 0.0

    with HandLandmarker.create_from_options(options) as detector:
        print("[INFO] Running — press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # FPS
            now = time.perf_counter()
            fps_window.append(1.0 / max(now - last_ts, 1e-6))
            if len(fps_window) > 30:
                fps_window.pop(0)
            current_fps = float(np.mean(fps_window))
            last_ts = now

            # Detect — Tasks API requires an mp.Image and timestamp in ms
            mp_image    = mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(time.time() * 1000)
            result       = detector.detect_for_video(mp_image, timestamp_ms)

            best_gesture = "Unknown"

            if result.hand_landmarks:
                for hand_lms in result.hand_landmarks:
                    draw_landmarks_on_frame(frame, hand_lms)
                    gesture = classify_gesture(hand_lms)
                    draw_gesture_label(frame, gesture, hand_lms, width, height)
                    if gesture != "Unknown":
                        best_gesture = gesture

            draw_hud(frame, current_fps, best_gesture, frame_idx)
            writer.write(frame)

            if display:
                cv2.imshow("Hand Gesture Recognition  [q=quit]", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved '{output_path}' ({frame_idx} frames).")

# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--camera",      type=int,   default=0)
    p.add_argument("--output",      type=str,   default="gesture_output.mp4")
    p.add_argument("--hands",       type=int,   default=2)
    p.add_argument("--det-conf",    type=float, default=0.7)
    p.add_argument("--trk-conf",    type=float, default=0.6)
    p.add_argument("--no-display",  action="store_true")
    args = p.parse_args()

    run(
        camera_index = args.camera,
        output_path  = args.output,
        max_hands    = args.hands,
        min_det_conf = args.det_conf,
        min_trk_conf = args.trk_conf,
        display      = not args.no_display,
    )