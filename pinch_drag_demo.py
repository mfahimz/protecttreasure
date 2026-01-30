import time
import cv2 
import numpy as np
import mediapipe as mp 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision 

# -----------------------------
# MediaPipe model
# -----------------------------
MODEL_PATH = "models/hand_landmarker.task"

THUMB_TIP = 4
INDEX_TIP = 8
INDEX_MCP = 5
PINKY_MCP = 17


PINCH_DOWN = 0.45
PINCH_UP   = 0.60

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)

landmarker = vision.HandLandmarker.create_from_options(options)

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

t0 = time.time()

# -----------------------------
# Helper
# -----------------------------
def l2(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

# -----------------------------
# Game state
# -----------------------------
grab_radius = 45
score = 0

balls_initialized = False
grabbed_ball = None
pinch_active = False

last_pointer = None   # 🔑 key fix

# -----------------------------
# Main loop
# -----------------------------
while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break

    h, w = frame_bgr.shape[:2]

    # -----------------------------
    # Bucket
    # -----------------------------
    bucket_w, bucket_h = 260, 70
    bucket_x = (w - bucket_w) // 2
    bucket_y = h - bucket_h - 10

    # -----------------------------
    # Balls (lowered)
    # -----------------------------
    if not balls_initialized:
        ball_r = 20
        spacing = 110
        start_x = w // 2 - spacing
        top_y = 150

        balls = [
            {"x": start_x,             "y": top_y, "r": ball_r, "state": "IDLE"},
            {"x": start_x + spacing,   "y": top_y, "r": ball_r, "state": "IDLE"},
            {"x": start_x + spacing*2, "y": top_y, "r": ball_r, "state": "IDLE"},
        ]
        balls_initialized = True

    # -----------------------------
    # MediaPipe
    # -----------------------------
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int((time.time() - t0) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    pointer = None
    pinch_ratio = None

    # -----------------------------
    # Hand processing
    # -----------------------------
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        pts = [(lm.x * w, lm.y * h) for lm in hand]

        thumb_tip = pts[THUMB_TIP]
        index_tip = pts[INDEX_TIP]

        pointer = (int(index_tip[0]), int(index_tip[1]))
        last_pointer = pointer  # 🔑 update last known pointer

        scale = l2(pts[INDEX_MCP], pts[PINKY_MCP]) + 1e-6
        pinch_dist = l2(thumb_tip, index_tip)
        pinch_ratio = pinch_dist / scale

        # Visuals
        cv2.circle(frame_bgr, pointer, 6, (0, 255, 0), -1)
        cv2.line(
            frame_bgr,
            (int(thumb_tip[0]), int(thumb_tip[1])),
            (int(index_tip[0]), int(index_tip[1])),
            (0, 255, 255),
            2
        )

        # Pinch state machine
        if not pinch_active and pinch_ratio < PINCH_DOWN:
            pinch_active = True

        if pinch_active and pinch_ratio > PINCH_UP:
            pinch_active = False

        # Grab ONLY on pinch start
        if pinch_active and grabbed_ball is None:
            for ball in balls:
                if ball["state"] == "IDLE":
                    if l2(pointer, (ball["x"], ball["y"])) < grab_radius:
                        grabbed_ball = ball
                        ball["state"] = "GRABBED"
                        break

    # -----------------------------
    # Update balls (robust tracking)
    # -----------------------------
    for ball in balls:
        if ball["state"] == "GRABBED" and last_pointer:
            ball["x"] = last_pointer[0]
            ball["y"] = last_pointer[1]

            # Lock inside bucket
            if (
                bucket_x < ball["x"] < bucket_x + bucket_w
                and bucket_y < ball["y"] < bucket_y + bucket_h
            ):
                ball["state"] = "LOCKED"
                ball["y"] = bucket_y + bucket_h // 2
                score += 1
                grabbed_ball = None

    # -----------------------------
    # Draw balls
    # -----------------------------
    for ball in balls:
        cv2.circle(
            frame_bgr,
            (int(ball["x"]), int(ball["y"])),
            ball["r"],
            (0, 255, 255),
            -1
        )

    # -----------------------------
    # Draw bucket
    # -----------------------------
    cv2.rectangle(
        frame_bgr,
        (bucket_x, bucket_y),
        (bucket_x + bucket_w, bucket_y + bucket_h),
        (255, 0, 0),
        3
    )

    # -----------------------------
    # UI
    # -----------------------------
    cv2.putText(
        frame_bgr,
        f"Score: {score}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    if pinch_ratio is not None:
        cv2.putText(
            frame_bgr,
            f"Pinch: {pinch_ratio:.2f}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    cv2.imshow("Proper Pinch Ball Drop Game (stable tracking)", frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
