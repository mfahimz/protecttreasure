import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =============================
# MediaPipe setup
# =============================
MODEL_PATH = "models/hand_landmarker.task"

THUMB_TIP = 4
INDEX_TIP = 8
INDEX_MCP = 5
PINKY_MCP = 17

PINCH_DOWN = 0.45
PINCH_UP = 0.60

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)

landmarker = vision.HandLandmarker.create_from_options(options)

# =============================
# Webcam
# =============================
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

t0 = time.time()

# =============================
# Helpers
# =============================
def l2(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

# =============================
# Tic Tac Toe state
# =============================
board = [[None for _ in range(3)] for _ in range(3)]
current_player = "X"
game_over = False
pinch_active = False
pinch_latched = False

# =============================
# Game logic
# =============================
def check_winner():
    lines = []

    # rows & columns
    for i in range(3):
        lines.append(board[i])
        lines.append([board[0][i], board[1][i], board[2][i]])

    # diagonals
    lines.append([board[0][0], board[1][1], board[2][2]])
    lines.append([board[0][2], board[1][1], board[2][0]])

    for line in lines:
        if line[0] and line.count(line[0]) == 3:
            return line[0]

    if all(board[r][c] for r in range(3) for c in range(3)):
        return "DRAW"

    return None

# =============================
# Main loop
# =============================
while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int((time.time() - t0) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    pointer = None
    pinch_ratio = None

    # =============================
    # Hand tracking
    # =============================
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        pts = [(lm.x * w, lm.y * h) for lm in hand]

        thumb = pts[THUMB_TIP]
        index = pts[INDEX_TIP]
        pointer = (int(index[0]), int(index[1]))

        scale = l2(pts[INDEX_MCP], pts[PINKY_MCP]) + 1e-6
        pinch_dist = l2(thumb, index)
        pinch_ratio = pinch_dist / scale

        # Visuals
        cv2.circle(frame, pointer, 6, (0, 255, 0), -1)
        cv2.line(
            frame,
            (int(thumb[0]), int(thumb[1])),
            (int(index[0]), int(index[1])),
            (0, 255, 255),
            2,
        )

        # Pinch detection (debounced)
        if not pinch_active and pinch_ratio < PINCH_DOWN:
            pinch_active = True
            pinch_latched = True

        if pinch_active and pinch_ratio > PINCH_UP:
            pinch_active = False

    # =============================
    # Grid geometry
    # =============================
    grid_size = 300
    cell = grid_size // 3
    start_x = (w - grid_size) // 2
    start_y = (h - grid_size) // 2

    # =============================
    # Place move
    # =============================
    if pointer and pinch_latched and not game_over:
        px, py = pointer

        if start_x <= px < start_x + grid_size and start_y <= py < start_y + grid_size:
            col = (px - start_x) // cell
            row = (py - start_y) // cell

            if board[row][col] is None:
                board[row][col] = current_player
                current_player = "O" if current_player == "X" else "X"
                game_over = check_winner()
                pinch_latched = False

    # =============================
    # Draw grid
    # =============================
    for i in range(1, 3):
        cv2.line(frame,
                 (start_x + i * cell, start_y),
                 (start_x + i * cell, start_y + grid_size),
                 (255, 255, 255), 2)
        cv2.line(frame,
                 (start_x, start_y + i * cell),
                 (start_x + grid_size, start_y + i * cell),
                 (255, 255, 255), 2)

    # =============================
    # Draw X and O
    # =============================
    for r in range(3):
        for c in range(3):
            cx = start_x + c * cell
            cy = start_y + r * cell

            if board[r][c] == "X":
                cv2.line(frame, (cx + 20, cy + 20), (cx + cell - 20, cy + cell - 20), (0, 0, 255), 3)
                cv2.line(frame, (cx + 20, cy + cell - 20), (cx + cell - 20, cy + 20), (0, 0, 255), 3)

            elif board[r][c] == "O":
                cv2.circle(frame, (cx + cell // 2, cy + cell // 2), cell // 2 - 20, (0, 255, 0), 3)

    # =============================
    # UI text
    # =============================
    if game_over:
        text = "DRAW" if game_over == "DRAW" else f"{game_over} WINS"
    else:
        text = f"Turn: {current_player}"

    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if pinch_ratio is not None:
        cv2.putText(frame, f"Pinch: {pinch_ratio:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Pinch Tic Tac Toe", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        board = [[None]*3 for _ in range(3)]
        current_player = "X"
        game_over = False

cap.release()
cv2.destroyAllWindows()
