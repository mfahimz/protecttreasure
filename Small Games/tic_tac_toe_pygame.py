import sys
import time
import math
import random
import cv2
import numpy as np
import pygame
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "models/hand_landmarker.task"
CAMERA_INDEX = 0

WIDTH, HEIGHT = 900, 600
ORB_RADIUS = 18
TARGET_RADIUS = 45
HOLD_TIME = 2.0

INDEX_TIP = 8
PALM_BASE = 0

# =====================================================
# INIT PYGAME
# =====================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Aura Wake – Awakening")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

# =====================================================
# MEDIAPIPE HAND LANDMARKER
# =====================================================
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)

landmarker = vision.HandLandmarker.create_from_options(options)

# =====================================================
# CAMERA
# =====================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

t0 = time.time()

# =====================================================
# GAME OBJECTS
# =====================================================
orb_pos = [random.randint(150, WIDTH - 150), random.randint(150, HEIGHT - 150)]
orb_angle = 0

target_pos = (WIDTH - 120, HEIGHT // 2)

holding = False
hold_start = None
level_complete = False

# =====================================================
# HELPERS
# =====================================================
def gradient_background(t):
    for y in range(HEIGHT):
        c = int(80 + 60 * math.sin(t + y * 0.01))
        pygame.draw.line(screen, (30, c, 120 + c // 2), (0, y), (WIDTH, y))

def draw_glow(pos, radius, color, intensity=3):
    for i in range(intensity):
        pygame.draw.circle(
            screen,
            (*color, max(20 - i * 6, 0)),
            pos,
            radius + i * 6,
            2
        )

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# =====================================================
# MAIN LOOP
# =====================================================
while True:
    clock.tick(60)
    t = time.time()

    # -----------------------------
    # CAMERA FRAME
    # -----------------------------
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int((time.time() - t0) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    hand_pos = None
    palm_open = False

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        tip = lm[INDEX_TIP]
        palm = lm[PALM_BASE]

        hand_pos = (
            int(tip.x * WIDTH),
            int(tip.y * HEIGHT)
        )

        palm_dist = abs(tip.y - palm.y)
        palm_open = palm_dist > 0.08

    # -----------------------------
    # GAME LOGIC
    # -----------------------------
    if hand_pos and not level_complete:
        if distance(hand_pos, orb_pos) < ORB_RADIUS + 25 and palm_open:
            if not holding:
                holding = True
                hold_start = time.time()
        else:
            holding = False
            hold_start = None

        if holding and time.time() - hold_start >= HOLD_TIME:
            orb_pos[0] += (hand_pos[0] - orb_pos[0]) * 0.15
            orb_pos[1] += (hand_pos[1] - orb_pos[1]) * 0.15

        if distance(orb_pos, target_pos) < TARGET_RADIUS:
            level_complete = True

    orb_angle += 0.02
    orb_pos[1] += math.sin(orb_angle) * 0.3

    # =================================================
    # DRAWING
    # =================================================
    gradient_background(t)

    # TARGET PORTAL
    pygame.draw.circle(screen, (120, 200, 255), target_pos, TARGET_RADIUS, 3)
    draw_glow(target_pos, TARGET_RADIUS, (100, 200, 255), 4)

    # ORB
    orb_color = (80, 200, 255) if not holding else (255, 220, 120)
    pygame.draw.circle(screen, orb_color, orb_pos, ORB_RADIUS)
    draw_glow(orb_pos, ORB_RADIUS, orb_color, 4)

    # HAND AURA
    if hand_pos:
        aura_color = (120, 200, 255)
        if holding:
            aura_color = (255, 220, 120)
        draw_glow(hand_pos, 25, aura_color, 4)

    # UI TEXT
    if level_complete:
        text = font.render("Energy Awakened", True, (255, 255, 255))
    else:
        text = font.render("Catch • Hold • Release", True, (230, 230, 230))

    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, 20))

    pygame.display.update()

    # EXIT
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            cap.release()
            sys.exit()
