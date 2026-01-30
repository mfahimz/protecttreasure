# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import os                               # Used to set environment variables (macOS safety)
import sys                              # Provides system-level functions like exiting the program
import time                             # Used for timers and elapsed time calculation
import random                           # Used to randomly spawn incoming threats
import math                             # Used for distance and angle calculations
import cv2                              # OpenCV used to access the webcam
import pygame                           # Pygame used for window, graphics, sound, and input
import numpy as np                      # NumPy used for vector math and positions
import mediapipe as mp                  # MediaPipe used for hand tracking
from mediapipe.tasks.python import vision  # MediaPipe vision API
from mediapipe.tasks import python          # MediaPipe base options

# ============================================================
# ENVIRONMENT SAFETY (macOS SDL warning suppression)
# ============================================================

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"  # Prevent macOS SDL runtime issues

# ============================================================
# CONFIGURATION VALUES
# ============================================================

WIDTH, HEIGHT = 1400, 950               # Game window width and height
CAMERA_INDEX = 1                     # Default webcam index
GAME_TIME = 30                         # Seconds to survive to win

MODEL_PATH = "models/hand_landmarker.task"  # Path to MediaPipe hand model

# ============================================================
# FILE PATHS
# ============================================================

CHEST_IMAGE_PATH = "/Users/fahim/Desktop/Landmark_detection/CHEST_IMAGE_PATH.png"   # Treasure image
THREAT_IMAGE_PATH = "/Users/fahim/Desktop/Landmark_detection/THREAT_IMAGE_PATH.png"  # Threat image
BACKGROUND_IMAGE_PATH = "/Users/fahim/Desktop/Landmark_detection/BACKGROUND_IMAGE_PATH.png"  # Background image

BACKGROUND_MUSIC_PATH = "/Users/fahim/Desktop/Landmark_detection/BACKGROUND_MUSIC_PATH.mp3"  # Background music file
HIT_SOUND_PATH = "/Users/fahim/Desktop/Landmark_detection/HIT_SOUND_PATH.wav"                  # Sound when treasure is hit

# ============================================================
# GAME OBJECT SIZES AND PHYSICS
# ============================================================

TREASURE_SIZE = 90                      # Treasure sprite size
THREAT_SIZE = 58                        # Threat sprite size

BASE_BORDER_RADIUS = 70                 # Radius of protection circle
BORDER_THICKNESS = 4                    # Thickness of protection circle

BASE_GRAB_RADIUS = 120                  # Distance required to grab treasure
MOVE_SMOOTHING = 0.18                   # Smooth following speed

GRAB_CONFIRM_FRAMES = 12                # Frames required to confirm grab
RELEASE_CONFIRM_FRAMES = 20             # Frames required to confirm release

MAX_LIVES = 2                           # Number of hits allowed

# ============================================================
# MEDIAPIPE HAND TRACKING SETUP
# ============================================================

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),  # Load hand model
    running_mode=vision.RunningMode.VIDEO,                         # Video stream mode
    num_hands=1                                                     # Track only one hand
)

landmarker = vision.HandLandmarker.create_from_options(options)    # Create hand tracker

# ============================================================
# PYGAME INITIALIZATION
# ============================================================

pygame.init()                          # Initialize pygame
pygame.mixer.init()                    # Initialize sound mixer

screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Create window
pygame.display.set_caption("Treasure Guard")       # Set window title

clock = pygame.time.Clock()            # Clock to control FPS
font = pygame.font.SysFont(None, 32)   # Font for HUD text

# ============================================================
# LOAD AUDIO
# ============================================================

try:
    pygame.mixer.music.load(BACKGROUND_MUSIC_PATH)  # Load background music
    pygame.mixer.music.set_volume(0.4)               # Set volume
    pygame.mixer.music.play(-1)                      # Loop forever
except:
    print("Background music not found")

try:
    hit_sound = pygame.mixer.Sound(HIT_SOUND_PATH)   # Load hit sound
    hit_sound.set_volume(0.7)                         # Set volume
except:
    hit_sound = None
    print("Hit sound not found")

# ============================================================
# LOAD IMAGES
# ============================================================

background_img = pygame.image.load(BACKGROUND_IMAGE_PATH).convert()
background_img = pygame.transform.smoothscale(background_img, (WIDTH, HEIGHT))

chest_img = pygame.image.load(CHEST_IMAGE_PATH).convert_alpha()
chest_img = pygame.transform.smoothscale(chest_img, (TREASURE_SIZE, TREASURE_SIZE))

threat_img = pygame.image.load(THREAT_IMAGE_PATH).convert_alpha()
threat_img = pygame.transform.smoothscale(threat_img, (THREAT_SIZE, THREAT_SIZE))

# ============================================================
# CAMERA SETUP
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)   # Open webcam
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

start_time_ref = time.time()           # Reference start time

# ============================================================
# GAME STATE VARIABLES
# ============================================================

treasure_pos = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)  # Treasure starts center
threats = []                                                     # Active threats list

state = "IDLE"                                                   # IDLE or GRABBED
grab_frames = 0                                                  # Grab counter
grab_start_time = None                                           # Grab start time

elapsed = 0                                                      # Elapsed grab time
game_over = False                                                # Game over flag
win = False                                                      # Win flag

lives = MAX_LIVES                                                # Remaining lives

# ============================================================
# HAND GESTURE FUNCTIONS
# ============================================================

def hand_closed(lm):
    palm = lm[0]                                                 # Palm landmark
    tips = [8, 12, 16, 20]                                        # Fingertips
    closed = 0                                                    # Folded fingers count

    for i in tips:
        if math.dist((palm.x, palm.y), (lm[i].x, lm[i].y)) < 0.13:
            closed += 1

    return closed >= 3                                            # Fist detected

def spawn_threat():
    angle = random.uniform(0, 2 * math.pi)                       # Random angle
    dist = max(WIDTH, HEIGHT)                                    # Spawn far away

    pos = np.array([
        WIDTH // 2 + math.cos(angle) * dist,
        HEIGHT // 2 + math.sin(angle) * dist
    ])

    direction = treasure_pos - pos                                # Move toward treasure
    direction /= np.linalg.norm(direction)                        # Normalize

    speed = random.uniform(3.5, 5.5)                              # Random speed

    return {"pos": pos, "vel": direction * speed}

# ============================================================
# MAIN GAME LOOP
# ============================================================

while True:
    clock.tick(60)                                                # Run at 60 FPS

    screen.blit(background_img, (0, 0))                           # Draw background

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()

    ret, frame = cap.read()                                       # Read webcam frame
    if not ret:
        continue

    frame = cv2.flip(frame, 1)                                    # Mirror webcam
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                  # Convert to RGB

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int((time.time() - start_time_ref) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp)

    hand_pos = None
    fist = False

    if result.hand_landmarks:
        palm = result.hand_landmarks[0][0]
        hand_pos = np.array([int(palm.x * WIDTH), int(palm.y * HEIGHT)])
        fist = hand_closed(result.hand_landmarks[0])

    if not game_over and hand_pos is not None:
        dist = np.linalg.norm(hand_pos - treasure_pos)

        if state == "IDLE" and fist and dist < BASE_GRAB_RADIUS:
            grab_frames += 1
            if grab_frames >= GRAB_CONFIRM_FRAMES:
                state = "GRABBED"
                grab_start_time = time.time()
                grab_frames = 0

        elif state == "GRABBED":
            elapsed = time.time() - grab_start_time
            treasure_pos += (hand_pos - treasure_pos) * MOVE_SMOOTHING

            if random.random() < 0.08:
                threats.append(spawn_threat())

            for i in range(len(threats) - 1, -1, -1):
                threats[i]["pos"] += threats[i]["vel"]

                if np.linalg.norm(threats[i]["pos"] - treasure_pos) < BASE_BORDER_RADIUS:
                    lives -= 1
                    threats.pop(i)
                    if hit_sound:
                        hit_sound.play()

            if elapsed >= GAME_TIME:
                game_over = True
                win = True

    screen.blit(chest_img, (int(treasure_pos[0] - TREASURE_SIZE // 2),
                            int(treasure_pos[1] - TREASURE_SIZE // 2)))

    pygame.draw.circle(screen, (255, 200, 100), treasure_pos.astype(int),
                       BASE_BORDER_RADIUS, BORDER_THICKNESS)

    for t in threats:
        screen.blit(threat_img, (int(t["pos"][0] - THREAT_SIZE // 2),
                                 int(t["pos"][1] - THREAT_SIZE // 2)))

    screen.blit(font.render(f"Lives: {max(0, lives)}", True, (255, 255, 255)), (20, 20))
    screen.blit(font.render(f"Time: {max(0, int(GAME_TIME - elapsed))}", True, (255, 255, 255)), (20, 55))

    pygame.display.flip()                                         # Update screen
