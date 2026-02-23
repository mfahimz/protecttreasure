# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import os                               # Used to set environment variables
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
# CONFIGURATION VALUES
# ============================================================

WIDTH, HEIGHT = 1400, 950               # Game window width and height
GAME_TIME = 30                          # Seconds to survive to win

MODEL_PATH = "models/hand_landmarker.task"  # Path to MediaPipe hand model

# ============================================================
# FILE PATHS (Relative - works on all OS)
# ============================================================

CHEST_IMAGE_PATH = "assets/chest.png"           # Treasure image
THREAT_IMAGE_PATH = "assets/threat.png"         # Threat image
BACKGROUND_IMAGE_PATH = "assets/background.png" # Background image

BACKGROUND_MUSIC_PATH = "assets/background_music.mp3"  # Background music file
HIT_SOUND_PATH = "assets/hit_sound.wav"                 # Sound when treasure is hit

# ============================================================
# CAMERA AUTO-DETECTION (Cross-platform)
# ============================================================

def auto_select_display():
    """Automatically select display - prefer extended screen if available"""
    num_displays = pygame.display.get_num_displays()
    if num_displays > 1:
        return 1  # Use extended screen
    return 0  # Use primary screen

def auto_select_camera():
    """Automatically detect best available camera on any OS"""
    available_cameras = []
    
    for i in range(4):
        # Try DirectShow (Windows) first, but fallback works on all OS
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(i)  # Default works on Mac/Linux
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                resolution = w * h
                available_cameras.append((i, w, h, resolution))
            cap.release()
    
    if not available_cameras:
        return 0  # Default to camera 0
    
    # Sort by resolution and pick best
    available_cameras.sort(key=lambda x: x[3], reverse=True)
    selected = available_cameras[0][0]
    
    # Prefer external camera if resolution is similar
    if len(available_cameras) > 1:
        best_res = available_cameras[0][3]
        for cam in available_cameras[1:]:
            if cam[0] > 0 and cam[3] >= (best_res * 0.8):
                selected = cam[0]
                break
    
    return selected

CAMERA_INDEX = auto_select_camera()

# ============================================================
# FULLSCREEN CONFIGURATION
# ============================================================

USE_FULLSCREEN = True
DISPLAY_INDEX = auto_select_display()

# Get screen resolution
try:
    if DISPLAY_INDEX < len(pygame.display.get_desktop_sizes()):
        WIDTH, HEIGHT = pygame.display.get_desktop_sizes()[DISPLAY_INDEX]
    else:
        WIDTH, HEIGHT = 1920, 1080
except:
    WIDTH, HEIGHT = 1920, 1080

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

# Create fullscreen window or regular window
if USE_FULLSCREEN:
    try:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=DISPLAY_INDEX)
        WIDTH, HEIGHT = screen.get_size()  # Update dimensions to actual screen size
    except Exception as e:
        screen = pygame.display.set_mode((1920, 1080))
        WIDTH, HEIGHT = 1920, 1080
        USE_FULLSCREEN = False
else:
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

pygame.display.set_caption("Treasure Guard")       # Set window title

clock = pygame.time.Clock()            # Clock to control FPS
font = pygame.font.SysFont(None, 32)   # Font for HUD text

# ============================================================
# LOAD AUDIO (with fallback)
# ============================================================

try:
    pygame.mixer.music.load(BACKGROUND_MUSIC_PATH)  # Load background music
    pygame.mixer.music.set_volume(0.4)               # Set volume
    pygame.mixer.music.play(-1)                      # Loop forever
except:
    print("Background music not found - continuing without music")

try:
    hit_sound = pygame.mixer.Sound(HIT_SOUND_PATH)   # Load hit sound
    hit_sound.set_volume(0.7)                         # Set volume
except:
    hit_sound = None
    print("Hit sound not found - continuing without sound")

# ============================================================
# LOAD IMAGES (with fallback colors and dynamic scaling)
# ============================================================

background_img = None
try:
    background_img = pygame.image.load(BACKGROUND_IMAGE_PATH).convert()
except:
    print("Background image not found - using solid color")

def get_scaled_background():
    """Get background scaled to current screen size"""
    global background_img
    if background_img is not None:
        try:
            return pygame.transform.smoothscale(background_img, (WIDTH, HEIGHT))
        except:
            return None
    return None

# Initial background scaling
scaled_background = get_scaled_background()

# Fallback if no background
if scaled_background is None:
    scaled_background = pygame.Surface((WIDTH, HEIGHT))
    scaled_background.fill((15, 15, 25))  # Dark blue background

try:
    chest_img = pygame.image.load(CHEST_IMAGE_PATH).convert_alpha()
    chest_img = pygame.transform.smoothscale(chest_img, (TREASURE_SIZE, TREASURE_SIZE))
except:
    print("Chest image not found - using colored square")
    chest_img = pygame.Surface((TREASURE_SIZE, TREASURE_SIZE))
    chest_img.fill((255, 215, 0))  # Gold color

try:
    threat_img = pygame.image.load(THREAT_IMAGE_PATH).convert_alpha()
    threat_img = pygame.transform.smoothscale(threat_img, (THREAT_SIZE, THREAT_SIZE))
except:
    print("Threat image not found - using colored square")
    threat_img = pygame.Surface((THREAT_SIZE, THREAT_SIZE))
    threat_img.fill((255, 82, 82))  # Red color

# ============================================================
# CAMERA SETUP (Cross-platform)
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # Try DirectShow (Windows)
if not cap.isOpened():
    cap = cv2.VideoCapture(CAMERA_INDEX)  # Fallback to default (Mac/Linux)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)  # Last resort - camera 0
    CAMERA_INDEX = 0

if not cap.isOpened():
    print("ERROR: No camera detected!")
    pygame.quit()
    sys.exit(1)

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
    """Detect if hand is making a fist"""
    palm = lm[0]                                                 # Palm landmark
    tips = [8, 12, 16, 20]                                        # Fingertips
    closed = 0                                                    # Folded fingers count

    for i in tips:
        if math.dist((palm.x, palm.y), (lm[i].x, lm[i].y)) < 0.13:
            closed += 1

    return closed >= 3                                            # Fist detected

def spawn_threat():
    """Spawn a threat from random direction moving toward treasure"""
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

def toggle_fullscreen():
    """Toggle between fullscreen and windowed mode"""
    global WIDTH, HEIGHT, USE_FULLSCREEN, screen, scaled_background
    USE_FULLSCREEN = not USE_FULLSCREEN
    if USE_FULLSCREEN:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        WIDTH, HEIGHT = screen.get_size()
    else:
        screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
        WIDTH, HEIGHT = 1920, 1080
    scaled_background = get_scaled_background()
    if scaled_background is None:
        scaled_background = pygame.Surface((WIDTH, HEIGHT))
        scaled_background.fill((15, 15, 25))

# ============================================================
# MAIN GAME LOOP
# ============================================================

print("Game starting...")
print(f"Using camera index: {CAMERA_INDEX}")

running = True

while running:
    clock.tick(60)                                                # Run at 60 FPS

    screen.blit(scaled_background, (0, 0))                        # Draw background

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_f:
                toggle_fullscreen()

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
                    
                    if lives <= 0:
                        game_over = True
                        win = False

            if elapsed >= GAME_TIME:
                game_over = True
                win = True

    # Draw treasure
    screen.blit(chest_img, (int(treasure_pos[0] - TREASURE_SIZE // 2),
                            int(treasure_pos[1] - TREASURE_SIZE // 2)))

    # Draw protection circle
    pygame.draw.circle(screen, (255, 200, 100), treasure_pos.astype(int),
                       BASE_BORDER_RADIUS, BORDER_THICKNESS)

    # Draw threats
    for t in threats:
        screen.blit(threat_img, (int(t["pos"][0] - THREAT_SIZE // 2),
                                 int(t["pos"][1] - THREAT_SIZE // 2)))

    # Draw HUD
    lives_text = font.render(f"Lives: {max(0, lives)}", True, (255, 255, 255))
    time_text = font.render(f"Time: {max(0, int(GAME_TIME - elapsed))}", True, (255, 255, 255))
    
    screen.blit(lives_text, (20, 20))
    screen.blit(time_text, (20, 55))

    # Game over screen
    if game_over:
        big_font = pygame.font.SysFont(None, 72)
        if win:
            end_text = big_font.render("VICTORY!", True, (0, 255, 0))
        else:
            end_text = big_font.render("GAME OVER", True, (255, 0, 0))
        
        text_rect = end_text.get_rect(center=(WIDTH//2, HEIGHT//2))
        screen.blit(end_text, text_rect)
        
        restart_text = font.render("Press ESC to exit  •  Press F for Fullscreen", True, (255, 255, 255))
        restart_rect = restart_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 60))
        screen.blit(restart_text, restart_rect)

    pygame.display.flip()                                         # Update screen

# ============================================================
# CLEANUP
# ============================================================

cap.release()
pygame.quit()
print("Game closed")