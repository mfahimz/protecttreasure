# ============================================================
# 1. IMPORTS & SETUP
# ============================================================
# Import OS for file paths and environment variables
import os
# Import Sys for system-level exit commands
import sys
# Import Time to track game duration, cooldowns, and frame deltas
import time
# Import Random to spawn threats at unpredictable locations
import random
# Import Math for trigonometry (angles) and distance calculations
import math
# Import OpenCV (cv2) to capture and process the webcam feed
import cv2
# Import Pygame for the game window, rendering graphics, and playing audio
import pygame
# Import NumPy for efficient vector math (positions, velocities)
import numpy as np
# Import MediaPipe to access Google's AI models
import mediapipe as mp
# Import specific MediaPipe tasks for vision (hand/face tracking)
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

# ---- Windows Hardware Configuration ----
# Centers the Pygame window on the screen
os.environ["SDL_VIDEO_CENTERED"] = "1"

# ============================================================
# 2. CONFIGURATION
# ============================================================
# Set the game window dimensions (1400px wide, 950px high)
WIDTH, HEIGHT = 1400, 950
# Select the webcam index (1 = External USB, 0 = Built-in)
# Try changing to 0 if the camera doesn't open.
CAMERA_INDEX = 1        
# Set the total match time to 60 seconds
GAME_TIME = 60          

# Define paths to the AI model files (Must exist in a 'models' folder next to this script)
MODEL_PATH_HAND = "models/hand_landmarker.task"

# Define file paths for game assets (Images & Audio)
# [WINDOWS UPDATE] Using 'r' for raw strings to handle backslashes correctly
CHEST_IMAGE_PATH = r"C:\Users\pc\protecttreasure\CHEST_IMAGE_PATH.png"
THREAT_IMAGE_PATH = r"C:\Users\pc\protecttreasure\THREAT_IMAGE_PATH.png"
BACKGROUND_IMAGE_PATH = r"C:\Users\pc\protecttreasure\BACKGROUND_IMAGE_PATH.png"
BACKGROUND_MUSIC_PATH = r"C:\Users\pc\protecttreasure\BACKGROUND_MUSIC_PATH.mp3"
HIT_SOUND_PATH = r"C:\Users\pc\protecttreasure\HIT_SOUND_PATH.wav"
ROAR_SOUND_PATH = r"C:\Users\pc\protecttreasure\ROAR_SOUND_PATH.wav" 

# ============================================================
# 3. PHYSICS TUNING (THE "FEEL" OF THE GAME)
# ============================================================
# Size of the Chest sprite (pixels)
TREASURE_SIZE = 90
# Size of the Threat/Bullet sprite (pixels)
THREAT_SIZE = 60
# Radius for collision detection (Chest vs Threat)
BASE_BORDER_RADIUS = 70
# Radius for hand interaction (Hand vs Chest)
BASE_GRAB_RADIUS = 180  
# Number of hits the chest can take before game over
MAX_LIVES = 3  

# --- PROTECTOR PHYSICS (Defense) ---
# Smoothing factor for the chest movement. 
# 0.15 means it moves heavily/smoothly. 1.0 would be instant snapping.
MOVE_SMOOTHING = 0.15    

# "Iron Grip" Logic for the Protector:
# Value 0.0 = Fist Closed, 1.0 = Hand Open.
# Hand must be squeezed tighter than 0.22 to pick up the chest.
P_GRAB_THRESH = 0.22   
# Hand must be opened wider than 0.45 to drop the chest.
# This "gap" prevents accidental drops if the hand twitches.
P_DROP_THRESH = 0.45   
# If tracking is lost, hold the chest for 1.0s before dropping it.
GRACE_PERIOD_DURATION = 1.0 

# --- ATTACKER PHYSICS (Offense) ---
# Attacker must be very close (60px) to grab a bullet.
ATTACKER_GRAB_RANGE = 60  
# Speed of the bullet when thrown by the Attacker.
THROW_SPEED = 30.0        

# Agile Grip Settings for Attacker:
# Same grab threshold as protector.
A_GRAB_THRESH = 0.22
# Lower drop threshold (0.32) makes it easier to let go/throw quickly.
A_DROP_THRESH = 0.32      

# Fling Mechanics:
# If the hand velocity exceeds 60.0, trigger a throw automatically.
FLING_SPEED_TRIGGER = 60.0 
# When a bullet is grabbed, lock it for 0.2s so it isn't thrown instantly by mistake.
GRAB_LOCK_TIME = 0.2      

# Difficulty Scaling
# Chance (0.0 to 1.0) of a new threat spawning per frame.
SPAWN_RATE = 0.04       
# Initial speed of threats (0.0 means they spawn static).
START_SPEED = 0.0       

# Color Palette (R, G, B)
WHITE = (255, 255, 255)
RED = (220, 60, 60)
GREEN = (80, 220, 120)
YELLOW = (255, 200, 100)
BLUE = (100, 200, 255) 
CYAN = (80, 200, 255)
MAGENTA = (220, 80, 220)
ORANGE = (255, 100, 0)
GRAY = (100, 100, 100)

# ============================================================
# 4. INITIALIZE SYSTEMS
# ============================================================
# Configure MediaPipe Hand Tracking
options_hand = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH_HAND),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=4, # Track up to 4 hands to ensure we catch both players
    min_hand_detection_confidence=0.1, # Accept blurry hands (fast movement)
    min_tracking_confidence=0.1        
)
# Create the Hand Landmarker instance
landmarker_hand = vision.HandLandmarker.create_from_options(options_hand)

# Configure MediaPipe Face Tracking
options_face = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH_FACE),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1 # Track only 1 face (usually the Protector)
)
# Create the Face Landmarker instance
landmarker_face = vision.FaceLandmarker.create_from_options(options_face)

# Initialize Pygame
pygame.init()
# Initialize the Pygame audio mixer
pygame.mixer.init()
# Create the game window surface
screen = pygame.display.set_mode((WIDTH, HEIGHT))
# Set the window title
pygame.display.set_caption("Treasure Guard – WINDOWS BUILD")
# Create a clock to manage framerate
clock = pygame.time.Clock()
# Create font objects for text rendering
font = pygame.font.SysFont(None, 30)
big_font = pygame.font.Font(None, 100)

# Load Background Image safely
try:
    background_img = pygame.image.load(BACKGROUND_IMAGE_PATH).convert()
    background_img = pygame.transform.smoothscale(background_img, (WIDTH, HEIGHT))
except: background_img = None # If file missing, use None

# Load Game Sprites (Chest and Threat)
# Note: You can add error handling here too if images are missing
try:
    chest_img = pygame.transform.smoothscale(pygame.image.load(CHEST_IMAGE_PATH).convert_alpha(), (TREASURE_SIZE, TREASURE_SIZE))
    threat_img = pygame.transform.smoothscale(pygame.image.load(THREAT_IMAGE_PATH).convert_alpha(), (THREAT_SIZE, THREAT_SIZE))
except:
    # Fallback to creating colored squares if images fail to load
    chest_img = pygame.Surface((TREASURE_SIZE, TREASURE_SIZE))
    chest_img.fill(YELLOW)
    threat_img = pygame.Surface((THREAT_SIZE, THREAT_SIZE))
    threat_img.fill(RED)

# Load Background Music
try: 
    pygame.mixer.music.load(BACKGROUND_MUSIC_PATH)
    pygame.mixer.music.set_volume(0.6)
    pygame.mixer.music.play(-1) # Loop indefinitely
except: pass

# Load Sound Effects
hit_sound = None; roar_sound = None
try: 
    hit_sound = pygame.mixer.Sound(HIT_SOUND_PATH)
    hit_sound.set_volume(0.9)
except: pass
try: 
    roar_sound = pygame.mixer.Sound(ROAR_SOUND_PATH)
    roar_sound.set_volume(1.0)
except: pass

# Initialize Webcam Feed using OpenCV
# CAP_DSHOW is often more stable on Windows than default
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    # Fallback to default if DSHOW fails or Index 1 is wrong
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FPS, 60) # Request 60 FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Request HD Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # Request HD Height
start_time_ref = time.time() # Timestamp for synchronization

# ============================================================
# 5. STATE VARIABLES
# ============================================================
# --- Protector State (Left Player) ---
# Position of the chest
treasure_pos = np.array([WIDTH // 4, HEIGHT // 2], dtype=float)
# Smoothed position of the Protector's hand
p_hand_smooth = np.array([WIDTH // 4, HEIGHT // 2], dtype=float)
# Current velocity of Protector's hand
p_velocity = np.array([0.0, 0.0], dtype=float)
# Grip score (0.0 to 1.0)
p_grip = 1.0 
# Is the hand currently lost by camera?
p_tracking_lost = True
# Timer for the safety grace period
p_grace_timer = 0.0

# --- Attacker State (Right Player) ---
# Smoothed position of the Attacker's hand
a_hand_smooth = np.array([WIDTH * 0.75, HEIGHT // 2], dtype=float)
# Current velocity of Attacker's hand
a_velocity = np.array([0.0, 0.0], dtype=float)
# Grip score
a_grip = 1.0
# Tracking status
a_tracking_lost = True
# ID of the specific threat currently held
held_threat_id = None 
# Time when the grab started (for locking)
held_start_time = 0.0 

# --- Global Game State ---
threats = [] # List to store active bullets
state = "IDLE" # Chest state: IDLE or GRABBED
grab_frames = 0 # Counter to ensure deliberate grabs
game_start_time = time.time() # Timestamp when app started
grab_start_time = None # Timestamp when GAME started (first grab)
elapsed = 0 # Time passed in game
game_over = False
win = False
lives = MAX_LIVES
message_text = "" # Text to flash on screen
message_time = 0 # Timestamp of message

# --- Roar State ---
last_roar_time = -999 # Allow roar immediately
roar_active = False # Is the shockwave active?
roar_radius = 0 # Current size of shockwave

# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def get_grip_value(lm):
    """Calculates how 'open' the hand is. 0.0 = Closed, 1.0 = Open."""
    palm = lm[0] # Wrist landmark
    tips = [8, 12, 16, 20] # Fingertips: Index, Middle, Ring, Pinky
    # Calculate average distance from wrist to all fingertips
    avg_dist = sum(math.dist((palm.x, palm.y), (lm[i].x, lm[i].y)) for i in tips)
    return avg_dist / 4.0 # Normalize

def is_mouth_open(face_landmarks):
    """Detects if mouth is wide open for the Roar mechanic."""
    upper = face_landmarks[13] # Upper lip point
    lower = face_landmarks[14] # Lower lip point
    # Check vertical distance
    return math.dist((upper.x, upper.y), (lower.x, lower.y)) > MOUTH_OPEN_THRESHOLD

def spawn_threat():
    """Spawns a static threat in the Attacker's zone (Right Side)."""
    # Random X in right half, Random Y within vertical bounds
    x = random.randint(int(WIDTH * 0.6), WIDTH - 80)
    y = random.randint(80, HEIGHT - 80)
    pos = np.array([x, y], dtype=float)
    
    return {
        "id": time.time() + random.random(), # Unique ID
        "pos": pos, 
        "vel": np.array([0.0, 0.0]), # Starts with zero velocity
        "state": "IDLE", # Starts IDLE
        "angle": 0, # Current rotation
        "rotation_speed": random.uniform(-2, 2) # Spin speed
    }

def update_hand_physics(raw_pos, current_smooth, current_vel):
    """
    Adaptive Smoothing Logic:
    - Calculates velocity.
    - Smooths movement based on speed.
    - Handles prediction if hand is lost.
    """
    if raw_pos is not None:
        # Calculate instant velocity vector
        new_velocity = raw_pos - current_smooth
        dist = np.linalg.norm(new_velocity)
        
        # Teleport if movement is HUGE (e.g. 400px jump)
        # This handles when the camera re-acquires the hand
        if dist > 400: return raw_pos, np.array([0.0, 0.0])
        
        # Apply Smoothing:
        # 0.3 (New) + 0.7 (Old) creates a very smooth, heavy feel
        smooth = (raw_pos * 0.3) + (current_smooth * 0.7)
        # Smooth the velocity vector too
        vel = (new_velocity * 0.6) + (current_vel * 0.4)
        return smooth, vel
    else:
        # Ghost Hand Logic:
        # If hand is lost, continue moving it based on last known velocity
        smooth = current_smooth + current_vel
        # Apply Friction (0.92) so it slows down over time
        vel = current_vel * 0.92 
        return smooth, vel

# ============================================================
# 7. MAIN GAME LOOP
# ============================================================
try:
    while True:
        # Cap game at 60 FPS
        clock.tick(60)
        current_time = time.time()
        
        # Calculate Difficulty Intensity (0.0 to 1.0) based on time passed
        if grab_start_time:
            uptime = current_time - grab_start_time
            intensity = min(1.0, uptime / 30.0)
        else: intensity = 0.0

        # --- DRAW BACKGROUND ---
        # Flash Red on Hit
        if message_text == "HIT!" and (current_time - message_time) < 0.1: screen.fill(RED) 
        # Draw Background Image if available
        elif background_img: screen.blit(background_img, (0, 0))
        # Otherwise fill with dark blue
        else: screen.fill((20, 20, 30))

        # Draw the Divider Line between Left/Right zones
        zone_line_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        # Transparent gray line
        pygame.draw.line(zone_line_surf, (100, 100, 100, 40), (WIDTH//2, 0), (WIDTH//2, HEIGHT), 4)
        screen.blit(zone_line_surf, (0,0))

        # Show Zone Labels before start
        if not grab_start_time:
            screen.blit(font.render("PROTECTOR ZONE (LEFT)", True, CYAN), (100, 20))
            screen.blit(font.render("ATTACKER ZONE (RIGHT)", True, MAGENTA), (WIDTH - 350, 20))

        # Check for OS Quit Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT: cap.release(); pygame.quit(); sys.exit()

        # --- 8. COMPUTER VISION ---
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret: continue
        # Flip frame horizontally (Mirror view)
        frame = cv2.flip(frame, 1) 
        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # Calculate timestamp in ms
        timestamp = int((current_time - start_time_ref) * 1000)
        
        # Run Hand Tracking Model
        res_hand = landmarker_hand.detect_for_video(mp_image, timestamp)
        # Run Face Tracking Model
        res_face = landmarker_face.detect_for_video(mp_image, timestamp)

        # Variables to store raw hand positions
        raw_p_pos, raw_a_pos = None, None
        
        # Process Detected Hands
        if res_hand.hand_landmarks:
            for lm in res_hand.hand_landmarks:
                wrist = lm[0]
                # Scale coordinates to screen size
                px = np.array([int(wrist.x * WIDTH), int(wrist.y * HEIGHT)], dtype=float)
                # Calculate grip strength
                grip = get_grip_value(lm)
                
                # Zone Sorting:
                # If hand is in the Left 48% of screen -> It is Protector
                if px[0] < WIDTH * 0.48:
                    raw_p_pos = px
                    p_grip = grip
                    p_tracking_lost = False
                # If hand is in the Right 48% of screen -> It is Attacker
                elif px[0] > WIDTH * 0.52:
                    raw_a_pos = px
                    a_grip = grip
                    a_tracking_lost = False
        
        # Mark tracking as lost if no hand found in zone
        if raw_p_pos is None: p_tracking_lost = True
        if raw_a_pos is None: a_tracking_lost = True

        # Apply Physics Smoothing to both hands
        p_hand_smooth, p_velocity = update_hand_physics(raw_p_pos, p_hand_smooth, p_velocity)
        a_hand_smooth, a_velocity = update_hand_physics(raw_a_pos, a_hand_smooth, a_velocity)

        # If Protector is holding chest and tracking is lost, assume they are still holding it (Prevent dropping)
        if p_tracking_lost and state == "GRABBED": p_grip = 0.0
        
        # Check Face for Roar Mechanic
        if res_face.face_landmarks and not game_over and grab_start_time:
            # Check Cooldown
            if (current_time - last_roar_time) > ROAR_COOLDOWN:
                # Check if Mouth is Open
                if is_mouth_open(res_face.face_landmarks[0]):
                    # Trigger Roar
                    last_roar_time = current_time; roar_active = True; roar_radius = 50
                    if roar_sound: roar_sound.play()
                    message_text = "SONIC ROAR!"; message_time = current_time

        # --- 9. GAME LOGIC ---
        if not game_over:
            
            # Update Roar Shockwave
            if roar_active:
                roar_radius += 45 # Expand radius
                pygame.draw.circle(screen, BLUE, treasure_pos.astype(int), roar_radius, 20)
                # Check collision with threats
                for i in range(len(threats) - 1, -1, -1):
                    if math.dist(threats[i]["pos"], treasure_pos) < roar_radius:
                        threats.pop(i) # Destroy threat
                if roar_radius > max(WIDTH, HEIGHT): roar_active = False # End effect

            # Spawn Threats based on rate
            if grab_start_time:
                if len(threats) < 8 and random.random() < SPAWN_RATE: 
                    threats.append(spawn_threat())

            # --- PROTECTOR GRIP LOGIC ---
            is_holding = False
            # Check grip against thresholds
            if not p_tracking_lost or state == "GRABBED":
                if state == "IDLE":
                    # Must squeeze tight to grab
                    if p_grip < P_GRAB_THRESH: is_holding = True
                elif state == "GRABBED":
                    # Must open wide to drop
                    if p_grip < P_DROP_THRESH: is_holding = True

            # Protector State Machine
            if state == "IDLE":
                if is_holding:
                    # Check distance to chest
                    if np.linalg.norm(p_hand_smooth - treasure_pos) < BASE_GRAB_RADIUS:
                        grab_frames += 1
                        # Require 2 frames of holding to confirm pickup
                        if grab_frames > 2:
                            state = "GRABBED" 
                            if grab_start_time is None: grab_start_time = time.time()
                            grab_frames = 0; p_grace_timer = 0
                else: grab_frames = 0
            
            elif state == "GRABBED":
                if is_holding:
                    p_grace_timer = 0
                    # Move Chest towards Hand (with Smoothing)
                    treasure_pos += (p_hand_smooth - treasure_pos) * MOVE_SMOOTHING
                else:
                    # Grace Period Logic: If hand opens, wait before dropping
                    if p_grace_timer == 0: p_grace_timer = current_time
                    if (current_time - p_grace_timer) > GRACE_PERIOD_DURATION:
                        state = "IDLE"; p_grace_timer = 0

            # Update timer if game is active
            if state == "GRABBED": elapsed = time.time() - grab_start_time

            # --- ATTACKER GRAB LOGIC ---
            # Attempt to grab a threat
            if held_threat_id is None and not a_tracking_lost:
                if a_grip < A_GRAB_THRESH: # If gripping
                    closest_dist = 9999; target = None
                    # Find closest threat
                    for t in threats:
                        if t["state"] == "IDLE":
                            d = math.dist(a_hand_smooth, t["pos"])
                            # Check distance (Must be very close)
                            if d < ATTACKER_GRAB_RANGE and d < closest_dist:
                                closest_dist = d; target = t
                    # Grab it!
                    if target:
                        target["state"] = "HELD"
                        held_threat_id = target["id"]
                        held_start_time = current_time # Lock grip timer

            # --- UPDATE THREATS ---
            for i in range(len(threats) - 1, -1, -1):
                t = threats[i]
                
                # Logic if Held by Attacker
                if t["state"] == "HELD":
                    if t["id"] == held_threat_id:
                        hand_speed = np.linalg.norm(a_velocity)
                        # Check lock timer (prevents instant throw)
                        is_locked = (current_time - held_start_time) < GRAB_LOCK_TIME
                        
                        # THROW CONDITION:
                        # Not Locked AND (Hand Open OR Moved Fast)
                        if not is_locked and (a_grip > A_DROP_THRESH or hand_speed > FLING_SPEED_TRIGGER): 
                            t["state"] = "FIRED"
                            # Calculate throw vector
                            if hand_speed > 5.0:
                                aim_dir = a_velocity / hand_speed # Throw where hand is moving
                            else:
                                aim_dir = treasure_pos - t["pos"]
                                aim_dir /= np.linalg.norm(aim_dir) # Auto-aim at chest if static
                            t["vel"] = aim_dir * THROW_SPEED
                            held_threat_id = None
                        else:
                            t["pos"] = a_hand_smooth.copy() # Stick to hand
                    else:
                        t["state"] = "IDLE" # Should not happen, fallback

                # Logic if Fired (Projectile)
                elif t["state"] == "FIRED":
                    t["pos"] += t["vel"] 
                    # Rotate sprite to face velocity
                    t["angle"] = math.degrees(math.atan2(-t["vel"][1], t["vel"][0]))

                # Logic if Idle (Static floating)
                elif t["state"] == "IDLE":
                    t["angle"] += t["rotation_speed"] # Slow spin
                
                # Collision Check (Chest)
                if np.linalg.norm(t["pos"] - treasure_pos) < BASE_BORDER_RADIUS + 30:
                    lives -= 1; threats.pop(i) # Remove threat
                    message_text = "HIT!"; message_time = time.time()
                    if held_threat_id == t["id"]: held_threat_id = None
                    if hit_sound: hit_sound.play()
                    if lives < 0: game_over = True; win = False
                    continue

                # Clean up off-screen items to save memory
                px, py = t["pos"]
                if px < -200 or px > WIDTH + 200 or py < -200 or py > HEIGHT + 200:
                    threats.pop(i)
                    if held_threat_id == t["id"]: held_threat_id = None

            # Check Win Condition
            if grab_start_time and (time.time() - grab_start_time) >= GAME_TIME:
                game_over = True; win = True

        # --- 10. RENDER ---
        # Determine Chest Color
        color = GREEN if state == "GRABBED" else YELLOW
        if state == "GRABBED" and p_grace_timer > 0: color = ORANGE
        
        # Draw Chest Outline
        pygame.draw.circle(screen, color, treasure_pos.astype(int), BASE_BORDER_RADIUS, 4)
        # Draw Chest Image
        screen.blit(chest_img, (int(treasure_pos[0]-45), int(treasure_pos[1]-45)))

        # Draw Threats
        for t in threats:
            col = MAGENTA if t["state"] == "HELD" else WHITE
            # Rotate image
            rotated_threat = pygame.transform.rotate(threat_img, t.get("angle", 0))
            rect = rotated_threat.get_rect(center=(int(t["pos"][0]), int(t["pos"][1])))
            screen.blit(rotated_threat, rect.topleft)
            
            # Draw aiming line if holding
            if t["state"] == "HELD":
                pygame.draw.circle(screen, MAGENTA, t["pos"].astype(int), 35, 3)
                pygame.draw.line(screen, MAGENTA, t["pos"], t["pos"] + (a_velocity * 10), 3)

        # Draw Hand Cursors
        if not p_tracking_lost:
            pygame.draw.circle(screen, CYAN, p_hand_smooth.astype(int), 15, 3)
            # Show label only if game hasn't started
            if not grab_start_time:
                screen.blit(font.render("Protector", True, CYAN), (p_hand_smooth[0], p_hand_smooth[1]-30))
        
        if not a_tracking_lost:
            cursor_col = MAGENTA if held_threat_id else WHITE
            pygame.draw.circle(screen, cursor_col, a_hand_smooth.astype(int), 15, 3)
            if not grab_start_time:
                screen.blit(font.render("Attacker", True, cursor_col), (a_hand_smooth[0], a_hand_smooth[1]-30))

        # Render UI Text
        lives_txt = "♥ " * (lives + 1)
        screen.blit(font.render(f"Lives: {lives_txt}", True, RED), (20, 20))
        
        # Render Time
        if grab_start_time: rem = max(0, int(GAME_TIME - (current_time - grab_start_time)))
        else: rem = GAME_TIME
        screen.blit(font.render(f"Time: {rem}s", True, WHITE), (20, 60))

        # Render Roar Cooldown Bar
        cooldown_pct = min(1.0, (current_time - last_roar_time) / ROAR_COOLDOWN)
        bar_col = BLUE if cooldown_pct >= 1.0 else (50, 50, 100)
        pygame.draw.rect(screen, bar_col, (WIDTH//2 - 150, 20, int(300 * cooldown_pct), 20))
        pygame.draw.rect(screen, WHITE, (WIDTH//2 - 150, 20, 300, 20), 2)
        screen.blit(font.render("SONIC ROAR", True, WHITE), (WIDTH//2 - 60, 22))

        # Render Messages
        if message_text and time.time() - message_time < 1.0:
            txt = big_font.render(message_text, True, YELLOW)
            screen.blit(txt, (WIDTH//2 - txt.get_width()//2, HEIGHT//2 - 100))

        # Render Start Instruction
        if not grab_start_time:
            if int(current_time * 2) % 2 == 0: # Flash effect
                instr = big_font.render("GRAB CHEST TO START", True, GREEN)
                screen.blit(instr, (WIDTH//2 - instr.get_width()//2, HEIGHT//2 - 50))

        # Render Game Over Screen
        if game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT)); overlay.set_alpha(200); overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            msg = "PROTECTOR WINS!" if win else "ATTACKER WINS!"
            col = GREEN if win else RED
            txt = big_font.render(msg, True, col)
            screen.blit(txt, (WIDTH//2 - txt.get_width()//2, HEIGHT//2 - 50))
            score_final = font.render(f"Final Score: {int(elapsed)}s", True, WHITE)
            screen.blit(score_final, (WIDTH//2 - score_final.get_width()//2, HEIGHT//2 + 50))

        # Update Display
        pygame.display.flip()

# Handle Exit cleanly
except KeyboardInterrupt:
    cap.release()
    pygame.quit()