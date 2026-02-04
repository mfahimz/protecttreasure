# ============================================================
# 1. IMPORTS & SETUP
# ============================================================
# Import the OS module to interact with the file system (paths) and environment variables.
import os
# Import Sys to handle system-specific parameters and exit functions.
import sys
# Import Time to track game duration, cooldowns, and frame deltas.
import time
# Import Random to generate unpredictable spawn locations and threat behaviors.
import random
# Import Math for trigonometric functions (angles) and distance calculations.
import math
# Import OpenCV (cv2) to capture video from the webcam and process images.
import cv2
# Import Pygame to create the game window, handle graphics, and play audio.
import pygame
# Import NumPy for high-performance vector mathematics (positions, velocities).
import numpy as np
# Import MediaPipe to access Google's machine learning solutions.
import mediapipe as mp
# Import specific MediaPipe modules for Hand and Face tracking tasks.
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

# ---- macOS Hardware Configuration ----
# This environment variable centers the Pygame window on the monitor.
os.environ["SDL_VIDEO_CENTERED"] = "1"
# [MAC SPECIFIC] Forces macOS to use the 'coreaudio' driver.
# Without this, Pygame sounds often lag or crash on Mac.
os.environ["SDL_AUDIODRIVER"] = "coreaudio"

# ============================================================
# 2. CONFIGURATION
# ============================================================
# Set the width of the game window in pixels.
WIDTH = 1400 
# Set the height of the game window in pixels.
HEIGHT = 950
# Select the camera input index.
# 0 is usually the built-in FaceTime Camera on Mac. 
# 1 is usually an external USB camera.
CAMERA_INDEX = 0        
# Set the total duration of a game match in seconds.
GAME_TIME = 60          

# Define the relative path to the Hand Landmark detection model file.
MODEL_PATH_HAND = "models/hand_landmarker.task"
# Define the relative path to the Face Landmark detection model file.
MODEL_PATH_FACE = "models/face_landmarker.task"

# Define file paths for visual and audio assets.
# [MAC UPDATE] Using standard Unix paths with forward slashes.
# Ensure your username is correct (replaced 'pc' with 'fahim' based on previous context).
CHEST_IMAGE_PATH = "/Users/fahim/Desktop/Landmark_detection/CHEST_IMAGE_PATH.png"
THREAT_IMAGE_PATH = "/Users/fahim/Desktop/Landmark_detection/THREAT_IMAGE_PATH.png"
BACKGROUND_IMAGE_PATH = "/Users/fahim/Desktop/Landmark_detection/BACKGROUND_IMAGE_PATH.png"
BACKGROUND_MUSIC_PATH = "/Users/fahim/Desktop/Landmark_detection/BACKGROUND_MUSIC_PATH.mp3"
HIT_SOUND_PATH = "/Users/fahim/Desktop/Landmark_detection/HIT_SOUND_PATH.wav"
ROAR_SOUND_PATH = "/Users/fahim/Desktop/Landmark_detection/ROAR_SOUND_PATH.wav"

# ============================================================
# 3. PHYSICS TUNING (THE "FEEL" OF THE GAME)
# ============================================================
# Set the pixel size for the Treasure Chest sprite.
TREASURE_SIZE = 90
# Set the pixel size for the Threat (Bullet) sprite.
THREAT_SIZE = 60
# Define the collision radius (in pixels) for the chest.
BASE_BORDER_RADIUS = 70
# Define the proximity radius (in pixels) required to grab the chest.
BASE_GRAB_RADIUS = 180  
# Set the maximum number of lives the Protector has.
MAX_LIVES = 3  

# --- PROTECTOR PHYSICS (Defense) ---
# Smoothing factor for chest movement. 
# 0.15 = Very smooth/heavy (only moves 15% of the distance to the hand per frame).
MOVE_SMOOTHING = 0.15    

# "Iron Grip" Logic for the Protector:
# Threshold (0.0=Closed, 1.0=Open) to initiate a grab. Must squeeze tight (0.22).
P_GRAB_THRESH = 0.22   
# Threshold to drop the chest. Must open hand wide (0.45).
# The gap between 0.22 and 0.45 creates "Hysteresis" preventing accidental drops.
P_DROP_THRESH = 0.45   
# Time in seconds to keep holding the chest if the camera loses track of the hand.
GRACE_PERIOD_DURATION = 1.0 

# --- ATTACKER PHYSICS (Offense) ---
# The Attacker must place their hand very close (60px) to a bullet to grab it.
ATTACKER_GRAB_RANGE = 60  
# The speed at which the bullet travels when thrown.
THROW_SPEED = 30.0        

# Agile Grip Settings for Attacker:
# Grab threshold is the same as protector.
A_GRAB_THRESH = 0.22
# Drop threshold is lower (0.32), making it easier/faster to release/throw items.
A_DROP_THRESH = 0.32      

# Fling Mechanics:
# If the hand moves faster than 60.0 pixels/frame, the code assumes a "Throw" action.
FLING_SPEED_TRIGGER = 60.0 
# When a bullet is grabbed, it is locked to the hand for 0.2s.
# This prevents the physics engine from throwing it instantly if the hand is moving fast.
GRAB_LOCK_TIME = 0.2

# --- FACE LOGIC (Sonic Roar) ---
# The vertical distance required between upper/lower lips to register "Open Mouth".
MOUTH_OPEN_THRESHOLD = 0.05 
# The cooldown time (seconds) required between sonic roars.
ROAR_COOLDOWN = 15.0        

# Difficulty Scaling
# The probability (0.0 to 1.0) of a new threat spawning in a single frame.
SPAWN_RATE = 0.04       
# Initial speed of threats (0.0 means they sit still until picked up).
START_SPEED = 0.0       

# Color Palette Definitions (RGB tuples)
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
# Configure the MediaPipe Hand tracking options.
options_hand = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH_HAND),
    running_mode=vision.RunningMode.VIDEO, # Optimize for video stream (uses previous frame data)
    num_hands=4, # Track up to 4 hands to ensure we don't lose players if extra hands appear
    min_hand_detection_confidence=0.1, # Low confidence allows tracking fast/blurry hands
    min_tracking_confidence=0.1        
)
# Create the Hand Landmarker instance with the specified options.
landmarker_hand = vision.HandLandmarker.create_from_options(options_hand)

# Configure the MediaPipe Face tracking options.
options_face = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH_FACE),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1 # We only need to track one face (usually the Protector's)
)
# Create the Face Landmarker instance.
landmarker_face = vision.FaceLandmarker.create_from_options(options_face)

# Initialize the Pygame library.
pygame.init()
# Initialize the Pygame audio mixer for sound effects.
pygame.mixer.init()
# Create the main game window with the defined dimensions.
screen = pygame.display.set_mode((WIDTH, HEIGHT))
# Set the caption/title of the window.
pygame.display.set_caption("Treasure Guard – MAC OS BUILD")
# Create a clock object to control the frame rate.
clock = pygame.time.Clock()
# Initialize font objects for rendering text on screen.
font = pygame.font.SysFont(None, 30)
big_font = pygame.font.Font(None, 100)

# Load Background Image safely
try:
    background_img = pygame.image.load(BACKGROUND_IMAGE_PATH).convert()
    # Scale background to fit the window exactly.
    background_img = pygame.transform.smoothscale(background_img, (WIDTH, HEIGHT))
except: 
    # If loading fails, set to None (code will use a solid color backup).
    background_img = None 

# Load and scale the Chest sprite.
chest_img = pygame.transform.smoothscale(pygame.image.load(CHEST_IMAGE_PATH).convert_alpha(), (TREASURE_SIZE, TREASURE_SIZE))
# Load and scale the Threat sprite.
threat_img = pygame.transform.smoothscale(pygame.image.load(THREAT_IMAGE_PATH).convert_alpha(), (THREAT_SIZE, THREAT_SIZE))

# Attempt to load and play background music.
try: 
    pygame.mixer.music.load(BACKGROUND_MUSIC_PATH)
    pygame.mixer.music.set_volume(0.6) # Set volume to 60%
    pygame.mixer.music.play(-1) # Play indefinitely (-1 loop)
except: pass # Ignore errors if file is missing

# Initialize sound effect variables.
hit_sound = None; roar_sound = None
# Attempt to load the 'Hit' sound effect.
try: 
    hit_sound = pygame.mixer.Sound(HIT_SOUND_PATH)
    hit_sound.set_volume(0.9)
except: pass
# Attempt to load the 'Roar' sound effect.
try: 
    roar_sound = pygame.mixer.Sound(ROAR_SOUND_PATH)
    roar_sound.set_volume(1.0)
except: pass

# Initialize the webcam using OpenCV.
# [MAC SPECIFIC] Use CAP_AVFOUNDATION to ensure permissions and stability on macOS.
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
# Set the webcam properties (FPS and Resolution).
cap.set(cv2.CAP_PROP_FPS, 60) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 
# Record the start time for tracking delta time.
start_time_ref = time.time() 

# ============================================================
# 5. STATE VARIABLES
# ============================================================
# --- Protector State (Left Player) ---
# Current position of the Treasure Chest [x, y].
treasure_pos = np.array([WIDTH // 4, HEIGHT // 2], dtype=float)
# Smoothed position of the Protector's hand cursor.
p_hand_smooth = np.array([WIDTH // 4, HEIGHT // 2], dtype=float)
# Current velocity vector of the Protector's hand.
p_velocity = np.array([0.0, 0.0], dtype=float)
# Grip strength (0.0 = Fist, 1.0 = Open Palm).
p_grip = 1.0 
# Boolean flag indicating if the Protector's hand is visible.
p_tracking_lost = True
# Timer to handle the grace period before dropping the chest.
p_grace_timer = 0.0

# --- Attacker State (Right Player) ---
# Smoothed position of the Attacker's hand cursor.
a_hand_smooth = np.array([WIDTH * 0.75, HEIGHT // 2], dtype=float)
# Current velocity vector of the Attacker's hand.
a_velocity = np.array([0.0, 0.0], dtype=float)
# Grip strength for Attacker.
a_grip = 1.0
# Boolean flag for Attacker visibility.
a_tracking_lost = True
# Stores the ID of the specific threat the Attacker is holding.
held_threat_id = None 
# Timestamp of when the Attacker started holding an object (for locking).
held_start_time = 0.0 

# --- Global Game State ---
# List to hold all active threat objects (dictionaries).
threats = [] 
# Current state of the chest ('IDLE' or 'GRABBED').
state = "IDLE" 
# Counter to validate grab stability (prevents flickering grabs).
grab_frames = 0 
# Time when the script started.
game_start_time = time.time() 
# Time when the actual gameplay started (first grab).
grab_start_time = None 
# Time elapsed during gameplay.
elapsed = 0 
# Game Over flag.
game_over = False
# Win flag.
win = False
# Current lives remaining.
lives = MAX_LIVES
# Text message to display on screen (e.g., "HIT!").
message_text = "" 
# Time when the message was set (for fading).
message_time = 0 

# --- Roar State ---
# Time when the last roar was used.
last_roar_time = -999 
# Is the roar shockwave currently active?
roar_active = False 
# Current radius of the shockwave circle.
roar_radius = 0 

# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def get_grip_value(lm):
    """
    Calculates the 'openness' of the hand based on landmarks.
    Returns a float: 0.0 (Closed Fist) to ~1.0 (Open Palm).
    """
    palm = lm[0] # The wrist landmark (Point 0)
    tips = [8, 12, 16, 20] # Indices for finger tips (Index, Middle, Ring, Pinky)
    # Calculate Euclidean distance from wrist to each fingertip.
    avg_dist = sum(math.dist((palm.x, palm.y), (lm[i].x, lm[i].y)) for i in tips)
    # Normalize by dividing by 4 (average).
    return avg_dist / 4.0 

def is_mouth_open(face_landmarks):
    """
    Checks if the mouth is open for the 'Roar' mechanic.
    Returns True if distance between lips > threshold.
    """
    upper = face_landmarks[13] # Upper lip landmark
    lower = face_landmarks[14] # Lower lip landmark
    # Check vertical distance between lips.
    return math.dist((upper.x, upper.y), (lower.x, lower.y)) > MOUTH_OPEN_THRESHOLD

def spawn_threat():
    """
    Creates a new threat object dictionary.
    Spawns in the 'Attacker Zone' (Right side of screen).
    """
    # Pick a random X coordinate on the right side (60% to edge).
    x = random.randint(int(WIDTH * 0.6), WIDTH - 80)
    # Pick a random Y coordinate within screen bounds.
    y = random.randint(80, HEIGHT - 80)
    pos = np.array([x, y], dtype=float)
    
    # Return dictionary with physics properties.
    return {
        "id": time.time() + random.random(), # Unique ID for grabbing logic
        "pos": pos, 
        "vel": np.array([0.0, 0.0]), # Velocity starts at 0 (Static)
        "state": "IDLE", # States: IDLE, HELD, FIRED
        "angle": 0, # Sprite rotation angle
        "rotation_speed": random.uniform(-2, 2) # Random spin
    }

def update_hand_physics(raw_pos, current_smooth, current_vel):
    """
    Applies adaptive smoothing to hand movement.
    - Fast movement -> Less smoothing (Responsive).
    - Slow movement -> More smoothing (Stable).
    - Tracking Lost -> Continues movement using inertia ('Ghost Hand').
    """
    if raw_pos is not None:
        # Calculate instant velocity (New Position - Old Position)
        new_velocity = raw_pos - current_smooth
        dist = np.linalg.norm(new_velocity)
        
        # Teleport Protection: If distance is huge (e.g. 400px), snap instantly.
        # This happens when the camera re-acquires a hand after losing it.
        if dist > 400: return raw_pos, np.array([0.0, 0.0])
        
        # Apply Exponential Moving Average (EMA) Smoothing.
        # 0.3 weight to new position, 0.7 weight to old. This creates a "heavy/smooth" feel.
        smooth = (raw_pos * 0.3) + (current_smooth * 0.7)
        # Smooth the velocity vector as well.
        vel = (new_velocity * 0.6) + (current_vel * 0.4)
        return smooth, vel
    else:
        # Ghost Hand Logic: If camera sees nothing, keep moving based on last velocity.
        smooth = current_smooth + current_vel
        # Apply Friction (0.92) so the ghost hand eventually stops.
        vel = current_vel * 0.92 
        return smooth, vel

# ============================================================
# 7. MAIN GAME LOOP
# ============================================================
try:
    # Infinite loop to keep the game running.
    while True:
        # Limit the loop to 60 Frames Per Second.
        clock.tick(60)
        # Get current timestamp.
        current_time = time.time()
        
        # Calculate Difficulty Intensity (Ramps from 0.0 to 1.0 over 30 seconds).
        if grab_start_time:
            uptime = current_time - grab_start_time
            intensity = min(1.0, uptime / 30.0)
        else: intensity = 0.0

        # --- DRAW BACKGROUND ---
        # If a hit message is active, flash the screen RED.
        if message_text == "HIT!" and (current_time - message_time) < 0.1: screen.fill(RED) 
        # Else, draw the background image.
        elif background_img: screen.blit(background_img, (0, 0))
        # Fallback to dark blue if no image.
        else: screen.fill((20, 20, 30))

        # --- DRAW ZONE DIVIDER ---
        # Create a transparent surface for the line.
        zone_line_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        # Draw a faint vertical line in the middle.
        pygame.draw.line(zone_line_surf, (100, 100, 100, 40), (WIDTH//2, 0), (WIDTH//2, HEIGHT), 4)
        # Blit the transparent surface onto the main screen.
        screen.blit(zone_line_surf, (0,0))

        # If game hasn't started, show zone labels.
        if not grab_start_time:
            screen.blit(font.render("PROTECTOR ZONE (LEFT)", True, CYAN), (100, 20))
            screen.blit(font.render("ATTACKER ZONE (RIGHT)", True, MAGENTA), (WIDTH - 350, 20))

        # --- EVENT HANDLING ---
        for event in pygame.event.get():
            # If user clicks X, release camera and exit.
            if event.type == pygame.QUIT: cap.release(); pygame.quit(); sys.exit()

        # --- 8. COMPUTER VISION PROCESSING ---
        # Read a frame from the webcam.
        ret, frame = cap.read()
        if not ret: continue # Skip if frame reading failed.
        # Flip the frame horizontally for a mirror effect.
        frame = cv2.flip(frame, 1) 
        # Convert BGR (OpenCV default) to RGB (MediaPipe requirement).
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Create a MediaPipe Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # Calculate the timestamp for MediaPipe.
        timestamp = int((current_time - start_time_ref) * 1000)
        
        # Detect Hands in the frame.
        res_hand = landmarker_hand.detect_for_video(mp_image, timestamp)
        # Detect Face in the frame.
        res_face = landmarker_face.detect_for_video(mp_image, timestamp)

        # Initialize raw position variables.
        raw_p_pos, raw_a_pos = None, None
        
        # If hands are detected...
        if res_hand.hand_landmarks:
            # Iterate through all detected hands.
            for lm in res_hand.hand_landmarks:
                wrist = lm[0] # Get wrist landmark.
                # Convert normalized coordinates (0-1) to pixel coordinates.
                px = np.array([int(wrist.x * WIDTH), int(wrist.y * HEIGHT)], dtype=float)
                # Calculate grip score for this hand.
                grip = get_grip_value(lm)
                
                # --- ZONE SORTING ---
                # Identify if hand belongs to Protector or Attacker based on screen side.
                
                # If hand is on the Left side (< 48% width) -> Assign to Protector.
                if px[0] < WIDTH * 0.48:
                    raw_p_pos = px
                    p_grip = grip
                    p_tracking_lost = False
                
                # If hand is on the Right side (> 52% width) -> Assign to Attacker.
                elif px[0] > WIDTH * 0.52:
                    raw_a_pos = px
                    a_grip = grip
                    a_tracking_lost = False
        
        # If no raw position assigned, mark tracking as lost.
        if raw_p_pos is None: p_tracking_lost = True
        if raw_a_pos is None: a_tracking_lost = True

        # Run the Physics Smoothing function on both hands.
        p_hand_smooth, p_velocity = update_hand_physics(raw_p_pos, p_hand_smooth, p_velocity)
        a_hand_smooth, a_velocity = update_hand_physics(raw_a_pos, a_hand_smooth, a_velocity)

        # Safety Check: If Protector holds chest but loses tracking, force grip to 'closed'
        # so they don't accidentally drop it due to a glitch.
        if p_tracking_lost and state == "GRABBED": p_grip = 0.0
        
        # --- FACE ROAR DETECTION ---
        if res_face.face_landmarks and not game_over and grab_start_time:
            # Only allow if cooldown has passed.
            if (current_time - last_roar_time) > ROAR_COOLDOWN:
                # Check if mouth is open.
                if is_mouth_open(res_face.face_landmarks[0]):
                    # Activate Roar!
                    last_roar_time = current_time
                    roar_active = True
                    roar_radius = 50
                    if roar_sound: roar_sound.play()
                    message_text = "SONIC ROAR!"
                    message_time = current_time

        # --- 9. GAME LOGIC ---
        if not game_over:
            
            # --- Roar Shockwave Logic ---
            if roar_active:
                roar_radius += 45 # Expand the circle quickly.
                # Draw the shockwave.
                pygame.draw.circle(screen, BLUE, treasure_pos.astype(int), roar_radius, 20)
                # Check collision with all threats.
                for i in range(len(threats) - 1, -1, -1):
                    # If threat is inside radius, destroy it.
                    if math.dist(threats[i]["pos"], treasure_pos) < roar_radius:
                        threats.pop(i) 
                # Stop expanding if it covers the whole screen.
                if roar_radius > max(WIDTH, HEIGHT): roar_active = False 

            # --- Spawning Logic ---
            if grab_start_time:
                # Spawn new threats if below limit (8) and probability check passes.
                if len(threats) < 8 and random.random() < SPAWN_RATE: 
                    threats.append(spawn_threat())

            # --- PROTECTOR PHYSICS (Gripping Chest) ---
            is_holding = False
            # Allow gripping if tracking is good OR if we are already in GRABBED state.
            if not p_tracking_lost or state == "GRABBED":
                if state == "IDLE":
                    # Require tight squeeze to pick up.
                    if p_grip < P_GRAB_THRESH: is_holding = True
                elif state == "GRABBED":
                    # Require wide open hand to drop.
                    if p_grip < P_DROP_THRESH: is_holding = True

            # Protector State Machine
            if state == "IDLE":
                if is_holding:
                    # Check if hand is close enough to chest.
                    if np.linalg.norm(p_hand_smooth - treasure_pos) < BASE_GRAB_RADIUS:
                        grab_frames += 1
                        # Wait 2 frames to confirm intent.
                        if grab_frames > 2:
                            state = "GRABBED" 
                            # Start game timer on first grab.
                            if grab_start_time is None: grab_start_time = time.time()
                            grab_frames = 0
                            p_grace_timer = 0
                else: grab_frames = 0
            
            elif state == "GRABBED":
                if is_holding:
                    p_grace_timer = 0
                    # Physics: Chest follows hand with smoothing.
                    treasure_pos += (p_hand_smooth - treasure_pos) * MOVE_SMOOTHING
                else:
                    # Grace Period: If grip is lost, wait before dropping.
                    if p_grace_timer == 0: p_grace_timer = current_time
                    if (current_time - p_grace_timer) > GRACE_PERIOD_DURATION:
                        state = "IDLE"; p_grace_timer = 0

            # Update Elapsed Time
            if state == "GRABBED": elapsed = time.time() - grab_start_time

            # --- ATTACKER PHYSICS (Grab & Throw) ---
            # Attempt to grab a threat if hand is empty.
            if held_threat_id is None and not a_tracking_lost:
                if a_grip < A_GRAB_THRESH: # Check if gripping
                    closest_dist = 9999; target = None
                    # Iterate through all idle threats.
                    for t in threats:
                        if t["state"] == "IDLE":
                            d = math.dist(a_hand_smooth, t["pos"])
                            # Check if hand is ON TOP of the threat.
                            if d < ATTACKER_GRAB_RANGE and d < closest_dist:
                                closest_dist = d; target = t
                    # If valid target found, grab it.
                    if target:
                        target["state"] = "HELD"
                        held_threat_id = target["id"]
                        held_start_time = current_time # Start lock timer

            # --- THREAT LOGIC LOOP ---
            for i in range(len(threats) - 1, -1, -1):
                t = threats[i]
                
                # Case 1: Threat is HELD by Attacker
                if t["state"] == "HELD":
                    if t["id"] == held_threat_id:
                        hand_speed = np.linalg.norm(a_velocity)
                        # Check if Grab Lock is active (prevent instant mis-throw).
                        is_locked = (current_time - held_start_time) < GRAB_LOCK_TIME
                        
                        # THROW LOGIC: 
                        # Throw if: Not Locked AND (Hand Open OR Hand Moving Super Fast).
                        if not is_locked and (a_grip > A_DROP_THRESH or hand_speed > FLING_SPEED_TRIGGER): 
                            t["state"] = "FIRED"
                            
                            # Calculate Throw Vector
                            if hand_speed > 5.0:
                                # Fling in direction of hand movement.
                                aim_dir = a_velocity / hand_speed 
                            else:
                                # If static release, auto-aim at chest.
                                aim_dir = treasure_pos - t["pos"]
                                aim_dir /= np.linalg.norm(aim_dir)
                            
                            t["vel"] = aim_dir * THROW_SPEED
                            held_threat_id = None
                        else:
                            # If still holding, glue threat position to hand.
                            t["pos"] = a_hand_smooth.copy() 
                    else:
                        t["state"] = "IDLE" # Fallback if ID mismatch.

                # Case 2: Threat is FIRED (Flying)
                elif t["state"] == "FIRED":
                    t["pos"] += t["vel"] # Apply velocity
                    # Rotate sprite to face direction of travel.
                    t["angle"] = math.degrees(math.atan2(-t["vel"][1], t["vel"][0]))

                # Case 3: Threat is IDLE (Floating)
                elif t["state"] == "IDLE":
                    t["angle"] += t["rotation_speed"] # Apply rotation
                
                # --- COLLISION DETECTION ---
                # Check distance between Threat and Chest.
                if np.linalg.norm(t["pos"] - treasure_pos) < BASE_BORDER_RADIUS + 30:
                    lives -= 1 # Deduct life
                    threats.pop(i) # Remove threat
                    message_text = "HIT!"; message_time = time.time()
                    
                    # Reset held ID if this was the held item.
                    if held_threat_id == t["id"]: held_threat_id = None
                    if hit_sound: hit_sound.play()
                    
                    # Check Game Over
                    if lives < 0: game_over = True; win = False
                    continue # Skip rest of loop for this item

                # --- CLEANUP ---
                # Remove threat if it flies off-screen.
                px, py = t["pos"]
                if px < -200 or px > WIDTH + 200 or py < -200 or py > HEIGHT + 200:
                    threats.pop(i)
                    if held_threat_id == t["id"]: held_threat_id = None

            # --- WIN CONDITION ---
            if grab_start_time and (time.time() - grab_start_time) >= GAME_TIME:
                game_over = True; win = True

        # --- 10. RENDERING (DRAWING TO SCREEN) ---
        # Determine chest color based on state (Green=Grabbed, Yellow=Idle, Orange=Grace).
        color = GREEN if state == "GRABBED" else YELLOW
        if state == "GRABBED" and p_grace_timer > 0: color = ORANGE
        
        # Draw Chest Outline.
        pygame.draw.circle(screen, color, treasure_pos.astype(int), BASE_BORDER_RADIUS, 4)
        # Draw Chest Sprite.
        screen.blit(chest_img, (int(treasure_pos[0]-45), int(treasure_pos[1]-45)))

        # Draw all Threats.
        for t in threats:
            # Color Highlight: Magenta if Held, White otherwise.
            col = MAGENTA if t["state"] == "HELD" else WHITE
            # Rotate sprite surface.
            rotated_threat = pygame.transform.rotate(threat_img, t.get("angle", 0))
            rect = rotated_threat.get_rect(center=(int(t["pos"][0]), int(t["pos"][1])))
            # Blit sprite.
            screen.blit(rotated_threat, rect.topleft)
            
            # If held, draw aiming line for visual feedback.
            if t["state"] == "HELD":
                pygame.draw.circle(screen, MAGENTA, t["pos"].astype(int), 35, 3)
                pygame.draw.line(screen, MAGENTA, t["pos"], t["pos"] + (a_velocity * 10), 3)

        # Draw Protector Cursor.
        if not p_tracking_lost:
            pygame.draw.circle(screen, CYAN, p_hand_smooth.astype(int), 15, 3)
            # Only show text label if game hasn't started yet.
            if not grab_start_time:
                screen.blit(font.render("Protector", True, CYAN), (p_hand_smooth[0], p_hand_smooth[1]-30))
        
        # Draw Attacker Cursor.
        if not a_tracking_lost:
            cursor_col = MAGENTA if held_threat_id else WHITE
            pygame.draw.circle(screen, cursor_col, a_hand_smooth.astype(int), 15, 3)
            if not grab_start_time:
                screen.blit(font.render("Attacker", True, cursor_col), (a_hand_smooth[0], a_hand_smooth[1]-30))

        # Render UI: Lives.
        lives_txt = "♥ " * (lives + 1)
        screen.blit(font.render(f"Lives: {lives_txt}", True, RED), (20, 20))
        
        # Render UI: Timer.
        if grab_start_time: rem = max(0, int(GAME_TIME - (current_time - grab_start_time)))
        else: rem = GAME_TIME
        screen.blit(font.render(f"Time: {rem}s", True, WHITE), (20, 60))

        # Render UI: Roar Cooldown Bar.
        cooldown_pct = min(1.0, (current_time - last_roar_time) / ROAR_COOLDOWN)
        bar_col = BLUE if cooldown_pct >= 1.0 else (50, 50, 100)
        pygame.draw.rect(screen, bar_col, (WIDTH//2 - 150, 20, int(300 * cooldown_pct), 20))
        pygame.draw.rect(screen, WHITE, (WIDTH//2 - 150, 20, 300, 20), 2)
        screen.blit(font.render("SONIC ROAR", True, WHITE), (WIDTH//2 - 60, 22))

        # Render Flashing Messages.
        if message_text and time.time() - message_time < 1.0:
            txt = big_font.render(message_text, True, YELLOW)
            screen.blit(txt, (WIDTH//2 - txt.get_width()//2, HEIGHT//2 - 100))

        # Render Start Instruction text.
        if not grab_start_time:
            if int(current_time * 2) % 2 == 0: # Flash logic (On/Off every 0.5s)
                instr = big_font.render("GRAB CHEST TO START", True, GREEN)
                screen.blit(instr, (WIDTH//2 - instr.get_width()//2, HEIGHT//2 - 50))

        # Render Game Over Overlay.
        if game_over:
            # Create semi-transparent black overlay.
            overlay = pygame.Surface((WIDTH, HEIGHT)); overlay.set_alpha(200); overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            
            # Determine Winner text.
            msg = "PROTECTOR WINS!" if win else "ATTACKER WINS!"
            col = GREEN if win else RED
            txt = big_font.render(msg, True, col)
            screen.blit(txt, (WIDTH//2 - txt.get_width()//2, HEIGHT//2 - 50))
            
            # Show Survival Time.
            score_final = font.render(f"Final Score: {int(elapsed)}s", True, WHITE)
            screen.blit(score_final, (WIDTH//2 - score_final.get_width()//2, HEIGHT//2 + 50))

        # Flip the display buffer to show the new frame.
        pygame.display.flip()

# Handle Keyboard Interrupt (Ctrl+C) to exit cleanly.
except KeyboardInterrupt:
    cap.release()
    pygame.quit()