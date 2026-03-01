import os, sys, time, random, math, cv2, pygame
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

print("TREASURE GUARD - Ultimate Challenge")

pygame.init()

# ============================================================
# DISPLAY & CAMERA AUTO-DETECTION
# ============================================================

def auto_select_display():
    num_displays = pygame.display.get_num_displays()
    if num_displays > 1:
        return 1
    return 0

def auto_select_camera():
    available_cameras = []
    
    for i in range(4):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                resolution = w * h
                available_cameras.append((i, w, h, resolution))
            cap.release()
    
    if not available_cameras:
        return 0
    
    available_cameras.sort(key=lambda x: x[3], reverse=True)
    selected = available_cameras[0][0]
    
    if len(available_cameras) > 1:
        best_res = available_cameras[0][3]
        for cam in available_cameras[1:]:
            if cam[0] > 0 and cam[3] >= (best_res * 0.8):
                selected = cam[0]
                break
    
    return selected

USE_FULLSCREEN = True
DISPLAY_INDEX = auto_select_display()

try:
    if DISPLAY_INDEX < len(pygame.display.get_desktop_sizes()):
        WIDTH, HEIGHT = pygame.display.get_desktop_sizes()[DISPLAY_INDEX]
    else:
        WIDTH, HEIGHT = 1920, 1080
except:
    WIDTH, HEIGHT = 1920, 1080

if USE_FULLSCREEN:
    try:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=DISPLAY_INDEX)
        WIDTH, HEIGHT = screen.get_size()
    except Exception as e:
        screen = pygame.display.set_mode((1920, 1080))
        WIDTH, HEIGHT = 1920, 1080
        USE_FULLSCREEN = False
else:
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

pygame.display.set_caption("Treasure Guard - Ultimate Challenge")

CAMERA_INDEX = auto_select_camera()
SHOW_CAMERA_DEBUG = True
DEBUG_WINDOW_MAX_WIDTH = 180
DEBUG_WINDOW_MAX_HEIGHT = 101

MODEL_PATH_HAND = "models/hand_landmarker.task"

CAMERA_ZOOM_ENABLED = True
CAMERA_ZOOM_AMOUNT = 0.10  # 10% ROI margin (reduced from 15% for tighter focus)

# ============================================================
# THREAT TYPE DEFINITIONS
# ============================================================

THREAT_TYPES = {
    "threat": {
        "name": "threat",
        "damage": 0.5,
        "size": 60,  # Threat
        "image_path": "assets/threat.png",
        "fallback_color": (255, 100, 100),
        "points": 10,
        "collision_radius": 20,
        "description": "Basic threat"
    },
    "rocket": {
        "name": "Rocket",
        "damage": 1.0,
        "size": 110,  # Rocket
        "image_path": "assets/rocket.png",
        "fallback_color": (255, 150, 50),
        "points": 20,
        "collision_radius": 40,
        "description": "Faster, more damage"
    },
    "grenade": {
        "name": "Grenade",
        "damage": 3.0,  # Instant death
        "size": 120,  # Nuke  # Grenade
        "image_path": "assets/grenade.png",
        "fallback_color": (255, 200, 0),
        "points": 30,
        "collision_radius": 40,
        "description": "One hit kill"
    },
    "nuke": {
        "name": "Nuke",
        "damage": 3.0,  # Instant death
        "size": 120,  # Nuke
        "image_path": "assets/nuke.png",
        "fallback_color": (200, 0, 255),
        "points": 50,
        "collision_radius": 40,
        "description": "Devastating!"
    }
}

# ============================================================
# LEVEL SYSTEM - 60 SECOND TOTAL GAME
# ============================================================

TOTAL_GAME_TIME = 60.0

LEVELS = {
    1: {
        "name": "WAVE 1",
        "duration": 15,
        "max_threats": 3,  # Reduced from 5 (30% reduction)
        "spawn_rate": 0.028,  # Reduced from 0.04 (30% reduction)
        "threat_speed": (2.8, 3.5),  # Reduced from (4.0, 5.0) (30% reduction)
        "allowed_threats": ["threat"],
        "threat_intervals": {
            "threat": 0.5,
        },
        "targeting_accuracy": 0.6,
        "description": "Warm up!",
        "color": (94, 234, 212)
    },
    2: {
        "name": "WAVE 2",
        "duration": 15,
        "max_threats": 5,  # Reduced from 7 (30% reduction)
        "spawn_rate": 0.042,  # Reduced from 0.06 (30% reduction)
        "threat_speed": (3.5, 4.55),  # Reduced from (5.0, 6.5) (30% reduction)
        "allowed_threats": ["threat", "rocket"],
        "threat_intervals": {
            "threat": 0.8,
            "rocket": 8.0
        },
        "targeting_accuracy": 0.85,
        "description": "Rockets incoming!",
        "color": (87, 242, 135)
    },
    3: {
        "name": "WAVE 3",
        "duration": 15,
        "max_threats": 7,  # Reduced from 10 (30% reduction)
        "spawn_rate": 0.056,  # Reduced from 0.08 (30% reduction)
        "threat_speed": (4.2, 5.6),  # Reduced from (6.0, 8.0) (30% reduction)
        "allowed_threats": ["threat", "rocket", "grenade"],
        "threat_intervals": {
            "threat": 0.7,
            "rocket": 5.0,
            "grenade": 10.0
        },
        "targeting_accuracy": 0.85,
        "description": "Chaos mode!",
        "color": (255, 154, 88)
    },
    4: {
        "name": "FINAL WAVE",
        "duration": 15,
        "max_threats": 10,  # Reduced from 15 (30% reduction)
        "spawn_rate": 0.084,  # Reduced from 0.12 (30% reduction)
        "threat_speed": (4.9, 7.0),  # Reduced from (7.0, 10.0) (30% reduction)
        "allowed_threats": ["threat", "rocket", "grenade", "nuke"],
        "threat_intervals": {
            "threat": 0.6,
            "rocket": 3.0,
            "grenade": 6.0,
            "nuke": 12.0
        },
        "targeting_accuracy": 0.99,
        "description": "HELL UNLEASHED!",
        "color": (239, 68, 68)
    }
}

# ============================================================
# FILE PATHS
# ============================================================

CHEST_IMAGE_PATH = "assets/chest.png"
BACKGROUND_IMAGE_PATH = "assets/background.jpeg"
BACKGROUND_MUSIC_PATH = "assets/background_music.mp3"
HIT_SOUND_PATH = "assets/hit_sound.wav"
LEVEL_UP_SOUND_PATH = "assets/level_up.wav"

# ============================================================
# GAME CONFIGURATION
# ============================================================

TREASURE_SIZE = 90
BASE_BORDER_RADIUS, BASE_GRAB_RADIUS = 70, 180
MAX_LIVES = 4.0
HIT_FLASH_DURATION = 0.4
SHAKE_INTENSITY = 12

P_GRAB_THRESH, P_DROP_THRESH = 1.3, 1.9
MOVE_SMOOTHING = 0.20

# ============================================================
# COLORS
# ============================================================

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_BG = (15, 15, 25)
RED = (255, 82, 82)
GREEN = (87, 242, 135)
YELLOW = (255, 234, 167)
CYAN = (94, 234, 212)
MAGENTA = (251, 113, 133)
GOLD = (255, 215, 0)
ORANGE = (255, 154, 88)
UI_BG = (30, 30, 45)
UI_BORDER = (100, 100, 130)
UI_ACCENT = (94, 234, 212)
SUCCESS = (34, 197, 94)
DANGER = (239, 68, 68)
WARNING = (245, 158, 11)
PURPLE = (200, 0, 255)

# ============================================================
# MEDIAPIPE SETUP
# ============================================================

options_hand = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH_HAND),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
landmarker_hand = vision.HandLandmarker.create_from_options(options_hand)

pygame.mixer.init()
clock = pygame.time.Clock()

# ============================================================
# FONTS - Gaming Style
# ============================================================

# Try to use gaming-style fonts, fallback to bold system fonts
def get_gaming_font(size):
    """Try gaming fonts, fallback to bold"""
    # Try gaming-style font names
    gaming_fonts = ['Arial Black', 'Impact', 'Bahnschrift', 'Tahoma Bold', 'Verdana Bold']
    
    for font_name in gaming_fonts:
        try:
            return pygame.font.SysFont(font_name, size, bold=True)
        except:
            pass
    
    # Final fallback
    try:
        return pygame.font.Font(None, size)
    except:
        return pygame.font.SysFont('Arial', size, bold=True)

title_font = get_gaming_font(100)  # Reduced from 160
big_font = get_gaming_font(70)     # Reduced from 100
medium_font = get_gaming_font(45)  # Reduced from 60
font = get_gaming_font(32)         # Reduced from 40
small_font = get_gaming_font(22)   # Reduced from 26

# ============================================================
# PARTICLE SYSTEM
# ============================================================

class ParticleSystem:
    def __init__(self):
        self.particles = []
        self.particle_surfaces = {}
        for size in range(4, 13):
            surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (255, 255, 255), (size, size), size)
            self.particle_surfaces[size] = surf
    
    def add_particles(self, x, y, color, count=20):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(3, 10)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - 2
            size = random.randint(4, 12)
            self.particles.append([x, y, vx, vy, size, 40, color])
    
    def update_and_draw(self, screen):
        for p in self.particles[:]:
            p[0] += p[2]
            p[1] += p[3]
            p[3] += 0.4
            p[5] -= 1
            
            if p[5] <= 0:
                self.particles.remove(p)
            else:
                size = int(p[4])
                if size in self.particle_surfaces:
                    alpha = int(255 * (p[5] / 40))
                    temp_surf = self.particle_surfaces[size].copy()
                    temp_surf.fill((*p[6], alpha), special_flags=pygame.BLEND_RGBA_MULT)
                    screen.blit(temp_surf, (int(p[0] - size), int(p[1] - size)))

# ============================================================
# LOAD ASSETS
# ============================================================

def load_sound(path):
    try:
        return pygame.mixer.Sound(path)
    except:
        return None

hit_sound = load_sound(HIT_SOUND_PATH)
level_up_sound = load_sound(LEVEL_UP_SOUND_PATH)

hit_sounds = []
for i in range(1, 4):
    path = f"assets/sounds/hit{i}.wav"
    if os.path.exists(path):
        try:
            sound = pygame.mixer.Sound(path)
            sound.set_volume(0.5)
            hit_sounds.append(sound)
        except:
            pass

if not hit_sounds and hit_sound:
    hit_sounds = [hit_sound]

dodge_sounds = []
for i in range(1, 4):
    path = f"assets/sounds/dodge{i}.wav"
    if os.path.exists(path):
        try:
            sound = pygame.mixer.Sound(path)
            sound.set_volume(0.4)
            dodge_sounds.append(sound)
        except:
            pass

def load_scale(path, size, fallback_color):
    # Handle both integer and tuple sizes
    if isinstance(size, tuple):
        size_tuple = size
        size_int = size[0]
    else:
        size_tuple = (size, size)
        size_int = size
    
    try:
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.smoothscale(img, size_tuple)
    except:
        # Create fallback surface with proper dimensions
        surf = pygame.Surface(size_tuple, pygame.SRCALPHA)
        pygame.draw.circle(surf, fallback_color, (size_int//2, size_int//2), size_int//2)
        return surf

chest_img = load_scale(CHEST_IMAGE_PATH, TREASURE_SIZE, YELLOW)

# Load all threat images
threat_images = {}
for threat_type, config in THREAT_TYPES.items():
    threat_images[threat_type] = load_scale(
        config["image_path"],
        (config["size"], config["size"]),
        config["fallback_color"]
    )

background_img = None
try:
    bg_temp = pygame.image.load(BACKGROUND_IMAGE_PATH).convert()
    background_img = bg_temp
except:
    pass

try:
    pygame.mixer.music.load(BACKGROUND_MUSIC_PATH)
    pygame.mixer.music.set_volume(0.3)
    pygame.mixer.music.play(-1)
except:
    pass

# ============================================================
# CAMERA SETUP
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    CAMERA_INDEX = 0

if not cap.isOpened():
    print("ERROR: No camera detected!")
    pygame.quit()
    sys.exit(1)

start_time_ref = time.time()

# ============================================================
# GAME STATE
# ============================================================

treasure_pos = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
hand_smooth = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
hand_velocity = np.array([0.0, 0.0], dtype=float)

hand_grip = 1.0
hand_tracking_lost = True
threats = []
chest_state = "IDLE"
grab_start_time = None
last_threat_spawn = {}
game_over = False
win = False
lives = MAX_LIVES
hit_anim_timer = 0.0
score = 0
threats_dodged = 0

combo = 0
combo_timer = 0
COMBO_TIMEOUT = 90
combo_flash_timer = 0

score_popups = []
particles = ParticleSystem()
heart_shake_timer = 0

level_flash_timer = 0
level_flash_color = (94, 234, 212)

current_level = 1
level_start_time = None
level_up_notification_timer = 0
level_up_notification_level = 0
total_score = 0

# Hand tracking and pause state
game_paused = False
pause_message_timer = 0
hand_was_detected = False
restart_cooldown = 0
RESTART_COOLDOWN_TIME = 12.0  # 12 seconds before restart allowed

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def play_random_sound(sound_list):
    if sound_list:
        random.choice(sound_list).play()

def get_grip_value(lm):
    wrist, mcp = lm[0], lm[9]
    hand_size = math.dist((wrist.x, wrist.y), (mcp.x, mcp.y))
    if hand_size < 0.01: return 1.0
    tips = [8, 12, 16, 20]
    avg_finger_dist = sum(math.dist((wrist.x, wrist.y), (lm[i].x, lm[i].y)) for i in tips) / 4.0
    return avg_finger_dist / hand_size

def is_valid_hand_size(lm):
    try:
        wrist, mcp = lm[0], lm[9]
        hand_size = math.dist((wrist.x, wrist.y), (mcp.x, mcp.y))
        return 0.05 < hand_size < 0.40
    except:
        return True

def apply_camera_zoom(frame):
    """Apply intelligent ROI processing with dynamic tracking"""
    if not CAMERA_ZOOM_ENABLED:
        return frame
    
    h, w = frame.shape[:2]
    
    # Define gameplay ROI (exclude far edges where no gameplay happens)
    roi_margin_x = int(w * 0.10)  # 10% margin on sides
    roi_margin_y = int(h * 0.10)  # 10% margin top/bottom
    
    # ROI coordinates (where actual gameplay happens)
    roi_x1 = roi_margin_x
    roi_y1 = roi_margin_y
    roi_x2 = w - roi_margin_x
    roi_y2 = h - roi_margin_y
    
    # Extract ROI
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # Enhance ROI for better hand detection
    roi_enhanced = roi.copy()
    
    # Apply brightness/contrast enhancement to ROI only
    try:
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(roi_enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
        l_enhanced = clahe.apply(l)
        
        # Slight brightness boost
        l_enhanced = cv2.add(l_enhanced, 8)
        l_enhanced = np.clip(l_enhanced, 0, 255).astype(np.uint8)
        
        # Merge back
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        roi_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    except:
        pass
    
    # Scale ROI to full frame size (zoom effect)
    zoomed = cv2.resize(roi_enhanced, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return zoomed

def spawn_threat(threat_type, chest_position=None):
    """Spawn specific threat type"""
    level_config = LEVELS[current_level]
    threat_config = THREAT_TYPES[threat_type]
    
    # Rocket and Nuke from RIGHT side ONLY
    if threat_type in ["rocket", "nuke"]:
        start_x = WIDTH + 100  # RIGHT SIDE
        start_y = random.randint(100, HEIGHT - 100)
    else:
        # Other threats from any side
        side = random.randint(0, 3)
        
        if side == 0:
            start_x = random.randint(100, WIDTH - 100)
            start_y = -100
        elif side == 1:
            start_x = WIDTH + 100
            start_y = random.randint(100, HEIGHT - 100)
        elif side == 2:
            start_x = random.randint(100, WIDTH - 100)
            start_y = HEIGHT + 100
        else:
            start_x = -100
            start_y = random.randint(100, HEIGHT - 100)
    
    start_pos = np.array([start_x, start_y], dtype=float)
    
    # ALWAYS target towards chest (current position)
    if chest_position is not None:
        target = chest_position
    else:
        target = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
    
    to_target = target - start_pos
    to_target_normalized = to_target / np.linalg.norm(to_target)
    
    accuracy = level_config["targeting_accuracy"]
    
    # Add slight randomness based on accuracy
    perpendicular = np.array([-to_target_normalized[1], to_target_normalized[0]])
    random_offset = perpendicular * random.uniform(-0.3, 0.3) * (1 - accuracy)
    
    direction = to_target_normalized + random_offset
    direction = direction / np.linalg.norm(direction)
    
    # Calculate angle to point towards target (for rockets/nukes)
    # BUT: Don't rotate rockets and nukes - keep them as-is
    if threat_type in ["rocket", "nuke"]:
        angle_to_target = 0  # No rotation for rockets/nukes
    else:
        angle_to_target = math.degrees(math.atan2(direction[1], direction[0]))
    
    speed_range = level_config["threat_speed"]
    base_speed = random.uniform(speed_range[0], speed_range[1])
    
    # Speed multiplier based on threat type
    if threat_type == "threat":
        speed_mult = 1.0
    elif threat_type == "rocket":
        speed_mult = 1.3
    elif threat_type == "grenade":
        speed_mult = 0.9
    elif threat_type == "nuke":
        speed_mult = 1.5
    
    speed = base_speed * speed_mult
    vel = direction * speed
    
    return {
        "id": time.time() + random.random(),
        "pos": start_pos,
        "vel": vel,
        "angle": angle_to_target,  # Point towards target
        "rotation_speed": 0,  # No spinning - stay pointed
        "lifetime": 0.0,
        "type": threat_type
    }

def update_hand_physics(raw_pos, current_smooth, current_vel):
    if raw_pos is not None:
        distance = np.linalg.norm(raw_pos - current_smooth)
        
        if distance > 50:
            smooth_factor = 0.45
        elif distance > 20:
            smooth_factor = 0.28
        else:
            smooth_factor = 0.15
        
        new_velocity = raw_pos - current_smooth
        smooth = (raw_pos * smooth_factor) + (current_smooth * (1 - smooth_factor))
        vel = (new_velocity * 0.5) + (current_vel * 0.5)
        return smooth, vel
    return current_smooth + current_vel, current_vel * 0.85

def reset_game():
    global treasure_pos, hand_smooth, hand_velocity, hand_grip, hand_tracking_lost
    global threats, chest_state, grab_start_time, game_over, win, lives, hit_anim_timer
    global score, threats_dodged, last_threat_spawn
    global current_level, level_start_time, total_score
    global combo, combo_timer, combo_flash_timer, score_popups
    global heart_shake_timer, level_flash_timer, level_flash_color
    global level_up_notification_timer, level_up_notification_level
    global game_paused, pause_message_timer, hand_was_detected, restart_cooldown
    
    treasure_pos = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
    hand_smooth = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
    hand_velocity = np.array([0.0, 0.0], dtype=float)
    
    hand_grip = 1.0
    hand_tracking_lost = True
    threats = []
    chest_state = "IDLE"
    grab_start_time = None
    last_threat_spawn = {}
    game_over = False
    win = False
    lives = MAX_LIVES
    hit_anim_timer = 0.0
    score = 0
    threats_dodged = 0
    
    combo = 0
    combo_timer = 0
    combo_flash_timer = 0
    score_popups.clear()
    heart_shake_timer = 0
    level_flash_timer = 0
    level_flash_color = (94, 234, 212)
    
    current_level = 1
    level_start_time = None
    level_up_notification_timer = 0
    level_up_notification_level = 0
    total_score = 0
    
    game_paused = False
    pause_message_timer = 0
    hand_was_detected = False
    restart_cooldown = 0

def advance_to_next_level():
    global current_level, level_start_time, grab_start_time
    global threats, last_threat_spawn, score, threats_dodged
    global combo, combo_timer, game_over, win
    global level_up_notification_timer, level_up_notification_level
    global level_flash_timer, level_flash_color
    
    current_level += 1
    
    # SEAMLESS - no interruption!
    level_up_notification_timer = 120  # 2 seconds
    level_up_notification_level = current_level
    
    # Continue from current time
    level_start_time = current_time
    threats.clear()
    last_threat_spawn.clear()
    score = 0
    threats_dodged = 0
    combo = 0
    combo_timer = 0
    game_over = False
    win = False
    
    level_flash_timer = 60
    level_flash_color = LEVELS[current_level]["color"]
    
    if level_up_sound:
        try:
            level_up_sound.play()
        except:
            pass

def get_scaled_background():
    global background_img
    if background_img is not None:
        try:
            return pygame.transform.smoothscale(background_img, (WIDTH, HEIGHT))
        except:
            return None
    return None

frame_count = 0
scaled_background = get_scaled_background()

# ============================================================
# MAIN LOOP
# ============================================================

try:
    running = True
    last_cleanup_time = time.time()
    
    while running:
        clock.tick(60)
        current_time = time.time()
        frame_count += 1
        
        if current_time - last_cleanup_time > 5.0:
            last_cleanup_time = current_time
            import gc
            gc.collect()
        
        ret, frame = cap.read()
        if not ret:
            if frame_count % 30 == 0:
                cap.release()
                cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(CAMERA_INDEX)
            continue
        
        frame = cv2.flip(frame, 1)
        frame = apply_camera_zoom(frame)  # Now includes ROI + enhancement
        
        # Frame is already enhanced by ROI function
        frame_proc = frame
        
        try:
            rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res_hand = landmarker_hand.detect_for_video(mp_image, int((current_time - start_time_ref) * 1000))
        except:
            res_hand = None
        
        raw_pos = None
        if res_hand and res_hand.hand_landmarks:
            lm = res_hand.hand_landmarks[0]
            wrist = lm[0]
            
            if wrist.x < 0.05 or wrist.x > 0.95 or wrist.y < 0.05 or wrist.y > 0.95:
                pass
            elif is_valid_hand_size(lm):
                px = np.array([wrist.x * WIDTH, wrist.y * HEIGHT], dtype=float)
                grip = get_grip_value(lm)
                
                if 0.3 < grip < 4.0:
                    raw_pos = px
                    hand_grip = grip
                    hand_tracking_lost = False
        
        if raw_pos is None:
            hand_tracking_lost = True
        else:
            hand_was_detected = True
        
        # Pause game if hand lost during gameplay
        if grab_start_time and not game_over:
            if hand_tracking_lost and hand_was_detected:
                if not game_paused:
                    game_paused = True
                    pause_message_timer = 180
            elif not hand_tracking_lost and game_paused:
                game_paused = False
        
        hand_smooth, hand_velocity = update_hand_physics(raw_pos, hand_smooth, hand_velocity)
        
        if scaled_background:
            screen.blit(scaled_background, (0, 0))
        else:
            screen.fill(DARK_BG)
        
        # Level flash
        if level_flash_timer > 0 and grab_start_time:
            alpha = int(150 * (level_flash_timer / 60))
            flash_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            flash_surf.fill((*level_flash_color, alpha))
            screen.blit(flash_surf, (0, 0))
        
        # Camera preview
        if ret:
            try:
                preview_size = (DEBUG_WINDOW_MAX_WIDTH, DEBUG_WINDOW_MAX_HEIGHT)
                debug_frame = cv2.resize(frame, preview_size)
                debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
                debug_frame = np.rot90(debug_frame)
                debug_frame = np.flipud(debug_frame)
                debug_surface = pygame.surfarray.make_surface(debug_frame)
                
                debug_x = WIDTH - preview_size[0] - 20
                debug_y = 20
                
                pygame.draw.rect(screen, (0, 0, 0), 
                               (debug_x - 3, debug_y - 3, preview_size[0] + 6, preview_size[1] + 6), 
                               border_radius=8)
                pygame.draw.rect(screen, UI_ACCENT, 
                               (debug_x - 2, debug_y - 2, preview_size[0] + 4, preview_size[1] + 4), 
                               2, border_radius=8)
                
                screen.blit(debug_surface, (debug_x, debug_y))
            except:
                pass
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Fist to restart on game over (with cooldown)
        if game_over:
            if restart_cooldown > 0:
                restart_cooldown = max(0, restart_cooldown - (1.0 / 60.0))
            
            if restart_cooldown <= 0 and not hand_tracking_lost:
                if hand_grip < P_GRAB_THRESH:  # Closed fist
                    reset_game()
        
        level_config = LEVELS[current_level]
        
        if not game_over and not game_paused:
            # Update timers
            if combo_timer > 0:
                combo_timer -= 1
                if combo_timer == 0:
                    combo = 0
            
            if combo_flash_timer > 0:
                combo_flash_timer -= 1
            
            if heart_shake_timer > 0:
                heart_shake_timer -= 1
            
            if level_flash_timer > 0:
                level_flash_timer -= 1
            
            if level_up_notification_timer > 0:
                level_up_notification_timer -= 1
            
            # Spawn threats
            if grab_start_time:
                # Regular spawning
                max_threats = level_config["max_threats"]
                spawn_rate = level_config["spawn_rate"]
                
                if len(threats) < max_threats and random.random() < spawn_rate:
                    # Pick random allowed threat
                    threat_type = random.choice(level_config["allowed_threats"])
                    threats.append(spawn_threat(threat_type, treasure_pos))
                
                # Interval-based special threats
                for threat_type, interval in level_config["threat_intervals"].items():
                    if threat_type not in last_threat_spawn:
                        last_threat_spawn[threat_type] = grab_start_time
                    
                    if (current_time - last_threat_spawn[threat_type]) >= interval:
                        if threat_type in level_config["allowed_threats"]:
                            threats.append(spawn_threat(threat_type, treasure_pos))
                            last_threat_spawn[threat_type] = current_time
            
            # Chest control
            is_holding = hand_grip < P_GRAB_THRESH and not hand_tracking_lost
            
            if chest_state == "IDLE" and is_holding:
                if np.linalg.norm(hand_smooth - treasure_pos) < BASE_GRAB_RADIUS:
                    chest_state = "GRABBED"
                    if grab_start_time is None:
                        grab_start_time = current_time
                        level_start_time = current_time
                        
                        level_flash_timer = 60
                        level_flash_color = level_config["color"]
                        
            elif chest_state == "GRABBED":
                if is_holding:
                    treasure_pos += (hand_smooth - treasure_pos) * MOVE_SMOOTHING
                else:
                    chest_state = "IDLE"
            
            # Check for AUTO level advancement
            if grab_start_time and level_start_time:
                time_in_level = current_time - level_start_time
                
                # AUTO-ADVANCE when level duration complete
                if time_in_level >= level_config["duration"] and current_level < 4:
                    advance_to_next_level()
                
                # Check total game time for WIN
                total_game_elapsed = current_time - grab_start_time
                if total_game_elapsed >= TOTAL_GAME_TIME:
                    game_over = True
                    win = True
                    restart_cooldown = RESTART_COOLDOWN_TIME
            
            # Update threats
            for i in range(len(threats) - 1, -1, -1):
                t = threats[i]
                t["lifetime"] += 1/60
                t["pos"] += t["vel"]
                t["angle"] += t["rotation_speed"]
                
                # Dodge - off screen
                if (t["pos"][0] < -200 or t["pos"][0] > WIDTH + 200 or 
                    t["pos"][1] < -200 or t["pos"][1] > HEIGHT + 200):
                    
                    combo += 1
                    combo_timer = COMBO_TIMEOUT
                    combo_flash_timer = 30
                    
                    threat_config = THREAT_TYPES[t["type"]]
                    base_points = threat_config["points"]
                    
                    # Particle color based on threat
                    if t["type"] == "threat":
                        particle_color = (100, 255, 100)
                    elif t["type"] == "rocket":
                        particle_color = (255, 200, 100)
                    elif t["type"] == "grenade":
                        particle_color = (255, 215, 0)
                    else:  # nuke
                        particle_color = (255, 100, 255)
                    
                    combo_bonus = combo * 2 if combo > 1 else 0
                    total_points = base_points + combo_bonus
                    
                    score += total_points
                    total_score += total_points
                    threats_dodged += 1
                    
                    particles.add_particles(t["pos"][0], t["pos"][1], particle_color, 15)
                    
                    score_popups.append({
                        "text": f"+{total_points}",
                        "pos": [t["pos"][0], t["pos"][1]],
                        "vel": [0, -2],
                        "lifetime": 60,
                        "color": (255, 215, 0)
                    })
                    
                    if combo > 1:
                        score_popups.append({
                            "text": f"{combo}x COMBO!",
                            "pos": [t["pos"][0], t["pos"][1] - 30],
                            "vel": [0, -3],
                            "lifetime": 60,
                            "color": (255, 150, 0)
                        })
                    
                    play_random_sound(dodge_sounds)
                    
                    threats.pop(i)
                    continue
                
                # Collision
                threat_config = THREAT_TYPES[t["type"]]
                collision_radius = BASE_BORDER_RADIUS + threat_config["collision_radius"]
                
                if np.linalg.norm(t["pos"] - treasure_pos) < collision_radius:
                    damage = threat_config["damage"]
                    
                    lives -= damage
                    threats.pop(i)
                    hit_anim_timer = current_time
                    
                    combo = 0
                    combo_timer = 0
                    
                    # Explosion color based on damage
                    if damage >= 3.0:
                        particle_color = (255, 100, 255)
                        particle_count = 40
                    elif damage >= 1.0:
                        particle_color = (255, 150, 0)
                        particle_count = 30
                    else:
                        particle_color = (255, 0, 0)
                        particle_count = 20
                    
                    particles.add_particles(t["pos"][0], t["pos"][1], particle_color, particle_count)
                    heart_shake_timer = 30
                    
                    play_random_sound(hit_sounds)
                    
                    if lives <= 0:
                        game_over = True
                        win = False
                        restart_cooldown = RESTART_COOLDOWN_TIME
            
            # Update score popups
            for popup in score_popups[:]:
                popup["pos"][1] += popup["vel"][1]
                popup["lifetime"] -= 1
                
                if popup["lifetime"] <= 0:
                    score_popups.remove(popup)
        
        # Draw game
        disp_pos = treasure_pos.copy()
        is_hit = (current_time - hit_anim_timer) < HIT_FLASH_DURATION
        
        if is_hit:
            disp_pos += np.array([random.randint(-SHAKE_INTENSITY, SHAKE_INTENSITY),
                                 random.randint(-SHAKE_INTENSITY, SHAKE_INTENSITY)])
            flash_alpha = int((1 - (current_time - hit_anim_timer) / HIT_FLASH_DURATION) * 200)
            flash = pygame.Surface((BASE_BORDER_RADIUS * 4, BASE_BORDER_RADIUS * 4))
            flash.set_alpha(flash_alpha)
            flash.fill(RED)
            screen.blit(flash, (int(disp_pos[0] - BASE_BORDER_RADIUS * 2), int(disp_pos[1] - BASE_BORDER_RADIUS * 2)))
            
            hit_text = big_font.render("HIT!", True, WHITE)
            for i in range(3):
                glow = big_font.render("HIT!", True, RED)
                glow.set_alpha(150 - i*50)
                screen.blit(glow, (int(treasure_pos[0]-60 + i), int(treasure_pos[1]-100 + i)))
            screen.blit(hit_text, (int(treasure_pos[0]-60), int(treasure_pos[1]-100)))
            c_col = DANGER
        else:
            if grab_start_time:
                c_col = SUCCESS
                pulse = abs(math.sin(current_time * 3)) * 0.3 + 0.7
                actual_radius = int((BASE_BORDER_RADIUS + 5) * pulse)
                pygame.draw.circle(screen, SUCCESS, disp_pos.astype(int), actual_radius, 4)
            else:
                c_col = GOLD
                pulse = abs(math.sin(current_time * 1.5)) * 0.3 + 0.7
                actual_radius = int((BASE_BORDER_RADIUS + 8) * pulse)
                pygame.draw.circle(screen, GOLD, disp_pos.astype(int), actual_radius, 4)
        
        pygame.draw.circle(screen, c_col, disp_pos.astype(int), BASE_BORDER_RADIUS, 5)
        pygame.draw.circle(screen, c_col, disp_pos.astype(int), BASE_BORDER_RADIUS - 5, 2)
        screen.blit(chest_img, (int(disp_pos[0]-45), int(disp_pos[1]-45)))
        
        # Initial instruction (before game starts)
        if not grab_start_time and not game_over:
            pulse = abs(math.sin(current_time * 2)) * 0.3 + 0.7
            instruction_color = tuple(int(c * pulse) for c in GOLD)
            
            instruction_text = big_font.render("👊 CLOSE YOUR FIST", True, instruction_color)
            instruction_rect = instruction_text.get_rect(center=(WIDTH//2, HEIGHT//2 - 100))
            screen.blit(instruction_text, instruction_rect)
            
            grab_text = medium_font.render("GRAB THE CHEST TO START!", True, WHITE)
            grab_rect = grab_text.get_rect(center=(WIDTH//2, HEIGHT//2 - 30))
            screen.blit(grab_text, grab_rect)
            
            # Show damage legend on start screen
            legend_x = WIDTH - 320
            legend_y = HEIGHT - 140
            
            legend_bg = pygame.Surface((300, 120))
            legend_bg.set_alpha(180)
            legend_bg.fill(UI_BG)
            screen.blit(legend_bg, (legend_x, legend_y))
            pygame.draw.rect(screen, UI_BORDER, (legend_x, legend_y, 300, 120), 2, border_radius=5)
            
            legend_title = small_font.render("THREAT DAMAGE:", True, WHITE)
            screen.blit(legend_title, (legend_x + 10, legend_y + 5))
            
            threats_info = [
                ("🔴 Threat: -0.5", (255, 100, 100)),
                ("🚀 Rocket: -1.0", (255, 150, 50)),
                ("💣 Grenade: DEATH", (255, 200, 0)),
                ("☢️  Nuke: DEATH", (200, 0, 255))
            ]
            
            for i, (text, color) in enumerate(threats_info):
                info_text = small_font.render(text, True, color)
                screen.blit(info_text, (legend_x + 10, legend_y + 30 + i * 22))
        
        # Draw threats
        for t in threats:
            threat_img = threat_images[t["type"]]
            rot = pygame.transform.rotate(threat_img, t.get("angle", 0))
            screen.blit(rot, rot.get_rect(center=(int(t["pos"][0]), int(t["pos"][1]))).topleft)
        
        # Particles
        particles.update_and_draw(screen)
        
        # Score popups
        for popup in score_popups:
            alpha = int(255 * (popup["lifetime"] / 60))
            popup_text = font.render(popup["text"], True, popup["color"])
            popup_text.set_alpha(alpha)
            screen.blit(popup_text, (int(popup["pos"][0]) - popup_text.get_width()//2, 
                                     int(popup["pos"][1])))
        
        # Hand cursor
        if not hand_tracking_lost:
            pygame.draw.circle(screen, (0, 0, 0), hand_smooth.astype(int), 22)
            pygame.draw.circle(screen, CYAN, hand_smooth.astype(int), 20, 4)
        
        # UI during gameplay
        if grab_start_time and not game_over:
            total_elapsed = current_time - grab_start_time
            total_remaining = max(0, int(TOTAL_GAME_TIME - total_elapsed))
            
            stats_x = 40
            stats_y = 40
            level_color = level_config["color"]
            
            # Wave indicator
            wave_surf = big_font.render(f"WAVE {current_level}/4", True, level_color)
            screen.blit(wave_surf, (stats_x, stats_y))
            
            # Hearts with pulse and shake
            full_hearts = int(lives)
            has_half = (lives % 1) >= 0.5
            
            heart_x = stats_x
            heart_y = stats_y + 80
            
            if heart_shake_timer > 0:
                heart_x += random.randint(-5, 5)
                heart_y += random.randint(-5, 5)
            
            for i in range(4):
                pulse = abs(math.sin(current_time * 3 + i * 0.5)) * 0.15 + 0.85
                
                if i < full_hearts:
                    heart = "❤️"
                    color = SUCCESS
                elif i == full_hearts and has_half:
                    heart = "💔"
                    color = WARNING
                else:
                    heart = "💔"
                    color = DANGER
                
                heart_surf = big_font.render(heart, True, color)
                
                if i < full_hearts or (i == full_hearts and has_half):
                    scaled_size = (int(heart_surf.get_width() * pulse), 
                                  int(heart_surf.get_height() * pulse))
                    heart_surf = pygame.transform.scale(heart_surf, scaled_size)
                
                screen.blit(heart_surf, (heart_x + i * 55, heart_y))
            
            # Total game timer (60s)
            timer_color = DANGER if total_remaining <= 10 else WARNING if total_remaining <= 20 else CYAN
            timer_surf = big_font.render(f"⏱️ {total_remaining}s", True, timer_color)
            screen.blit(timer_surf, (stats_x, stats_y + 160))
            
            # Score
            score_surf = big_font.render(f"⭐ {total_score}", True, GOLD)
            screen.blit(score_surf, (stats_x, stats_y + 240))
            
            # Threat damage legend (bottom right)
            legend_x = WIDTH - 320
            legend_y = HEIGHT - 140
            
            legend_bg = pygame.Surface((300, 120))
            legend_bg.set_alpha(180)
            legend_bg.fill(UI_BG)
            screen.blit(legend_bg, (legend_x, legend_y))
            pygame.draw.rect(screen, UI_BORDER, (legend_x, legend_y, 300, 120), 2, border_radius=5)
            
            legend_title = small_font.render("THREAT DAMAGE:", True, WHITE)
            screen.blit(legend_title, (legend_x + 10, legend_y + 5))
            
            threats_info = [
                ("🔴 Threat: -0.5", (255, 100, 100)),
                ("🚀 Rocket: -1.0", (255, 150, 50)),
                ("💣 Grenade: DEATH", (255, 200, 0)),
                ("☢️  Nuke: DEATH", (200, 0, 255))
            ]
            
            for i, (text, color) in enumerate(threats_info):
                info_text = small_font.render(text, True, color)
                screen.blit(info_text, (legend_x + 10, legend_y + 30 + i * 22))
            
            # Combo display - smaller, more transparent, positioned higher
            if combo > 1:
                pulse = abs(math.sin(current_time * 5)) * 0.3 + 0.7
                combo_color = (int(255 * pulse), int(215 * pulse), 0)
                
                # Smaller font for combo to not block view
                combo_text = big_font.render(f"{combo}x COMBO!", True, combo_color)
                
                # More transparent - max 180 instead of 255
                alpha = min(180, combo_timer * 2)
                combo_text.set_alpha(alpha)
                
                # Position higher up - 100px instead of 200px
                combo_rect = combo_text.get_rect(center=(WIDTH//2, 100))
                
                # Lighter glow effect
                for i in range(2):  # Only 2 glow layers instead of 3
                    glow = big_font.render(f"{combo}x COMBO!", True, combo_color)
                    glow.set_alpha(max(0, alpha - 80 - i*40))
                    glow_rect = glow.get_rect(center=(WIDTH//2 + i*2, 100 + i*2))
                    screen.blit(glow, glow_rect)
                
                screen.blit(combo_text, combo_rect)
        
        # Pause overlay (when hand lost)
        if game_paused and pause_message_timer > 0:
            pause_message_timer -= 1
            
            overlay = pygame.Surface((WIDTH, HEIGHT))
            overlay.set_alpha(200)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            
            pulse = abs(math.sin(current_time * 3)) * 0.3 + 0.7
            pause_color = tuple(int(c * pulse) for c in WARNING)
            
            pause_text = title_font.render("⚠️ PAUSED ⚠️", True, pause_color)
            pause_rect = pause_text.get_rect(center=(WIDTH//2, HEIGHT//2 - 50))
            screen.blit(pause_text, pause_rect)
            
            hand_text = big_font.render("Show your hand to resume", True, WHITE)
            hand_rect = hand_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 50))
            screen.blit(hand_text, hand_rect)
        
        # Level up notification (seamless transition)
        if level_up_notification_timer > 0:
            if level_up_notification_timer > 90:
                alpha = int(255 * ((120 - level_up_notification_timer) / 30))
            elif level_up_notification_timer < 30:
                alpha = int(255 * (level_up_notification_timer / 30))
            else:
                alpha = 255
            
            level_color = LEVELS[level_up_notification_level]["color"]
            
            level_text = title_font.render(f"WAVE {level_up_notification_level}", True, level_color)
            level_text.set_alpha(alpha)
            level_rect = level_text.get_rect(center=(WIDTH//2, HEIGHT//2))
            
            for i in range(3):
                glow = title_font.render(f"WAVE {level_up_notification_level}", True, level_color)
                glow.set_alpha(max(0, alpha - 100 - i*30))
                glow_rect = glow.get_rect(center=(WIDTH//2 + i*2, HEIGHT//2 + i*2))
                screen.blit(glow, glow_rect)
            
            screen.blit(level_text, level_rect)
        
        # Game over screen
        if game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT))
            overlay.set_alpha(230)
            overlay.fill(DARK_BG)
            screen.blit(overlay, (0, 0))
            
            if win:
                result_text = title_font.render("VICTORY!", True, GOLD)
                result_color = GOLD
            else:
                result_text = title_font.render("GAME OVER", True, RED)
                result_color = RED
            
            for i in range(3):
                glow = title_font.render("VICTORY!" if win else "GAME OVER", True, result_color)
                glow.set_alpha(100 - i*30)
                glow_rect = glow.get_rect(center=(WIDTH//2 + i*3, HEIGHT//4 + i*3))
                screen.blit(glow, glow_rect)
            
            result_rect = result_text.get_rect(center=(WIDTH//2, HEIGHT//4))
            screen.blit(result_text, result_rect)
            
            final_score_surf = big_font.render(f"Final Score: {total_score}", True, WHITE)
            final_score_rect = final_score_surf.get_rect(center=(WIDTH//2, HEIGHT//2))
            screen.blit(final_score_surf, final_score_rect)
            
            dodged_surf = medium_font.render(f"Threats Dodged: {threats_dodged}", True, CYAN)
            dodged_rect = dodged_surf.get_rect(center=(WIDTH//2, HEIGHT//2 + 80))
            screen.blit(dodged_surf, dodged_rect)
            
            pulse = abs(math.sin(current_time * 2)) * 0.3 + 0.7
            restart_color = tuple(int(c * pulse) for c in UI_ACCENT)
            
            if restart_cooldown > 0:
                # Show cooldown timer
                cooldown_text = f"Wait {int(restart_cooldown)}s..."
                restart_surf = medium_font.render(cooldown_text, True, (150, 150, 150))
            else:
                # Ready to restart
                restart_surf = medium_font.render("👊 CLOSE FIST TO RESTART", True, restart_color)
            
            restart_rect = restart_surf.get_rect(center=(WIDTH//2, HEIGHT - 150))
            screen.blit(restart_surf, restart_rect)
        
        pygame.display.flip()

except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    pygame.quit()
