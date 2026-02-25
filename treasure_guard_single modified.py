import os, sys, time, random, math, cv2, pygame
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

print("TREASURE GUARD - Level Mode ENHANCED")

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

pygame.display.set_caption("Treasure Guard - Enhanced")

CAMERA_INDEX = auto_select_camera()
SHOW_CAMERA_DEBUG = True
DEBUG_WINDOW_MAX_WIDTH = 180
DEBUG_WINDOW_MAX_HEIGHT = 101

MODEL_PATH_HAND = "models/hand_landmarker.task"

# Camera zoom settings - FIXED at 1.5x
CAMERA_ZOOM_ENABLED = True
CAMERA_ZOOM_AMOUNT = 0.15

# ============================================================
# LEVEL SYSTEM CONFIGURATION
# ============================================================

LEVELS = {
    1: {
        "name": "ROOKIE GUARD",
        "duration": 30,
        "max_threats": 3,
        "spawn_rate": 0.02,
        "threat_speed": (2.5, 3.5),
        "grenade_enabled": False,
        "grenade_interval": None,
        "grenade_speed": None,
        "boss_attack": False,
        "description": "Learn the basics",
        "color": (94, 234, 212),
        "required_score": 50
    },
    2: {
        "name": "SKILLED DEFENDER",
        "duration": 35,
        "max_threats": 4,
        "spawn_rate": 0.03,
        "threat_speed": (3.0, 4.5),
        "grenade_enabled": True,
        "grenade_interval": 15.0,
        "grenade_speed": (2.5, 3.0),
        "boss_attack": False,
        "description": "Grenades incoming!",
        "color": (87, 242, 135),
        "required_score": 100
    },
    3: {
        "name": "EXPERT PROTECTOR",
        "duration": 40,
        "max_threats": 5,
        "spawn_rate": 0.04,
        "threat_speed": (3.5, 5.5),
        "grenade_enabled": True,
        "grenade_interval": 10.0,
        "grenade_speed": (2.8, 3.5),
        "boss_attack": False,
        "description": "Chaos intensifies",
        "color": (255, 154, 88),
        "required_score": 150
    },
    4: {
        "name": "FINAL STAND",
        "duration": 45,
        "max_threats": 6,
        "spawn_rate": 0.05,
        "threat_speed": (4.0, 6.0),
        "grenade_enabled": True,
        "grenade_interval": 8.0,
        "grenade_speed": (3.0, 4.0),
        "boss_attack": True,
        "description": "SURVIVE THE ONSLAUGHT!",
        "color": (239, 68, 68),
        "required_score": 200
    }
}

# ============================================================
# FILE PATHS
# ============================================================

CHEST_IMAGE_PATH = "assets/chest.png"
THREAT_IMAGE_PATH = "assets/threat.png"
GRENADE_IMAGE_PATH = "assets/grenade.png"
BACKGROUND_IMAGE_PATH = "assets/background.png"
BACKGROUND_MUSIC_PATH = "assets/background_music.mp3"
HIT_SOUND_PATH = "assets/hit_sound.wav"
LEVEL_UP_SOUND_PATH = "assets/level_up.wav"

# ============================================================
# GAME CONFIGURATION
# ============================================================

TREASURE_SIZE, THREAT_SIZE = 90, 60
GRENADE_SIZE = 90
BASE_BORDER_RADIUS, BASE_GRAB_RADIUS = 70, 180
MAX_LIVES = 3.0
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

# ============================================================
# MEDIAPIPE SETUP
# ============================================================

options_hand = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH_HAND),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.3,  # Lowered from 0.4
    min_tracking_confidence=0.3          # Lowered from 0.4
)
landmarker_hand = vision.HandLandmarker.create_from_options(options_hand)

pygame.mixer.init()
clock = pygame.time.Clock()

# ============================================================
# FONTS
# ============================================================

try:
    title_font = pygame.font.Font(None, 180)
    big_font = pygame.font.Font(None, 120)
    medium_font = pygame.font.Font(None, 70)
    font = pygame.font.Font(None, 45)
    small_font = pygame.font.Font(None, 32)
except:
    title_font = pygame.font.Font(None, 150)
    big_font = pygame.font.Font(None, 100)
    medium_font = pygame.font.Font(None, 60)
    font = pygame.font.SysFont(None, 35)
    small_font = pygame.font.Font(None, 28)

# ============================================================
# PARTICLE SYSTEM
# ============================================================

class ParticleSystem:
    """Pre-rendered particle system for performance"""
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

# Load sound variations
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
    try:
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.smoothscale(img, size)
    except:
        surf = pygame.Surface(size)
        surf.fill(fallback_color)
        return surf

chest_img = load_scale(CHEST_IMAGE_PATH, (TREASURE_SIZE, TREASURE_SIZE), YELLOW)
threat_img = load_scale(THREAT_IMAGE_PATH, (THREAT_SIZE, THREAT_SIZE), RED)
grenade_img = load_scale(GRENADE_IMAGE_PATH, (GRENADE_SIZE, GRENADE_SIZE), ORANGE)

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
last_grenade_time = None
game_over = False
win = False
lives = MAX_LIVES
hit_anim_timer = 0.0
score = 0
threats_dodged = 0

# Combo system
combo = 0
combo_timer = 0
COMBO_TIMEOUT = 90
combo_flash_timer = 0

# Score popups
score_popups = []

# Particle system
particles = ParticleSystem()

# Heart shake
heart_shake_timer = 0

# Boss warning
boss_warning_timer = 0
boss_warned = False

# Level flash
level_flash_timer = 0
level_flash_color = (94, 234, 212)

# Level system state
current_level = 1
level_start_time = None
level_up_notification_timer = 0  # Show "LEVEL 2" briefly
level_up_notification_level = 0
total_score = 0
level_scores = {1: 0, 2: 0, 3: 0, 4: 0}

# Boss attack state
boss_attack_active = False
boss_attack_start = None
BOSS_ATTACK_DURATION = 10.0

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def play_random_sound(sound_list):
    """Play random sound from list"""
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
    if not CAMERA_ZOOM_ENABLED:
        return frame
    
    h, w = frame.shape[:2]
    top = int(h * CAMERA_ZOOM_AMOUNT)
    bottom = int(h * (1 - CAMERA_ZOOM_AMOUNT))
    left = int(w * CAMERA_ZOOM_AMOUNT)
    right = int(w * (1 - CAMERA_ZOOM_AMOUNT))
    
    cropped = frame[top:bottom, left:right]
    zoomed = cv2.resize(cropped, (w, h))
    
    return zoomed

def spawn_threat(is_grenade=False, chest_position=None):
    """Spawn threat with level-specific parameters"""
    level_config = LEVELS[current_level]
    
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
    
    if chest_position is not None:
        to_chest = chest_position - start_pos
        to_chest_normalized = to_chest / np.linalg.norm(to_chest)
        
        if boss_attack_active:
            accuracy = 0.95
        elif is_grenade:
            accuracy = random.uniform(0.7, 0.85)
        else:
            accuracy = random.uniform(0.5, 0.75)
        
        perpendicular = np.array([-to_chest_normalized[1], to_chest_normalized[0]])
        random_offset = perpendicular * random.uniform(-0.8, 0.8)
        
        direction = to_chest_normalized * accuracy + random_offset * (1 - accuracy)
        direction = direction / np.linalg.norm(direction)
    else:
        to_center = np.array([WIDTH // 2, HEIGHT // 2], dtype=float) - start_pos
        direction = to_center / np.linalg.norm(to_center)
    
    if is_grenade:
        speed_range = level_config["grenade_speed"]
        speed = random.uniform(speed_range[0], speed_range[1])
    else:
        speed_range = level_config["threat_speed"]
        speed = random.uniform(speed_range[0], speed_range[1])
        
        if boss_attack_active:
            speed *= 1.5
    
    vel = direction * speed
    
    return {
        "id": time.time() + random.random(),
        "pos": start_pos,
        "vel": vel,
        "angle": 0,
        "rotation_speed": random.uniform(-3, 3),
        "lifetime": 0.0,
        "type": "grenade" if is_grenade else "regular"
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

def draw_level_complete(level_num, level_score):
    """Victory screen for completing a level"""
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(230)
    overlay.fill(DARK_BG)
    screen.blit(overlay, (0, 0))
    
    complete_text = title_font.render("LEVEL COMPLETE!", True, GOLD)
    for i in range(3):
        glow = title_font.render("LEVEL COMPLETE!", True, (255, 215, 0, 100 - i*30))
        glow_rect = glow.get_rect(center=(WIDTH//2, HEIGHT//4 + i*3))
        screen.blit(glow, glow_rect)
    
    complete_rect = complete_text.get_rect(center=(WIDTH//2, HEIGHT//4))
    screen.blit(complete_text, complete_rect)
    
    stars = "⭐" * min(3, max(1, level_score // 50))
    stars_surf = title_font.render(stars, True, GOLD)
    stars_rect = stars_surf.get_rect(center=(WIDTH//2, HEIGHT//2 - 50))
    screen.blit(stars_surf, stars_rect)
    
    score_surf = big_font.render(f"Score: {level_score}", True, WHITE)
    score_rect = score_surf.get_rect(center=(WIDTH//2, HEIGHT//2 + 70))
    screen.blit(score_surf, score_rect)
    
    if level_num < 4:
        next_pulse = abs(math.sin(time.time() * 2)) * 0.3 + 0.7
        next_color = tuple(int(c * next_pulse) for c in UI_ACCENT)
        next_surf = medium_font.render("Press SPACE for Next Level", True, next_color)
        next_rect = next_surf.get_rect(center=(WIDTH//2, HEIGHT - 150))
        screen.blit(next_surf, next_rect)

def reset_game():
    global treasure_pos, hand_smooth, hand_velocity, hand_grip, hand_tracking_lost
    global threats, chest_state, grab_start_time, game_over, win, lives, hit_anim_timer
    global score, threats_dodged, last_grenade_time
    global current_level, level_start_time, total_score
    global combo, combo_timer, combo_flash_timer, score_popups
    global heart_shake_timer, boss_warning_timer, boss_warned
    global level_flash_timer, level_flash_color, boss_attack_active, boss_attack_start
    global level_up_notification_timer, level_up_notification_level
    
    treasure_pos = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
    hand_smooth = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
    hand_velocity = np.array([0.0, 0.0], dtype=float)
    
    hand_grip = 1.0
    hand_tracking_lost = True
    threats = []
    chest_state = "IDLE"
    grab_start_time = None
    last_grenade_time = None
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
    boss_warning_timer = 0
    boss_warned = False
    level_flash_timer = 0
    level_flash_color = (94, 234, 212)
    
    current_level = 1
    level_start_time = None
    level_up_notification_timer = 0
    level_up_notification_level = 0
    boss_attack_active = False
    boss_attack_start = None
    total_score = 0
    level_scores.clear()
    for i in range(1, 5):
        level_scores[i] = 0

def advance_to_next_level():
    global current_level, level_start_time
    global grab_start_time, chest_state, threats, last_grenade_time
    global boss_attack_active, boss_attack_start, score, threats_dodged
    global combo, combo_timer, boss_warned
    global level_up_notification_timer, level_up_notification_level, game_over, win
    
    level_scores[current_level] = score
    current_level += 1
    
    # Show level up notification
    level_up_notification_timer = 180  # 3 seconds
    level_up_notification_level = current_level
    
    # Don't reset game - continue seamlessly
    grab_start_time = current_time  # Continue from current time
    level_start_time = current_time
    threats.clear()
    last_grenade_time = None
    boss_attack_active = False
    boss_attack_start = None
    score = 0
    threats_dodged = 0
    combo = 0
    combo_timer = 0
    boss_warned = False
    game_over = False
    win = False
    
    # Level flash
    level_flash_timer = 60
    level_flash_color = LEVELS[current_level]["color"]
    
    if level_up_sound:
        try:
            level_up_sound.play()
        except:
            pass

def draw_text_with_shadow(surface, text, font, color, x, y, shadow_offset=3):
    shadow = font.render(text, True, (0, 0, 0))
    shadow.set_alpha(100)
    surface.blit(shadow, (x + shadow_offset, y + shadow_offset))
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, (x, y))
    return text_surface.get_rect(topleft=(x, y))

def draw_ui_panel(surface, x, y, width, height, bg_color=UI_BG, border_color=UI_BORDER, alpha=200):
    panel = pygame.Surface((width, height))
    panel.set_alpha(alpha)
    panel.fill(bg_color)
    surface.blit(panel, (x, y))
    pygame.draw.rect(surface, border_color, (x, y, width, height), 2, border_radius=8)

def draw_pulse_circle(surface, center, radius, color, pulse_time):
    pulse = abs(math.sin(pulse_time * 3)) * 0.3 + 0.7
    actual_radius = int(radius * pulse)
    pygame.draw.circle(surface, color, center, actual_radius, 4)

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
        frame = apply_camera_zoom(frame)
        
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            l = cv2.add(l, 5)
            l = np.clip(l, 0, 255).astype(np.uint8)
            limg = cv2.merge((l, a, b))
            frame_proc = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except:
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
        
        hand_smooth, hand_velocity = update_hand_physics(raw_pos, hand_smooth, hand_velocity)
        
        if scaled_background:
            screen.blit(scaled_background, (0, 0))
        else:
            screen.fill(DARK_BG)
        
        # Level flash effect
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
                elif event.key == pygame.K_SPACE:
                    if game_over and win and current_level < 4:
                        advance_to_next_level()
                    elif game_over:
                        reset_game()
        
        level_config = LEVELS[current_level]
        
        if not game_over:
            # Update timers
            if combo_timer > 0:
                combo_timer -= 1
                if combo_timer == 0:
                    combo = 0
            
            if combo_flash_timer > 0:
                combo_flash_timer -= 1
            
            if heart_shake_timer > 0:
                heart_shake_timer -= 1
            
            if boss_warning_timer > 0:
                boss_warning_timer -= 1
            
            if level_flash_timer > 0:
                level_flash_timer -= 1
            
            if level_up_notification_timer > 0:
                level_up_notification_timer -= 1
            
            # Spawn threats
            if grab_start_time:
                max_threats = level_config["max_threats"]
                
                if boss_attack_active:
                    max_threats *= 2
                
                spawn_rate = level_config["spawn_rate"]
                if len(threats) < max_threats and random.random() < spawn_rate:
                    threats.append(spawn_threat(is_grenade=False, chest_position=treasure_pos))
                
                if level_config["grenade_enabled"]:
                    if last_grenade_time is None:
                        last_grenade_time = grab_start_time
                    
                    grenade_interval = level_config["grenade_interval"]
                    if boss_attack_active:
                        grenade_interval = 5.0
                    
                    if (current_time - last_grenade_time) >= grenade_interval:
                        threats.append(spawn_threat(is_grenade=True, chest_position=treasure_pos))
                        last_grenade_time = current_time
                
                # Boss warning and activation
                if level_config["boss_attack"]:
                    time_remaining = level_config["duration"] - (current_time - grab_start_time)
                    
                    if 10 < time_remaining <= 15 and not boss_warned:
                        boss_warned = True
                        boss_warning_timer = 180
                    
                    if time_remaining <= BOSS_ATTACK_DURATION and not boss_attack_active:
                        boss_attack_active = True
                        boss_attack_start = current_time
                        
                        for threat in threats:
                            threat["vel"] *= 1.5
            
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
                    
                    if t["type"] == "regular":
                        base_points = 10
                        particle_color = (0, 255, 100)
                    else:
                        base_points = 25
                        particle_color = (255, 200, 0)
                    
                    combo_bonus = combo * 2 if combo > 1 else 0
                    total_points = base_points + combo_bonus
                    
                    score += total_points
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
                collision_radius = BASE_BORDER_RADIUS + (35 if t["type"] == "grenade" else 20)
                if np.linalg.norm(t["pos"] - treasure_pos) < collision_radius:
                    if t["type"] == "grenade":
                        lives = 0
                        threats.pop(i)
                        hit_anim_timer = current_time
                        
                        combo = 0
                        combo_timer = 0
                        
                        particles.add_particles(t["pos"][0], t["pos"][1], (255, 100, 0), 30)
                        heart_shake_timer = 30
                        
                        play_random_sound(hit_sounds)
                        
                        game_over = True
                        win = False
                    else:
                        lives -= 0.5
                        threats.pop(i)
                        hit_anim_timer = current_time
                        
                        combo = 0
                        combo_timer = 0
                        
                        particles.add_particles(t["pos"][0], t["pos"][1], (255, 0, 0), 20)
                        heart_shake_timer = 30
                        
                        play_random_sound(hit_sounds)
                        
                        if lives <= 0:
                            game_over = True
                            win = False
            
            # Update score popups
            for popup in score_popups[:]:
                popup["pos"][1] += popup["vel"][1]
                popup["lifetime"] -= 1
                
                if popup["lifetime"] <= 0:
                    score_popups.remove(popup)
            
            # Level completion
            if grab_start_time and (current_time - grab_start_time) >= level_config["duration"]:
                game_over = True
                win = True
        
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
                draw_pulse_circle(screen, disp_pos.astype(int), BASE_BORDER_RADIUS + 5, SUCCESS, current_time)
            else:
                c_col = GOLD
                draw_pulse_circle(screen, disp_pos.astype(int), BASE_BORDER_RADIUS + 8, GOLD, current_time * 0.5)
        
        pygame.draw.circle(screen, c_col, disp_pos.astype(int), BASE_BORDER_RADIUS, 5)
        pygame.draw.circle(screen, c_col, disp_pos.astype(int), BASE_BORDER_RADIUS - 5, 2)
        screen.blit(chest_img, (int(disp_pos[0]-45), int(disp_pos[1]-45)))
        
        # Draw threats
        for t in threats:
            rot = pygame.transform.rotate(
                grenade_img if t["type"] == "grenade" else threat_img, 
                t.get("angle", 0)
            )
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
            time_elapsed = current_time - grab_start_time
            time_remaining = max(0, int(level_config["duration"] - time_elapsed))
            
            stats_x = 40
            stats_y = 40
            level_color = level_config["color"]
            
            level_surf = big_font.render(f"LEVEL {current_level}", True, level_color)
            screen.blit(level_surf, (stats_x, stats_y))
            
            # Hearts with pulse and shake
            full_hearts = int(lives)
            has_half = (lives % 1) >= 0.5
            
            heart_x = stats_x
            heart_y = stats_y + 80
            
            if heart_shake_timer > 0:
                heart_x += random.randint(-5, 5)
                heart_y += random.randint(-5, 5)
            
            for i in range(3):
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
            
            timer_color = DANGER if time_remaining <= 10 else WARNING if time_remaining <= 15 else CYAN
            timer_surf = big_font.render(f"⏱️ {time_remaining}s", True, timer_color)
            screen.blit(timer_surf, (stats_x, stats_y + 160))
            
            score_surf = big_font.render(f"⭐ {score}", True, GOLD)
            screen.blit(score_surf, (stats_x, stats_y + 240))
            
            # Combo display
            if combo > 1:
                pulse = abs(math.sin(current_time * 5)) * 0.3 + 0.7
                combo_color = (int(255 * pulse), int(215 * pulse), 0)
                combo_text = title_font.render(f"{combo}x COMBO!", True, combo_color)
                
                alpha = min(255, combo_timer * 3)
                combo_text.set_alpha(alpha)
                
                combo_rect = combo_text.get_rect(center=(WIDTH//2, 200))
                
                for i in range(3):
                    glow = title_font.render(f"{combo}x COMBO!", True, combo_color)
                    glow.set_alpha(max(0, alpha - 100 - i*30))
                    glow_rect = glow.get_rect(center=(WIDTH//2 + i*2, 200 + i*2))
                    screen.blit(glow, glow_rect)
                
                screen.blit(combo_text, combo_rect)
            
            # Boss attack warning
            if boss_attack_active:
                boss_pulse = abs(math.sin(current_time * 8)) * 0.5 + 0.5
                boss_color = tuple(int(c * boss_pulse) for c in DANGER)
                boss_text = title_font.render("FINAL WAVE!", True, boss_color)
                boss_rect = boss_text.get_rect(center=(WIDTH//2, 100))
                
                border_alpha = int(180 * boss_pulse)
                border_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                pygame.draw.rect(border_surf, (220, 0, 0, border_alpha), 
                               (0, 0, WIDTH, HEIGHT), 20)
                screen.blit(border_surf, (0, 0))
                
                for i in range(5):
                    glow = title_font.render("FINAL WAVE!", True, DANGER)
                    glow.set_alpha(100 - i*20)
                    glow_rect = glow.get_rect(center=(WIDTH//2 + i*3, 100 + i*3))
                    screen.blit(glow, glow_rect)
                
                screen.blit(boss_text, boss_rect)
            
            elif boss_warning_timer > 0:
                pulse = abs(math.sin(current_time * 4)) * 0.4 + 0.6
                warning_color = (int(255 * pulse), int(100 * pulse), 0)
                
                warning_text = big_font.render("⚠️ FINAL WAVE APPROACHING ⚠️", True, warning_color)
                warning_rect = warning_text.get_rect(center=(WIDTH//2, 200))
                screen.blit(warning_text, warning_rect)
        
        # Level up notification (seamless transition)
        if level_up_notification_timer > 0:
            # Calculate fade based on timer
            if level_up_notification_timer > 150:  # First 0.5s - fade in
                alpha = int(255 * ((180 - level_up_notification_timer) / 30))
            elif level_up_notification_timer < 30:  # Last 0.5s - fade out
                alpha = int(255 * (level_up_notification_timer / 30))
            else:  # Middle - full opacity
                alpha = 255
            
            level_color = LEVELS[level_up_notification_level]["color"]
            
            # "LEVEL 2" text
            level_text = title_font.render(f"LEVEL {level_up_notification_level}", True, level_color)
            level_text.set_alpha(alpha)
            level_rect = level_text.get_rect(center=(WIDTH//2, HEIGHT//2))
            
            # Glow effect
            for i in range(3):
                glow = title_font.render(f"LEVEL {level_up_notification_level}", True, level_color)
                glow.set_alpha(max(0, alpha - 100 - i*30))
                glow_rect = glow.get_rect(center=(WIDTH//2 + i*2, HEIGHT//2 + i*2))
                screen.blit(glow, glow_rect)
            
            screen.blit(level_text, level_rect)
        
        # Level complete screen
        if game_over and win:
            draw_level_complete(current_level, score)
        
        pygame.display.flip()

except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    pygame.quit()