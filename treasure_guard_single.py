# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================

import os, sys, time, random, math, cv2, pygame
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

print("TREASURE GUARD - Single Player")

pygame.init()

# ============================================================
# DISPLAY & CAMERA AUTO-DETECTION
# ============================================================

def auto_select_display():
    """Automatically select display - prefer extended screen if available"""
    num_displays = pygame.display.get_num_displays()
    if num_displays > 1:
        return 1
    return 0

def auto_select_camera():
    """Automatically detect best available camera on any OS"""
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

pygame.display.set_caption("Treasure Guard - Single Player")

CAMERA_INDEX = auto_select_camera()
SHOW_CAMERA_DEBUG = False
DEBUG_WINDOW_SIZE = (320, 180)
GAME_TIME = 30  # Changed from 60 to 30 seconds
MODEL_PATH_HAND = "models/hand_landmarker.task"

# Camera zoom settings
CAMERA_ZOOM_ENABLED = True
CAMERA_CROP_TOP = 0.05
CAMERA_CROP_BOTTOM = 0.05
CAMERA_CROP_LEFT = 0.05
CAMERA_CROP_RIGHT = 0.05

# ============================================================
# FILE PATHS
# ============================================================

CHEST_IMAGE_PATH = "assets/chest.png"
THREAT_IMAGE_PATH = "assets/threat.png"
GRENADE_IMAGE_PATH = "assets/grenade.png"  # New grenade threat
BACKGROUND_IMAGE_PATH = "assets/background.png"
BACKGROUND_MUSIC_PATH = "assets/background_music.mp3"
HIT_SOUND_PATH = "assets/hit_sound.wav"

# ============================================================
# GAME CONFIGURATION
# ============================================================

TREASURE_SIZE, THREAT_SIZE = 90, 60
GRENADE_SIZE = 70  # Larger than regular threats
BASE_BORDER_RADIUS, BASE_GRAB_RADIUS = 70, 180
MAX_LIVES = 3.0  # Using float for half-lives (0.5 damage per hit)
HIT_FLASH_DURATION = 0.4
SHAKE_INTENSITY = 12

GRENADE_INTERVAL = 8.0  # Grenade spawns every 8 seconds

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
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
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
# LOAD ASSETS
# ============================================================

def load_sound(path):
    try:
        return pygame.mixer.Sound(path)
    except:
        return None

hit_sound = load_sound(HIT_SOUND_PATH)

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
last_grenade_time = None  # Track when last grenade was spawned
game_over = False
win = False
lives = MAX_LIVES
hit_anim_timer = 0.0
score = 0
threats_dodged = 0

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_grip_value(lm):
    wrist, mcp = lm[0], lm[9]
    hand_size = math.dist((wrist.x, wrist.y), (mcp.x, mcp.y))
    if hand_size < 0.01: return 1.0
    tips = [8, 12, 16, 20]
    avg_finger_dist = sum(math.dist((wrist.x, wrist.y), (lm[i].x, lm[i].y)) for i in tips) / 4.0
    return avg_finger_dist / hand_size

def is_valid_hand_size(lm):
    """Filter out background objects by hand size"""
    try:
        wrist, mcp = lm[0], lm[9]
        hand_size = math.dist((wrist.x, wrist.y), (mcp.x, mcp.y))
        return 0.07 < hand_size < 0.35
    except:
        return True

def apply_camera_zoom(frame):
    """Crop camera frame to focus on play area"""
    if not CAMERA_ZOOM_ENABLED:
        return frame
    
    h, w = frame.shape[:2]
    top = int(h * CAMERA_CROP_TOP)
    bottom = int(h * (1 - CAMERA_CROP_BOTTOM))
    left = int(w * CAMERA_CROP_LEFT)
    right = int(w * (1 - CAMERA_CROP_RIGHT))
    
    cropped = frame[top:bottom, left:right]
    zoomed = cv2.resize(cropped, (w, h))
    
    return zoomed

def spawn_threat(is_grenade=False, chest_position=None):
    """Spawn threat from random direction - semi-targeted toward chest"""
    # Choose random side: 0=top, 1=right, 2=bottom, 3=left
    side = random.randint(0, 3)
    
    if side == 0:  # Top
        start_x = random.randint(100, WIDTH - 100)
        start_y = -100
    elif side == 1:  # Right
        start_x = WIDTH + 100
        start_y = random.randint(100, HEIGHT - 100)
    elif side == 2:  # Bottom
        start_x = random.randint(100, WIDTH - 100)
        start_y = HEIGHT + 100
    else:  # Left
        start_x = -100
        start_y = random.randint(100, HEIGHT - 100)
    
    start_pos = np.array([start_x, start_y], dtype=float)
    
    # Calculate direction toward chest (if provided)
    if chest_position is not None:
        # Vector from spawn to chest
        to_chest = chest_position - start_pos
        to_chest_normalized = to_chest / np.linalg.norm(to_chest)
        
        # Add randomness: 60-80% toward chest, rest is random
        # This makes it challenging but dodgeable
        if is_grenade:
            accuracy = random.uniform(0.7, 0.85)  # Grenades more accurate (70-85%)
        else:
            accuracy = random.uniform(0.5, 0.75)  # Regular less accurate (50-75%)
        
        # Random perpendicular offset
        perpendicular = np.array([-to_chest_normalized[1], to_chest_normalized[0]])
        random_offset = perpendicular * random.uniform(-0.8, 0.8)
        
        # Combine targeted direction with random offset
        direction = to_chest_normalized * accuracy + random_offset * (1 - accuracy)
        direction = direction / np.linalg.norm(direction)  # Normalize
    else:
        # Fallback: aim toward center
        to_center = np.array([WIDTH // 2, HEIGHT // 2], dtype=float) - start_pos
        direction = to_center / np.linalg.norm(to_center)
    
    # Speed: grenades slower but deadly, regular threats balanced
    if is_grenade:
        speed = random.uniform(2.5, 3.5)  # Slower - more time to see it
    else:
        speed = random.uniform(3.5, 5.0)  # Balanced speed
    
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

def draw_game_over_screen(won, final_score, time_survived, dodged):
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(220)
    overlay.fill(DARK_BG)
    screen.blit(overlay, (0, 0))
    
    if won:
        result_text = title_font.render("VICTORY!", True, GOLD)
        for i in range(3):
            glow = title_font.render("VICTORY!", True, (255, 215, 0, 100 - i*30))
            glow_rect = glow.get_rect(center=(WIDTH//2, HEIGHT//4 + i*3))
            screen.blit(glow, glow_rect)
        
        result_rect = result_text.get_rect(center=(WIDTH//2, HEIGHT//4))
        screen.blit(result_text, result_rect)
        subtitle = medium_font.render("🏆 Treasure Protected! 🏆", True, GREEN)
    else:
        result_text = title_font.render("GAME OVER", True, DANGER)
        result_rect = result_text.get_rect(center=(WIDTH//2, HEIGHT//4))
        screen.blit(result_text, result_rect)
        subtitle = medium_font.render("💔 Treasure Lost 💔", True, RED)
    
    subtitle_rect = subtitle.get_rect(center=(WIDTH//2, HEIGHT//4 + 100))
    screen.blit(subtitle, subtitle_rect)
    
    stats_start_y = HEIGHT//2 - 50
    panel_width = 600
    panel_x = (WIDTH - panel_width) // 2
    
    draw_ui_panel(screen, panel_x - 20, stats_start_y - 20, panel_width + 40, 320, alpha=230)
    
    stats = [
        ("⏱️", "Time Survived", f"{int(time_survived)}s / {GAME_TIME}s", CYAN),
        ("🎯", "Threats Dodged", str(dodged), GREEN),
        ("⭐", "Final Score", str(final_score), GOLD),
        ("❤️", "Lives Left", f"{lives:.1f}" if lives > 0 else "0.0", DANGER if lives == 0 else SUCCESS)
    ]
    
    for i, (icon, label, value, color) in enumerate(stats):
        y_pos = stats_start_y + i * 75
        pygame.draw.circle(screen, color, (panel_x + 40, y_pos + 25), 25, 3)
        icon_surf = medium_font.render(icon, True, color)
        icon_rect = icon_surf.get_rect(center=(panel_x + 40, y_pos + 25))
        screen.blit(icon_surf, icon_rect)
        
        label_surf = small_font.render(label, True, (180, 180, 200))
        screen.blit(label_surf, (panel_x + 85, y_pos + 8))
        
        value_surf = medium_font.render(value, True, color)
        screen.blit(value_surf, (panel_x + 85, y_pos + 30))
    
    pulse = abs(math.sin(time.time() * 2)) * 0.3 + 0.7
    restart_color = tuple(int(c * pulse) for c in UI_ACCENT)
    
    draw_text_with_shadow(screen, "Press SPACE to Play Again", font, restart_color, WIDTH//2 - 200, HEIGHT - 120)
    draw_text_with_shadow(screen, "Press ESC to Exit", small_font, (150, 150, 170), WIDTH//2 - 100, HEIGHT - 70)

def reset_game():
    global treasure_pos, hand_smooth, hand_velocity, hand_grip, hand_tracking_lost
    global threats, chest_state, grab_start_time, game_over, win, lives, hit_anim_timer, score, threats_dodged
    global last_grenade_time
    
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

def toggle_fullscreen():
    global WIDTH, HEIGHT, USE_FULLSCREEN, screen, scaled_background
    USE_FULLSCREEN = not USE_FULLSCREEN
    if USE_FULLSCREEN:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        WIDTH, HEIGHT = screen.get_size()
    else:
        screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
        WIDTH, HEIGHT = 1920, 1080
    scaled_background = get_scaled_background()

def toggle_camera_debug():
    global SHOW_CAMERA_DEBUG
    SHOW_CAMERA_DEBUG = not SHOW_CAMERA_DEBUG

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
            
            if wrist.x < 0.1 or wrist.x > 0.9 or wrist.y < 0.1 or wrist.y > 0.9:
                pass
            elif is_valid_hand_size(lm):
                px = np.array([wrist.x * WIDTH, wrist.y * HEIGHT], dtype=float)
                grip = get_grip_value(lm)
                
                if 0.5 < grip < 3.0:
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
        
        # Camera debug window
        if SHOW_CAMERA_DEBUG and ret:
            try:
                debug_frame = cv2.resize(frame, DEBUG_WINDOW_SIZE)
                debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
                debug_frame = np.rot90(debug_frame)
                debug_frame = np.flipud(debug_frame)
                debug_surface = pygame.surfarray.make_surface(debug_frame)
                
                debug_x = WIDTH - DEBUG_WINDOW_SIZE[0] - 30
                debug_y = 30
                
                pygame.draw.rect(screen, (0, 0, 0), (debug_x - 5, debug_y - 5, DEBUG_WINDOW_SIZE[0] + 10, DEBUG_WINDOW_SIZE[1] + 10), border_radius=10)
                pygame.draw.rect(screen, UI_ACCENT, (debug_x - 4, debug_y - 4, DEBUG_WINDOW_SIZE[0] + 8, DEBUG_WINDOW_SIZE[1] + 8), 3, border_radius=10)
                screen.blit(debug_surface, (debug_x, debug_y))
                
                status_bar_y = debug_y + DEBUG_WINDOW_SIZE[1] + 10
                draw_ui_panel(screen, debug_x - 5, status_bar_y, DEBUG_WINDOW_SIZE[0] + 10, 35, alpha=240)
                
                cam_dot = small_font.render("●", True, SUCCESS)
                screen.blit(cam_dot, (debug_x + 5, status_bar_y + 5))
                
                cam_text = f"CAM {CAMERA_INDEX}"
                cam_label = small_font.render(cam_text, True, WHITE)
                screen.blit(cam_label, (debug_x + 25, status_bar_y + 8))
                
                fps_text = f"{int(clock.get_fps())} FPS"
                fps_label = small_font.render(fps_text, True, (150, 150, 170))
                screen.blit(fps_label, (debug_x + DEBUG_WINDOW_SIZE[0] - 70, status_bar_y + 8))
            except:
                pass
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and game_over:
                    reset_game()
                elif event.key == pygame.K_f:
                    toggle_fullscreen()
                elif event.key == pygame.K_c:
                    toggle_camera_debug()
        
        if not game_over:
            # Spawn regular threats - pass chest position for semi-targeting
            if grab_start_time and len(threats) < 6 and random.random() < 0.04:
                threats.append(spawn_threat(is_grenade=False, chest_position=treasure_pos))
            
            # Spawn grenade every 8 seconds
            if grab_start_time:
                elapsed_time = current_time - grab_start_time
                if last_grenade_time is None:
                    last_grenade_time = grab_start_time
                
                if (current_time - last_grenade_time) >= GRENADE_INTERVAL:
                    threats.append(spawn_threat(is_grenade=True, chest_position=treasure_pos))
                    last_grenade_time = current_time
            
            is_holding = (chest_state == "IDLE" and hand_grip < P_GRAB_THRESH) or \
                        (chest_state == "GRABBED" and hand_grip < P_DROP_THRESH)
            
            if chest_state == "IDLE" and is_holding and not hand_tracking_lost:
                if np.linalg.norm(hand_smooth - treasure_pos) < BASE_GRAB_RADIUS:
                    chest_state = "GRABBED"
                    if grab_start_time is None:
                        grab_start_time = current_time
            elif chest_state == "GRABBED":
                if is_holding or hand_tracking_lost:
                    treasure_pos += (hand_smooth - treasure_pos) * MOVE_SMOOTHING
                else:
                    chest_state = "IDLE"
            
            for i in range(len(threats) - 1, -1, -1):
                t = threats[i]
                t["lifetime"] += 1/60
                t["pos"] += t["vel"]
                t["angle"] += t["rotation_speed"]
                
                # Remove if off screen
                if (t["pos"][0] < -200 or t["pos"][0] > WIDTH + 200 or 
                    t["pos"][1] < -200 or t["pos"][1] > HEIGHT + 200):
                    threats.pop(i)
                    if t["type"] == "regular":
                        threats_dodged += 1
                        score += 10
                    else:
                        threats_dodged += 1
                        score += 25  # More points for dodging grenades
                    continue
                
                # Check collision with chest
                collision_radius = BASE_BORDER_RADIUS + (35 if t["type"] == "grenade" else 20)
                if np.linalg.norm(t["pos"] - treasure_pos) < collision_radius:
                    # Grenade = instant death, Regular = 0.5 damage
                    if t["type"] == "grenade":
                        lives = 0  # All lives lost
                        threats.pop(i)
                        hit_anim_timer = current_time
                        if hit_sound:
                            try:
                                hit_sound.play()
                            except:
                                pass
                        game_over = True
                        win = False
                    else:
                        lives -= 0.5  # Half a life
                        threats.pop(i)
                        hit_anim_timer = current_time
                        if hit_sound:
                            try:
                                hit_sound.play()
                            except:
                                pass
                        if lives <= 0:
                            game_over = True
                            win = False
            
            if grab_start_time and (current_time - grab_start_time) >= GAME_TIME:
                game_over = True
                win = True
        
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
        
        # Draw threats with different visuals for grenades
        for t in threats:
            if t["type"] == "grenade":
                # Grenade with pulsing red glow
                pulse_size = int(5 + abs(math.sin(current_time * 10)) * 10)
                glow_surf = pygame.Surface((GRENADE_SIZE + pulse_size*2, GRENADE_SIZE + pulse_size*2))
                glow_surf.set_alpha(150)
                pygame.draw.circle(glow_surf, DANGER, (GRENADE_SIZE//2 + pulse_size, GRENADE_SIZE//2 + pulse_size), 
                                 GRENADE_SIZE//2 + pulse_size)
                screen.blit(glow_surf, (int(t["pos"][0] - GRENADE_SIZE//2 - pulse_size),
                                       int(t["pos"][1] - GRENADE_SIZE//2 - pulse_size)))
                
                rot = pygame.transform.rotate(grenade_img, t.get("angle", 0))
                screen.blit(rot, rot.get_rect(center=(int(t["pos"][0]), int(t["pos"][1]))).topleft)
            else:
                # Regular threat
                rot = pygame.transform.rotate(threat_img, t.get("angle", 0))
                screen.blit(rot, rot.get_rect(center=(int(t["pos"][0]), int(t["pos"][1]))).topleft)
        
        # Hand cursor
        if not hand_tracking_lost:
            pygame.draw.circle(screen, (0, 0, 0), hand_smooth.astype(int), 22)
            pygame.draw.circle(screen, CYAN, hand_smooth.astype(int), 20, 4)
        
        if not grab_start_time and not game_over:
            pulse = abs(math.sin(current_time * 2)) * 0.3 + 0.7
            pulse_color = tuple(int(c * pulse) for c in GREEN)
            
            instr = big_font.render("GRAB THE CHEST TO START", True, pulse_color)
            instr_rect = instr.get_rect(center=(WIDTH//2, HEIGHT//2 - 50))
            
            shadow = big_font.render("GRAB THE CHEST TO START", True, (0, 0, 0))
            shadow.set_alpha(150)
            shadow_rect = shadow.get_rect(center=(WIDTH//2 + 3, HEIGHT//2 - 47))
            screen.blit(shadow, shadow_rect)
            screen.blit(instr, instr_rect)
            
            inst = font.render("Defend the chest from incoming threats!", True, (180, 180, 200))
            inst_rect = inst.get_rect(center=(WIDTH//2, HEIGHT//2 + 30))
            screen.blit(inst, inst_rect)
            
        elif not game_over:
            rem = max(0, int(GAME_TIME - (current_time - grab_start_time)))
            
            stats_x = 40
            stats_y = 40
            
            # Display lives with half-hearts
            full_hearts = int(lives)
            has_half = (lives % 1) >= 0.5
            
            hearts = "❤️ " * full_hearts
            if has_half:
                hearts += "💔 "
            
            if lives <= 0:
                hearts = "💔 💔 💔"
            
            hearts_surf = big_font.render(hearts.strip(), True, DANGER if lives <= 1 else SUCCESS)
            screen.blit(hearts_surf, (stats_x, stats_y))
            
            # Timer
            timer_color = DANGER if rem <= 10 else WARNING if rem <= 15 else CYAN
            timer_surf = big_font.render(f"⏱️ {rem}s", True, timer_color)
            screen.blit(timer_surf, (stats_x, stats_y + 80))
            
            # Score
            score_surf = big_font.render(f"⭐ {score}", True, GOLD)
            screen.blit(score_surf, (stats_x, stats_y + 160))
        
        if game_over:
            time_survived = current_time - grab_start_time if grab_start_time else 0
            draw_game_over_screen(win, score, time_survived, threats_dodged)
        
        pygame.display.flip()

except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    pygame.quit()