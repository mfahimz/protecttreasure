import os, sys, time, random, math, cv2, pygame
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

print("TREASURE GUARD - Multiplayer")

pygame.init()

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

pygame.display.set_caption("Treasure Guard - Multiplayer")

CAMERA_INDEX = auto_select_camera()

# Camera preview ALWAYS ON (arcade mode)
SHOW_CAMERA_DEBUG = True
DEBUG_WINDOW_MAX_WIDTH = 180
DEBUG_WINDOW_MAX_HEIGHT = 101

GAME_TIME = 60
MODEL_PATH_HAND = "models/hand_landmarker.task"

# Camera zoom settings - FIXED at 1.75x for maximum far-distance detection
CAMERA_ZOOM_ENABLED = True
CAMERA_ZOOM_AMOUNT = 0.18  # 18% crop per side = 1.75x zoom (was 0.15)

CHEST_IMAGE_PATH = "assets/chest.png"
THREAT_IMAGE_PATH = "assets/threat.png"
BACKGROUND_IMAGE_PATH = "assets/background.png"
BACKGROUND_MUSIC_PATH = "assets/background_music.mp3"
HIT_SOUND_PATH = "assets/hit_sound.wav"

TREASURE_SIZE, THREAT_SIZE = 90, 60
BASE_BORDER_RADIUS, BASE_GRAB_RADIUS = 70, 180
MAX_LIVES = 3
HIT_FLASH_DURATION = 0.4
SHAKE_INTENSITY = 12

P_GRAB_THRESH, P_DROP_THRESH = 1.3, 1.9
A_GRAB_THRESH, A_DROP_THRESH = 1.2, 1.75
RELEASE_BUFFER_MAX = 1
MOVE_SMOOTHING, THROW_SPEED = 0.20, 65.0
FLING_SPEED_TRIGGER, GRAB_LOCK_TIME = 90.0, 0.2

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

# Lower confidence for better far detection + faster tracking
options_hand = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH_HAND),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,  # Track exactly 2 hands
    min_hand_detection_confidence=0.3,  # Very low for fast movements
    min_tracking_confidence=0.3          # Very low for continuous tracking
)
landmarker_hand = vision.HandLandmarker.create_from_options(options_hand)

pygame.mixer.init()
clock = pygame.time.Clock()

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

treasure_pos = np.array([WIDTH // 4, HEIGHT // 2], dtype=float)
p_hand_smooth = np.array([WIDTH // 4, HEIGHT // 2], dtype=float)
p_velocity = np.array([0.0, 0.0], dtype=float)
a_hand_smooth = np.array([WIDTH * 0.75, HEIGHT // 2], dtype=float)
a_velocity = np.array([0.0, 0.0], dtype=float)

p_grip = a_grip = 1.0
p_tracking_lost = a_tracking_lost = True

# Hand identity tracking for stable detection
p_last_valid_pos = None
a_last_valid_pos = None
HAND_IDENTITY_THRESHOLD = 200  # Max pixels a hand can move between frames

held_threat_id = None
held_start_time = 0.0
a_release_counter = 0

threats = []
chest_state = "IDLE"
grab_start_time = None
game_over = False
win = False
lives = MAX_LIVES
hit_anim_timer = 0.0
score = 0
threats_dodged = 0

def get_grip_value(lm):
    wrist, mcp = lm[0], lm[9]
    hand_size = math.dist((wrist.x, wrist.y), (mcp.x, mcp.y))
    if hand_size < 0.01: return 1.0
    tips = [8, 12, 16, 20]
    avg_finger_dist = sum(math.dist((wrist.x, wrist.y), (lm[i].x, lm[i].y)) for i in tips) / 4.0
    return avg_finger_dist / hand_size

def is_valid_hand_size(lm):
    """Filter out background objects by hand size - relaxed for far distance"""
    try:
        wrist, mcp = lm[0], lm[9]
        hand_size = math.dist((wrist.x, wrist.y), (mcp.x, mcp.y))
        return 0.04 < hand_size < 0.45  # Even more lenient for fast/far hands
    except:
        return True

def assign_hands_with_tracking(detected_hands, p_last_pos, a_last_pos):
    """
    Intelligently assign detected hands to protector/attacker roles.
    Uses position history to maintain hand identity even during fast movements.
    """
    if not detected_hands:
        return None, None
    
    # Convert to screen coordinates with hand info
    hands_data = []
    for lm in detected_hands:
        wrist = lm[0]
        pos = np.array([wrist.x * WIDTH, wrist.y * HEIGHT], dtype=float)
        grip = get_grip_value(lm)
        
        # Validation checks
        if wrist.x < 0.02 or wrist.x > 0.98 or wrist.y < 0.02 or wrist.y > 0.98:
            continue  # Out of bounds
        if not is_valid_hand_size(lm):
            continue  # Invalid size
        if not (0.2 < grip < 4.5):  # Very wide grip range
            continue
        
        hands_data.append({
            'pos': pos,
            'grip': grip,
            'lm': lm,
            'x': pos[0]
        })
    
    if not hands_data:
        return None, None
    
    # Sort by X position (left to right)
    hands_data.sort(key=lambda h: h['x'])
    
    raw_p, raw_a = None, None
    p_grip_val, a_grip_val = 1.0, 1.0
    
    if len(hands_data) == 1:
        # Only one hand detected - use position history to assign
        hand = hands_data[0]
        
        if p_last_pos is not None and a_last_pos is not None:
            # Check which previous position is closer
            dist_to_p = np.linalg.norm(hand['pos'] - p_last_pos)
            dist_to_a = np.linalg.norm(hand['pos'] - a_last_pos)
            
            if dist_to_p < dist_to_a and dist_to_p < HAND_IDENTITY_THRESHOLD:
                raw_p = hand['pos']
                p_grip_val = hand['grip']
            elif dist_to_a < HAND_IDENTITY_THRESHOLD:
                raw_a = hand['pos']
                a_grip_val = hand['grip']
        else:
            # No history - assign by screen position
            if hand['x'] < WIDTH * 0.5:
                raw_p = hand['pos']
                p_grip_val = hand['grip']
            else:
                raw_a = hand['pos']
                a_grip_val = hand['grip']
    
    elif len(hands_data) == 2:
        # Two hands detected - use smart assignment
        left_hand = hands_data[0]
        right_hand = hands_data[1]
        
        # Use position history if available for validation
        if p_last_pos is not None and a_last_pos is not None:
            # Calculate distances to previous positions
            left_to_p = np.linalg.norm(left_hand['pos'] - p_last_pos)
            left_to_a = np.linalg.norm(left_hand['pos'] - a_last_pos)
            right_to_p = np.linalg.norm(right_hand['pos'] - p_last_pos)
            right_to_a = np.linalg.norm(right_hand['pos'] - a_last_pos)
            
            # Best match assignment
            if left_to_p < right_to_p and left_to_a > right_to_a:
                # Left matches P, Right matches A
                raw_p = left_hand['pos']
                p_grip_val = left_hand['grip']
                raw_a = right_hand['pos']
                a_grip_val = right_hand['grip']
            elif right_to_p < left_to_p and right_to_a > left_to_a:
                # Right matches P, Left matches A (hands crossed)
                raw_p = right_hand['pos']
                p_grip_val = right_hand['grip']
                raw_a = left_hand['pos']
                a_grip_val = left_hand['grip']
            else:
                # Ambiguous - use screen position
                raw_p = left_hand['pos']
                p_grip_val = left_hand['grip']
                raw_a = right_hand['pos']
                a_grip_val = right_hand['grip']
        else:
            # No history - simple left/right assignment
            raw_p = left_hand['pos']
            p_grip_val = left_hand['grip']
            raw_a = right_hand['pos']
            a_grip_val = right_hand['grip']
    
    else:
        # More than 2 hands detected (noise) - take leftmost and rightmost
        raw_p = hands_data[0]['pos']
        p_grip_val = hands_data[0]['grip']
        raw_a = hands_data[-1]['pos']
        a_grip_val = hands_data[-1]['grip']
    
    return (raw_p, p_grip_val) if raw_p is not None else None, \
           (raw_a, a_grip_val) if raw_a is not None else None

def apply_camera_zoom(frame):
    """Crop camera frame with fixed 1.5x zoom"""
    if not CAMERA_ZOOM_ENABLED:
        return frame
    
    h, w = frame.shape[:2]
    
    # Fixed 15% crop per side
    top = int(h * CAMERA_ZOOM_AMOUNT)
    bottom = int(h * (1 - CAMERA_ZOOM_AMOUNT))
    left = int(w * CAMERA_ZOOM_AMOUNT)
    right = int(w * (1 - CAMERA_ZOOM_AMOUNT))
    
    cropped = frame[top:bottom, left:right]
    zoomed = cv2.resize(cropped, (w, h))
    
    return zoomed

def spawn_threat():
    return {
        "id": time.time() + random.random(),
        "pos": np.array([random.randint(int(WIDTH * 0.6), WIDTH - 80),
                         random.randint(HEIGHT // 2 - 150, HEIGHT // 2 + 150)], dtype=float),
        "vel": np.array([0.0, 0.0]),
        "state": "IDLE",
        "angle": 0,
        "rotation_speed": random.uniform(-2, 2),
        "lifetime": 0.0
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
        ("❤️", "Lives Left", str(lives), DANGER if lives == 0 else SUCCESS)
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
    global treasure_pos, p_hand_smooth, p_velocity, a_hand_smooth, a_velocity
    global p_grip, a_grip, p_tracking_lost, a_tracking_lost, held_threat_id
    global held_start_time, a_release_counter, threats, chest_state
    global grab_start_time, game_over, win, lives, hit_anim_timer, score, threats_dodged
    global p_last_valid_pos, a_last_valid_pos
    
    treasure_pos = np.array([WIDTH // 4, HEIGHT // 2], dtype=float)
    p_hand_smooth = np.array([WIDTH // 4, HEIGHT // 2], dtype=float)
    p_velocity = np.array([0.0, 0.0], dtype=float)
    a_hand_smooth = np.array([WIDTH * 0.75, HEIGHT // 2], dtype=float)
    a_velocity = np.array([0.0, 0.0], dtype=float)
    
    p_grip = a_grip = 1.0
    p_tracking_lost = a_tracking_lost = True
    p_last_valid_pos = None
    a_last_valid_pos = None
    held_threat_id = None
    held_start_time = 0.0
    a_release_counter = 0
    
    threats = []
    chest_state = "IDLE"
    grab_start_time = None
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
        
        # Advanced 2-hand detection with identity tracking
        detected_hands = res_hand.hand_landmarks if (res_hand and res_hand.hand_landmarks) else []
        
        p_result, a_result = assign_hands_with_tracking(
            detected_hands,
            p_last_valid_pos,
            a_last_valid_pos
        )
        
        # Update raw positions and grip values
        if p_result:
            raw_p_pos, p_grip = p_result
            p_tracking_lost = False
            p_last_valid_pos = raw_p_pos.copy()
        else:
            raw_p_pos = None
            p_tracking_lost = True
        
        if a_result:
            raw_a_pos, a_grip = a_result
            a_tracking_lost = False
            a_last_valid_pos = raw_a_pos.copy()
        else:
            raw_a_pos = None
            a_tracking_lost = True
        
        # Update smoothed positions
        p_hand_smooth, p_velocity = update_hand_physics(raw_p_pos, p_hand_smooth, p_velocity)
        a_hand_smooth, a_velocity = update_hand_physics(raw_a_pos, a_hand_smooth, a_velocity)

        if scaled_background:
            screen.blit(scaled_background, (0, 0))
        else:
            screen.fill(DARK_BG)

        # Camera preview - ALWAYS ON
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
                elif event.key == pygame.K_SPACE and game_over:
                    reset_game()

        if not game_over:
            if grab_start_time and len(threats) < 6 and random.random() < 0.04:
                threats.append(spawn_threat())

            is_p_holding = (chest_state == "IDLE" and p_grip < P_GRAB_THRESH) or \
                          (chest_state == "GRABBED" and p_grip < P_DROP_THRESH)
            
            if chest_state == "IDLE" and is_p_holding and not p_tracking_lost:
                if np.linalg.norm(p_hand_smooth - treasure_pos) < BASE_GRAB_RADIUS:
                    chest_state = "GRABBED"
                    if grab_start_time is None:
                        grab_start_time = current_time
            elif chest_state == "GRABBED":
                if is_p_holding or p_tracking_lost:
                    treasure_pos += (p_hand_smooth - treasure_pos) * MOVE_SMOOTHING
                else:
                    chest_state = "IDLE"

            if held_threat_id is None and not a_tracking_lost and a_grip < A_GRAB_THRESH: 
                target = next((t for t in threats if t["state"] == "IDLE" and \
                              math.dist(a_hand_smooth, t["pos"]) < 150), None)
                if target: 
                    target["state"] = "HELD"
                    held_threat_id = target["id"]
                    held_start_time = current_time
                    a_release_counter = 0

            for i in range(len(threats) - 1, -1, -1):
                t = threats[i]
                t["lifetime"] += 1/60
                
                if t["state"] == "HELD" and t["id"] == held_threat_id:
                    t["pos"] = a_hand_smooth.copy()
                    if (current_time - held_start_time) > GRAB_LOCK_TIME:
                        release_signal = (a_grip > A_DROP_THRESH + 0.1) or \
                                       (np.linalg.norm(a_velocity) > FLING_SPEED_TRIGGER)
                        
                        if release_signal and not a_tracking_lost:
                            a_release_counter += 1
                            if a_release_counter >= RELEASE_BUFFER_MAX:
                                t["state"] = "FIRED"
                                t["vel"] = np.array([-1.0, 0.0]) * THROW_SPEED
                                held_threat_id = None
                        else:
                            a_release_counter = max(0, a_release_counter - 1)
                            
                elif t["state"] == "FIRED":
                    t["pos"] += t["vel"]
                    t["angle"] = math.degrees(math.atan2(-t["vel"][1], t["vel"][0]))
                    
                    if t["pos"][0] < -100:
                        threats.pop(i)
                        threats_dodged += 1
                        score += 10
                        if held_threat_id == t["id"]:
                            held_threat_id = None
                        continue
                
                if np.linalg.norm(t["pos"] - treasure_pos) < BASE_BORDER_RADIUS + 20:
                    lives -= 1
                    threats.pop(i)
                    if held_threat_id == t["id"]:
                        held_threat_id = None
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
            screen.blit(flash, (int(disp_pos[0] - BASE_BORDER_RADIUS * 2),
                               int(disp_pos[1] - BASE_BORDER_RADIUS * 2)))
            
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
        
        for t in threats:
            rot = pygame.transform.rotate(threat_img, t.get("angle", 0))
            screen.blit(rot, rot.get_rect(center=(int(t["pos"][0]), int(t["pos"][1]))).topleft)
        
        if not p_tracking_lost:
            pygame.draw.circle(screen, (0, 0, 0), p_hand_smooth.astype(int), 22)
            pygame.draw.circle(screen, CYAN, p_hand_smooth.astype(int), 20, 4)
            
        if not a_tracking_lost:
            hand_color = ORANGE if held_threat_id else MAGENTA
            pygame.draw.circle(screen, (0, 0, 0), a_hand_smooth.astype(int), 22)
            pygame.draw.circle(screen, hand_color, a_hand_smooth.astype(int), 20, 4)

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
            
            inst = font.render("Left hand protects • Right hand attacks", True, (180, 180, 200))
            inst_rect = inst.get_rect(center=(WIDTH//2, HEIGHT//2 + 30))
            screen.blit(inst, inst_rect)
            
        elif not game_over:
            rem = max(0, int(GAME_TIME - (current_time - grab_start_time)))
            
            stats_x = 40
            stats_y = 40
            
            hearts = "❤️ " * lives if lives > 0 else "💔 💔 💔"
            hearts_surf = big_font.render(hearts.strip(), True, DANGER if lives <= 1 else SUCCESS)
            screen.blit(hearts_surf, (stats_x, stats_y))
            
            timer_color = DANGER if rem <= 10 else WARNING if rem <= 30 else CYAN
            timer_surf = big_font.render(f"⏱️ {rem}s", True, timer_color)
            screen.blit(timer_surf, (stats_x, stats_y + 80))
            
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