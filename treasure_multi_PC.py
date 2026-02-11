import os, sys, time, random, math, cv2, pygame
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

os.environ["SDL_VIDEO_CENTERED"] = "1"

print("=" * 70)
print("               TREASURE GUARD - Enhanced Edition")
print("=" * 70)
print("\n[INFO] AUTOMATIC SETUP - No user input required")
print("[INFO] System will auto-detect and configure:")
print("       • Best available display (prefers extended screens)")
print("       • Best camera (prefers external USB cameras)")
print("       • Optimal resolution for detected display")
print("\n[INFO] GAMEPLAY CONTROLS:")
print("       • Left hand (Protector): Close fist to grab treasure")
print("       • Right hand (Attacker): Grab and throw threats")
print("\n[INFO] KEYBOARD SHORTCUTS:")
print("       • F - Toggle fullscreen")
print("       • C - Toggle camera debug view")
print("       • ESC - Quit game")
print("       • SPACE - Restart (after game over)")
print("\n[INFO] Camera debug feed will appear in top-right corner")
print("=" * 70)

pygame.init()

def auto_select_display():
    num_displays = pygame.display.get_num_displays()
    print(f"\n[DEBUG] Display Detection:")
    print(f"[DEBUG] Found {num_displays} display(s)")
    
    if num_displays > 1:
        for i in range(num_displays):
            try:
                bounds = pygame.display.get_desktop_sizes()[i]
                print(f"[DEBUG] Display {i}: {bounds[0]}x{bounds[1]}")
            except:
                print(f"[DEBUG] Display {i}: Available (resolution unknown)")
        
        selected = 1
        print(f"[INFO] Multiple displays detected - Auto-selecting Display {selected} (extended screen)")
    else:
        info = pygame.display.Info()
        print(f"[DEBUG] Display 0: {info.current_w}x{info.current_h}")
        print(f"[INFO] Single display detected - Using Display 0")
        selected = 0
    
    return selected

def auto_select_camera():
    print(f"\n[DEBUG] Camera Detection:")
    available_cameras = []
    
    for i in range(4):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                resolution = w * h
                available_cameras.append((i, w, h, resolution))
                print(f"[DEBUG] Camera {i}: {w}x{h} ({resolution} pixels)")
            else:
                print(f"[DEBUG] Camera {i}: Opened but no signal")
            cap.release()
    
    if not available_cameras:
        print("[WARNING] No cameras detected - Defaulting to Camera 0")
        print("[WARNING] If this fails, check USB connections and camera permissions")
        return 0
    
    available_cameras.sort(key=lambda x: x[3], reverse=True)
    selected = available_cameras[0][0]
    selected_res = f"{available_cameras[0][1]}x{available_cameras[0][2]}"
    
    if len(available_cameras) > 1:
        best_res = available_cameras[0][3]
        for cam in available_cameras[1:]:
            if cam[0] > 0 and cam[3] >= (best_res * 0.8):
                selected = cam[0]
                selected_res = f"{cam[1]}x{cam[2]}"
                print(f"[INFO] External USB camera detected - Preferring Camera {selected}")
                break
    
    print(f"[INFO] Auto-selected Camera {selected} ({selected_res})")
    return selected

USE_FULLSCREEN = True
DISPLAY_INDEX = auto_select_display()

try:
    if DISPLAY_INDEX < len(pygame.display.get_desktop_sizes()):
        WIDTH, HEIGHT = pygame.display.get_desktop_sizes()[DISPLAY_INDEX]
        print(f"[DEBUG] Using native resolution: {WIDTH}x{HEIGHT}")
    else:
        WIDTH, HEIGHT = 1920, 1080
        print(f"[DEBUG] Defaulting to Full HD: {WIDTH}x{HEIGHT}")
except:
    WIDTH, HEIGHT = 1920, 1080
    print(f"[DEBUG] Error detecting resolution - Using Full HD: {WIDTH}x{HEIGHT}")

print(f"\n[DEBUG] Window Initialization:")
if USE_FULLSCREEN:
    try:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=DISPLAY_INDEX)
        WIDTH, HEIGHT = screen.get_size()
        print(f"[INFO] ✓ Fullscreen mode: {WIDTH}x{HEIGHT} on Display {DISPLAY_INDEX}")
    except Exception as e:
        print(f"[WARNING] Fullscreen failed: {e}")
        print(f"[INFO] Falling back to windowed mode...")
        screen = pygame.display.set_mode((1920, 1080))
        WIDTH, HEIGHT = 1920, 1080
        USE_FULLSCREEN = False
else:
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    print(f"[INFO] ✓ Windowed mode: {WIDTH}x{HEIGHT}")

pygame.display.set_caption("Treasure Guard - Enhanced Edition")

CAMERA_INDEX = auto_select_camera()
SHOW_CAMERA_DEBUG = True
DEBUG_WINDOW_SIZE = (320, 180)
GAME_TIME = 60
MODEL_PATH_HAND = "models/hand_landmarker.task"

CHEST_IMAGE_PATH = r"C:\Users\pc\protecttreasure\CHEST_IMAGE_PATH.png"
THREAT_IMAGE_PATH = r"C:\Users\pc\protecttreasure\THREAT_IMAGE_PATH.png"
BACKGROUND_IMAGE_PATH = r"C:\Users\pc\protecttreasure\BACKGROUND_IMAGE_PATH.png"
BACKGROUND_MUSIC_PATH = r"C:\Users\pc\protecttreasure\BACKGROUND_MUSIC_PATH.mp3"
HIT_SOUND_PATH = r"C:\Users\pc\protecttreasure\HIT_SOUND_PATH.wav"

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
PURPLE = (167, 139, 250)
ORANGE = (255, 154, 88)
BLUE_GLOW = (59, 130, 246)
UI_BG = (30, 30, 45)
UI_BORDER = (100, 100, 130)
UI_ACCENT = (94, 234, 212)
SUCCESS = (34, 197, 94)
DANGER = (239, 68, 68)
WARNING = (245, 158, 11)

options_hand = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH_HAND),
    running_mode=vision.RunningMode.VIDEO, num_hands=4,
    min_hand_detection_confidence=0.5, min_tracking_confidence=0.5
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
except Exception as e:
    print(f"[WARNING] Could not load custom fonts, using defaults: {e}")
    title_font = pygame.font.Font(None, 150)
    big_font = pygame.font.Font(None, 100)
    medium_font = pygame.font.Font(None, 60)
    font = pygame.font.SysFont(None, 35)
    small_font = pygame.font.Font(None, 28)

def load_sound(path):
    try:
        return pygame.mixer.Sound(path)
    except Exception as e:
        print(f"[WARNING] Could not load sound {path}: {e}")
        return None

hit_sound = load_sound(HIT_SOUND_PATH)

def load_scale(path, size, fallback_color):
    try:
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.smoothscale(img, size)
    except Exception as e:
        print(f"[WARNING] Could not load image {path}: {e}")
        surf = pygame.Surface(size)
        surf.fill(fallback_color)
        return surf

chest_img = load_scale(CHEST_IMAGE_PATH, (TREASURE_SIZE, TREASURE_SIZE), YELLOW)
threat_img = load_scale(THREAT_IMAGE_PATH, (THREAT_SIZE, THREAT_SIZE), RED)

background_img = None
try:
    bg_temp = pygame.image.load(BACKGROUND_IMAGE_PATH).convert()
    background_img = bg_temp
    print(f"[DEBUG] Background image loaded successfully")
except Exception as e:
    print(f"[WARNING] Could not load background image: {e}")
    print(f"[INFO] Game will use solid color background instead")

try:
    pygame.mixer.music.load(BACKGROUND_MUSIC_PATH)
    pygame.mixer.music.set_volume(0.3)
    pygame.mixer.music.play(-1)
except Exception as e:
    print(f"[WARNING] Could not load background music: {e}")

print(f"\n[DEBUG] Camera Initialization:")
print(f"[INFO] Attempting to open Camera {CAMERA_INDEX}...")

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"[WARNING] Camera {CAMERA_INDEX} failed to open with CAP_DSHOW")
    print(f"[INFO] Trying Camera {CAMERA_INDEX} without CAP_DSHOW...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"[WARNING] Camera {CAMERA_INDEX} still failed to open")
    print(f"[INFO] Falling back to default Camera 0...")
    cap = cv2.VideoCapture(0)
    CAMERA_INDEX = 0

if cap.isOpened():
    ret, test_frame = cap.read()
    if ret:
        h, w = test_frame.shape[:2]
        print(f"[INFO] ✓ Camera {CAMERA_INDEX} opened successfully ({w}x{h})")
    else:
        print(f"[WARNING] Camera {CAMERA_INDEX} opened but no video signal")
else:
    print("[ERROR] ❌ No camera available! Game cannot start.")
    print("[ERROR] Please check:")
    print("[ERROR]   1. Camera is connected via USB")
    print("[ERROR]   2. Camera permissions enabled in Windows Settings")
    print("[ERROR]   3. No other app is using the camera")
    pygame.quit()
    sys.exit(1)

start_time_ref = time.time()
print(f"\n[INFO] ═══════════════════════════════════════════════════")
print(f"[INFO] Game initialization complete - Starting game loop...")
print(f"[INFO] ═══════════════════════════════════════════════════")
print(f"[INFO] ")
print(f"[INFO] WHAT YOU SHOULD SEE:")
print(f"[INFO] • Main game window at {WIDTH}x{HEIGHT}")
if background_img:
    print(f"[INFO] • Background image loaded and scaled")
else:
    print(f"[INFO] • Dark blue background (no image file)")
print(f"[INFO] • Camera feed in TOP-RIGHT corner")
print(f"[INFO] • Yellow chest in center (grab to start)")
print(f"[INFO] • Press C to toggle camera view ON/OFF")
print(f"[INFO] ")
print(f"[INFO] ═══════════════════════════════════════════════════\n")

treasure_pos = np.array([WIDTH // 4, HEIGHT // 2], dtype=float)
p_hand_smooth = np.array([WIDTH // 4, HEIGHT // 2], dtype=float)
p_velocity = np.array([0.0, 0.0], dtype=float)
a_hand_smooth = np.array([WIDTH * 0.75, HEIGHT // 2], dtype=float)
a_velocity = np.array([0.0, 0.0], dtype=float)

p_grip = a_grip = 1.0
p_tracking_lost = a_tracking_lost = True
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
            smooth_factor = 0.30
        elif distance > 20:
            smooth_factor = 0.22
        else:
            smooth_factor = 0.15
        
        new_velocity = raw_pos - current_smooth
        smooth = (raw_pos * smooth_factor) + (current_smooth * (1 - smooth_factor))
        vel = (new_velocity * 0.4) + (current_vel * 0.6)
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
    
    draw_ui_panel(screen, panel_x - 20, stats_start_y - 20,
                  panel_width + 40, 320, alpha=230)
    
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
    
    draw_text_with_shadow(screen, "Press SPACE to Play Again",
                          font, restart_color,
                          WIDTH//2 - 200, HEIGHT - 120)
    
    draw_text_with_shadow(screen, "Press ESC to Exit",
                          small_font, (150, 150, 170),
                          WIDTH//2 - 100, HEIGHT - 70)

def reset_game():
    global treasure_pos, p_hand_smooth, p_velocity, a_hand_smooth, a_velocity
    global p_grip, a_grip, p_tracking_lost, a_tracking_lost, held_threat_id
    global held_start_time, a_release_counter, threats, chest_state
    global grab_start_time, game_over, win, lives, hit_anim_timer, score, threats_dodged
    
    treasure_pos = np.array([WIDTH // 4, HEIGHT // 2], dtype=float)
    p_hand_smooth = np.array([WIDTH // 4, HEIGHT // 2], dtype=float)
    p_velocity = np.array([0.0, 0.0], dtype=float)
    a_hand_smooth = np.array([WIDTH * 0.75, HEIGHT // 2], dtype=float)
    a_velocity = np.array([0.0, 0.0], dtype=float)
    
    p_grip = a_grip = 1.0
    p_tracking_lost = a_tracking_lost = True
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

def draw_stat_display(surface, icon, label, value, x, y, color=WHITE):
    panel_width = 280
    panel_height = 60
    draw_ui_panel(surface, x, y, panel_width, panel_height, alpha=180)
    
    icon_text = big_font.render(icon, True, color)
    surface.blit(icon_text, (x + 15, y + 5))
    
    label_surface = small_font.render(label, True, (200, 200, 220))
    surface.blit(label_surface, (x + 80, y + 10))
    
    value_surface = medium_font.render(str(value), True, color)
    surface.blit(value_surface, (x + 80, y + 28))

def draw_pulse_circle(surface, center, radius, color, pulse_time):
    pulse = abs(math.sin(pulse_time * 3)) * 0.3 + 0.7
    actual_radius = int(radius * pulse)
    pygame.draw.circle(surface, color, center, actual_radius, 4)

def get_scaled_background():
    global background_img
    if background_img is not None:
        try:
            return pygame.transform.smoothscale(background_img, (WIDTH, HEIGHT))
        except Exception as e:
            print(f"[WARNING] Failed to scale background: {e}")
            return None
    return None

frame_count = 0
last_debug_time = time.time()
scaled_background = get_scaled_background()

def toggle_fullscreen():
    global WIDTH, HEIGHT, USE_FULLSCREEN, screen, scaled_background
    USE_FULLSCREEN = not USE_FULLSCREEN
    if USE_FULLSCREEN:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        WIDTH, HEIGHT = screen.get_size()
        print(f"[INFO] Fullscreen enabled: {WIDTH}x{HEIGHT}")
    else:
        screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
        WIDTH, HEIGHT = 1920, 1080
        print(f"[INFO] Windowed mode enabled: {WIDTH}x{HEIGHT}")
    scaled_background = get_scaled_background()

def toggle_camera_debug():
    global SHOW_CAMERA_DEBUG
    SHOW_CAMERA_DEBUG = not SHOW_CAMERA_DEBUG
    print(f"[INFO] Camera debug view: {'ON' if SHOW_CAMERA_DEBUG else 'OFF'}")

try:
    running = True
    while running:
        clock.tick(60)
        current_time = time.time()
        frame_count += 1
        
        if current_time - last_debug_time >= 5.0:
            print(f"[DEBUG] Game running - Frame: {frame_count}, FPS: {clock.get_fps():.1f}, "
                  f"Lives: {lives}, Score: {score}, Threats: {len(threats)}")
            last_debug_time = current_time
        
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] Camera frame read failed at frame {frame_count}")
            if frame_count % 30 == 0:
                print(f"[INFO] Attempting to reconnect camera...")
                cap.release()
                cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            continue
            
        frame = cv2.flip(frame, 1)
        
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, ac, bc = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
            l = clahe.apply(l)
            limg = cv2.merge((l, ac, bc))
            frame_proc = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception as e:
            print(f"[WARNING] Frame processing failed: {e}")
            frame_proc = frame
        
        try:
            rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res_hand = landmarker_hand.detect_for_video(mp_image, int((current_time - start_time_ref) * 1000))
        except Exception as e:
            print(f"[WARNING] Hand detection failed: {e}")
            res_hand = None
        
        raw_p_pos, raw_a_pos = None, None
        if res_hand and res_hand.hand_landmarks:
            for lm in res_hand.hand_landmarks:
                wrist = lm[0]
                px = np.array([wrist.x * WIDTH, wrist.y * HEIGHT], dtype=float)
                grip = get_grip_value(lm)
                if px[0] < WIDTH * 0.48:
                    raw_p_pos, p_grip, p_tracking_lost = px, grip, False
                elif px[0] > WIDTH * 0.52:
                    raw_a_pos, a_grip, a_tracking_lost = px, grip, False
        
        if raw_p_pos is None: p_tracking_lost = True
        if raw_a_pos is None: a_tracking_lost = True
        p_hand_smooth, p_velocity = update_hand_physics(raw_p_pos, p_hand_smooth, p_velocity)
        a_hand_smooth, a_velocity = update_hand_physics(raw_a_pos, a_hand_smooth, a_velocity)

        if scaled_background:
            screen.blit(scaled_background, (0, 0))
        else:
            screen.fill(DARK_BG)

        if SHOW_CAMERA_DEBUG and ret:
            try:
                debug_frame = cv2.resize(frame, DEBUG_WINDOW_SIZE)
                debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
                debug_frame = np.rot90(debug_frame)
                debug_frame = np.flipud(debug_frame)
                debug_surface = pygame.surfarray.make_surface(debug_frame)
                
                debug_x = WIDTH - DEBUG_WINDOW_SIZE[0] - 30
                debug_y = 30
                
                border_color = UI_ACCENT
                pygame.draw.rect(screen, (0, 0, 0),
                               (debug_x - 5, debug_y - 5, DEBUG_WINDOW_SIZE[0] + 10, DEBUG_WINDOW_SIZE[1] + 10),
                               border_radius=10)
                pygame.draw.rect(screen, border_color,
                               (debug_x - 4, debug_y - 4, DEBUG_WINDOW_SIZE[0] + 8, DEBUG_WINDOW_SIZE[1] + 8),
                               3, border_radius=10)
                
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
                
                if res_hand and res_hand.hand_landmarks:
                    for lm in res_hand.hand_landmarks:
                        wrist = lm[0]
                        px = int(wrist.x * DEBUG_WINDOW_SIZE[0]) + debug_x
                        py = int(wrist.y * DEBUG_WINDOW_SIZE[1]) + debug_y
                        
                        if wrist.x < 0.48:
                            pygame.draw.circle(screen, (0, 0, 0), (px, py), 12)
                            pygame.draw.circle(screen, CYAN, (px, py), 10, 3)
                            badge = small_font.render("L", True, WHITE)
                            screen.blit(badge, (px - 5, py - 8))
                        elif wrist.x > 0.52:
                            pygame.draw.circle(screen, (0, 0, 0), (px, py), 12)
                            pygame.draw.circle(screen, MAGENTA, (px, py), 10, 3)
                            badge = small_font.render("R", True, WHITE)
                            screen.blit(badge, (px - 5, py - 8))
            except Exception as e:
                print(f"[WARNING] Camera debug window rendering failed: {e}")
        elif SHOW_CAMERA_DEBUG and not ret:
            debug_x = WIDTH - DEBUG_WINDOW_SIZE[0] - 30
            debug_y = 30
            
            draw_ui_panel(screen, debug_x, debug_y, DEBUG_WINDOW_SIZE[0], DEBUG_WINDOW_SIZE[1],
                         bg_color=DANGER, alpha=200)
            
            error_icon = big_font.render("⚠", True, WHITE)
            error_rect = error_icon.get_rect(center=(debug_x + DEBUG_WINDOW_SIZE[0]//2, debug_y + 60))
            screen.blit(error_icon, error_rect)
            
            error_text = font.render("CAMERA ERROR", True, WHITE)
            error_rect2 = error_text.get_rect(center=(debug_x + DEBUG_WINDOW_SIZE[0]//2, debug_y + 120))
            screen.blit(error_text, error_rect2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                print("[INFO] User closed window - Exiting game")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    print("[INFO] ESC pressed - Exiting game")
                elif event.key == pygame.K_SPACE and game_over:
                    print("[INFO] SPACE pressed - Restarting game")
                    reset_game()
                elif event.key == pygame.K_f:
                    toggle_fullscreen()
                elif event.key == pygame.K_c:
                    toggle_camera_debug()

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
                        print(f"[INFO] Game started! Protector grabbed the treasure at {current_time:.2f}s")
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
            
            if held_threat_id is None and not a_tracking_lost:
                for t in threats:
                    if t["state"] == "IDLE":
                        dist = math.dist(a_hand_smooth, t["pos"])
                        if dist < 150:
                            if a_grip < A_GRAB_THRESH + 0.1:
                                ring_color = ORANGE
                                ring_alpha = int((1 - (dist / 150)) * 200)
                                ring_surf = pygame.Surface((THREAT_SIZE + 20, THREAT_SIZE + 20))
                                ring_surf.set_alpha(ring_alpha)
                                pygame.draw.circle(ring_surf, ring_color,
                                                 (THREAT_SIZE//2 + 10, THREAT_SIZE//2 + 10),
                                                 THREAT_SIZE//2 + 5, 3)
                                screen.blit(ring_surf, (int(t["pos"][0] - THREAT_SIZE//2 - 10),
                                                       int(t["pos"][1] - THREAT_SIZE//2 - 10)))

            for i in range(len(threats) - 1, -1, -1):
                t = threats[i]
                t["lifetime"] += 1/60
                
                if t["state"] == "HELD" and t["id"] == held_threat_id:
                    t["pos"] = a_hand_smooth.copy()
                    if (current_time - held_start_time) > GRAB_LOCK_TIME:
                        if (a_grip > A_DROP_THRESH or np.linalg.norm(a_velocity) > FLING_SPEED_TRIGGER) \
                           and not a_tracking_lost:
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
                    print(f"[INFO] Treasure hit! Lives remaining: {lives}")
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
                        print(f"[INFO] GAME OVER - Attacker wins! Treasure destroyed.")

            if grab_start_time and (current_time - grab_start_time) >= GAME_TIME:
                game_over = True
                win = True
                print(f"[INFO] GAME OVER - Protector wins! Treasure survived {GAME_TIME} seconds!")

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
            pygame.draw.circle(screen, (0, 0, 0), p_hand_smooth.astype(int), 20)
            pygame.draw.circle(screen, CYAN, p_hand_smooth.astype(int), 18, 4)
            pygame.draw.circle(screen, (255, 255, 255), p_hand_smooth.astype(int), 10, 2)
            
        if not a_tracking_lost:
            hand_color = ORANGE if held_threat_id else MAGENTA
            
            if a_grip < A_GRAB_THRESH + 0.2:
                grip_strength = max(0, min(1, (A_GRAB_THRESH + 0.2 - a_grip) / 0.3))
                glow_size = int(20 + grip_strength * 10)
                glow_alpha = int(grip_strength * 150)
                
                glow = pygame.Surface((glow_size * 2, glow_size * 2))
                glow.set_alpha(glow_alpha)
                pygame.draw.circle(glow, hand_color, (glow_size, glow_size), glow_size)
                screen.blit(glow, (int(a_hand_smooth[0] - glow_size),
                                  int(a_hand_smooth[1] - glow_size)))
            
            pygame.draw.circle(screen, (0, 0, 0), a_hand_smooth.astype(int), 20)
            pygame.draw.circle(screen, hand_color, a_hand_smooth.astype(int), 18, 4)
            
            inner_size = int(10 * (a_grip / 2.0))
            if inner_size > 2:
                pygame.draw.circle(screen, (255, 255, 255), a_hand_smooth.astype(int), inner_size, 2)

        if not grab_start_time and not game_over:
            pulse = abs(math.sin(current_time * 2)) * 0.3 + 0.7
            pulse_color = tuple(int(c * pulse) for c in GREEN)
            
            instr = big_font.render("GRAB THE CHEST", True, pulse_color)
            instr_rect = instr.get_rect(center=(WIDTH//2, HEIGHT//2 - 30))
            
            shadow = big_font.render("GRAB THE CHEST", True, (0, 0, 0))
            shadow.set_alpha(150)
            shadow_rect = shadow.get_rect(center=(WIDTH//2 + 3, HEIGHT//2 - 27))
            screen.blit(shadow, shadow_rect)
            screen.blit(instr, instr_rect)
            
            sub = medium_font.render("to begin your defense", True, (150, 150, 170))
            sub_rect = sub.get_rect(center=(WIDTH//2, HEIGHT//2 + 50))
            screen.blit(sub, sub_rect)
            
        elif not game_over:
            rem = max(0, int(GAME_TIME - (current_time - grab_start_time)))
            
            stats_panel_x = 25
            stats_panel_y = 25
            
            hearts_display = "❤️ " * lives
            if lives == 0:
                hearts_display = "💔 💔 💔"
            
            draw_stat_display(screen, "❤️", "LIVES", hearts_display.strip(),
                            stats_panel_x, stats_panel_y, DANGER if lives <= 1 else SUCCESS)
            
            timer_color = DANGER if rem <= 10 else WARNING if rem <= 30 else CYAN
            draw_stat_display(screen, "⏱️", "TIME", f"{rem}s",
                            stats_panel_x, stats_panel_y + 75, timer_color)
            
            draw_stat_display(screen, "⭐", "SCORE", score,
                            stats_panel_x, stats_panel_y + 150, GOLD)
            
            draw_stat_display(screen, "🎯", "DODGED", threats_dodged,
                            stats_panel_x, stats_panel_y + 225, GREEN)
            
            hint_y = HEIGHT - 40
            hint_text = small_font.render("F: Fullscreen  •  C: Camera  •  ESC: Quit",
                                         True, (100, 100, 120))
            hint_rect = hint_text.get_rect(center=(WIDTH//2, hint_y))
            screen.blit(hint_text, hint_rect)
        
        if game_over:
            time_survived = current_time - grab_start_time if grab_start_time else 0
            draw_game_over_screen(win, score, time_survived, threats_dodged)

        pygame.display.flip()
        
except KeyboardInterrupt:
    print("\n[INFO] Game interrupted by user (Ctrl+C)")
except Exception as e:
    print(f"\n[ERROR] Unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("[DEBUG] Cleaning up resources...")
    cap.release()
    pygame.quit()
    print("[INFO] ═══════════════════════════════════════════════════")
    print("[INFO] Game closed successfully")
    print("[INFO] ═══════════════════════════════════════════════════")