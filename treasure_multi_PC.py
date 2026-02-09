# 1. IMPORTS & SETUP
# ============================================================
import os, sys, time, random, math, cv2, pygame
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

os.environ["SDL_VIDEO_CENTERED"] = "1"

# ============================================================
# 2. DISPLAY CONFIGURATION - FIXED FOR TV/EXTENDED SCREENS
# ============================================================
pygame.init()

# Let user choose display if multiple monitors exist
def select_display():
    num_displays = pygame.display.get_num_displays()
    print(f"\n=== DISPLAY SELECTION ===")
    print(f"Found {num_displays} display(s)")
    
    for i in range(num_displays):
        info = pygame.display.Info()
        print(f"Display {i}: {info.current_w}x{info.current_h}")
    
    if num_displays > 1:
        choice = input(f"\nSelect display (0-{num_displays-1}) [default=0]: ").strip()
        return int(choice) if choice.isdigit() else 0
    return 0

# Configuration options
USE_FULLSCREEN = True  # Set to False for windowed mode
DISPLAY_INDEX = select_display()
WIDTH, HEIGHT = 1920, 1080  # Full HD - adjust if needed

if USE_FULLSCREEN:
    # True fullscreen on selected display
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{DISPLAY_INDEX},0"
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=DISPLAY_INDEX)
    WIDTH, HEIGHT = screen.get_size()
    print(f"Fullscreen mode: {WIDTH}x{HEIGHT} on Display {DISPLAY_INDEX}")
else:
    # Windowed mode
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    print(f"Windowed mode: {WIDTH}x{HEIGHT}")

pygame.display.set_caption("Treasure Guard - Enhanced Edition")

CAMERA_INDEX = 1        
GAME_TIME = 60          
MODEL_PATH_HAND = "models/hand_landmarker.task"

# Asset Paths
CHEST_IMAGE_PATH = "CHEST_IMAGE_PATH.png"
THREAT_IMAGE_PATH = "THREAT_IMAGE_PATH.png"
BACKGROUND_IMAGE_PATH = "BACKGROUND_IMAGE_PATH.jpeg"
BACKGROUND_MUSIC_PATH = "BACKGROUND_MUSIC_PATH.mp3"
HIT_SOUND_PATH = "HIT_SOUND_PATH.wav"

# ============================================================
# 3. PHYSICS & STABILITY TUNING
# ============================================================
TREASURE_SIZE, THREAT_SIZE = 90, 60
BASE_BORDER_RADIUS, BASE_GRAB_RADIUS = 70, 180  
MAX_LIVES = 3  
HIT_FLASH_DURATION = 0.4  
SHAKE_INTENSITY = 12      

P_GRAB_THRESH, P_DROP_THRESH = 1.3, 1.9
A_GRAB_THRESH, A_DROP_THRESH = 1.15, 1.85    
RELEASE_BUFFER_MAX = 2
MOVE_SMOOTHING, THROW_SPEED = 0.15, 55.0
FLING_SPEED_TRIGGER, GRAB_LOCK_TIME = 110.0, 0.3 

# Colors
WHITE = (255, 255, 255); RED = (220, 60, 60); GREEN = (80, 220, 120)
YELLOW = (255, 200, 100); CYAN = (80, 200, 255); MAGENTA = (220, 80, 220)
DARK_BG = (20, 20, 30); GOLD = (255, 215, 0)

# ============================================================
# 4. INITIALIZE SYSTEMS
# ============================================================
options_hand = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH_HAND),
    running_mode=vision.RunningMode.VIDEO, num_hands=4,
    min_hand_detection_confidence=0.5, min_tracking_confidence=0.5        
)
landmarker_hand = vision.HandLandmarker.create_from_options(options_hand)

pygame.mixer.init()
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 35)
big_font = pygame.font.Font(None, 100)
title_font = pygame.font.Font(None, 150)
medium_font = pygame.font.Font(None, 60)

# Load sounds with error handling
def load_sound(path):
    try:
        return pygame.mixer.Sound(path)
    except:
        print(f"Warning: Could not load sound {path}")
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

try:
    background_img = pygame.image.load(BACKGROUND_IMAGE_PATH).convert()
    background_img = pygame.transform.smoothscale(background_img, (WIDTH, HEIGHT))
except: 
    background_img = None 

# Background music
try:
    pygame.mixer.music.load(BACKGROUND_MUSIC_PATH)
    pygame.mixer.music.set_volume(0.3)
    pygame.mixer.music.play(-1)
except:
    print("Warning: Could not load background music")

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened(): 
    print(f"Camera {CAMERA_INDEX} not available, trying default camera...")
    cap = cv2.VideoCapture(0)
start_time_ref = time.time() 

# ============================================================
# 5. STATE VARIABLES
# ============================================================
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
score = 0  # NEW: Score system
threats_dodged = 0  # NEW: Dopamine counter

# ============================================================
# 6. HELPERS
# ============================================================
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
        "lifetime": 0.0  # NEW: Track threat lifetime
    }

def update_hand_physics(raw_pos, current_smooth, current_vel):
    if raw_pos is not None:
        new_velocity = raw_pos - current_smooth
        smooth = (raw_pos * 0.15) + (current_smooth * 0.85)
        vel = (new_velocity * 0.3) + (current_vel * 0.7)
        return smooth, vel
    return current_smooth + current_vel, current_vel * 0.80 

def draw_game_over_screen(won, final_score, time_survived, dodged):
    """NEW: Draw comprehensive game over screen"""
    # Semi-transparent overlay
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(200)
    overlay.fill(DARK_BG)
    screen.blit(overlay, (0, 0))
    
    # Main result
    if won:
        result_text = title_font.render("VICTORY!", True, GOLD)
        subtitle = medium_font.render("You Protected the Treasure!", True, GREEN)
    else:
        result_text = title_font.render("DEFEAT!", True, RED)
        subtitle = medium_font.render("The Treasure Was Lost...", True, RED)
    
    screen.blit(result_text, (WIDTH//2 - result_text.get_width()//2, HEIGHT//4))
    screen.blit(subtitle, (WIDTH//2 - subtitle.get_width()//2, HEIGHT//4 + 120))
    
    # Stats
    stats_y = HEIGHT//2
    stats = [
        f"Time Survived: {int(time_survived)}s / {GAME_TIME}s",
        f"Threats Dodged: {dodged}",
        f"Final Score: {final_score}",
        f"Lives Remaining: {lives}"
    ]
    
    for i, stat in enumerate(stats):
        text = medium_font.render(stat, True, WHITE)
        screen.blit(text, (WIDTH//2 - text.get_width()//2, stats_y + i*60))
    
    # Instructions
    restart_text = font.render("Press SPACE to Restart  |  ESC to Quit", True, CYAN)
    screen.blit(restart_text, (WIDTH//2 - restart_text.get_width()//2, HEIGHT - 100))

def reset_game():
    """NEW: Reset all game variables"""
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

# ============================================================
# 7. MAIN GAME LOOP
# ============================================================
try:
    running = True
    while running:
        clock.tick(60)
        current_time = time.time()
        
        # Handle window resize
        if not USE_FULLSCREEN:
            current_size = screen.get_size()
            if current_size != (WIDTH, HEIGHT):
                WIDTH, HEIGHT = current_size
        
        ret, frame = cap.read()
        if not ret: 
            continue
        frame = cv2.flip(frame, 1)
        
        # Light stabilization
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, ac, bc = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
        l = clahe.apply(l)
        limg = cv2.merge((l, ac, bc))
        frame_proc = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res_hand = landmarker_hand.detect_for_video(mp_image, int((current_time - start_time_ref) * 1000))
        
        raw_p_pos, raw_a_pos = None, None
        if res_hand.hand_landmarks:
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

        # Draw background
        if background_img: 
            screen.blit(background_img, (0, 0))
        else: 
            screen.fill(DARK_BG)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and game_over:
                    reset_game()
                elif event.key == pygame.K_f:  # Toggle fullscreen
                    USE_FULLSCREEN = not USE_FULLSCREEN
                    if USE_FULLSCREEN:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        WIDTH, HEIGHT = screen.get_size()
                    else:
                        screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
                        WIDTH, HEIGHT = 1920, 1080

        if not game_over:
            # Spawn threats
            if grab_start_time and len(threats) < 6 and random.random() < 0.04: 
                threats.append(spawn_threat())

            # Protector Logic
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

            # Attacker Logic
            if held_threat_id is None and not a_tracking_lost and a_grip < A_GRAB_THRESH: 
                target = next((t for t in threats if t["state"] == "IDLE" and \
                              math.dist(a_hand_smooth, t["pos"]) < 150), None)
                if target: 
                    target["state"] = "HELD"
                    held_threat_id = target["id"]
                    held_start_time = current_time
                    a_release_counter = 0

            # Update threats
            for i in range(len(threats) - 1, -1, -1):
                t = threats[i]
                t["lifetime"] += 1/60  # Track time
                
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
                    
                    # Remove if off screen
                    if t["pos"][0] < -100:
                        threats.pop(i)
                        threats_dodged += 1  # NEW: Count dodged threats
                        score += 10  # NEW: Score for dodging
                        if held_threat_id == t["id"]: 
                            held_threat_id = None
                        continue
                
                # Hit detection - FIXED
                if np.linalg.norm(t["pos"] - treasure_pos) < BASE_BORDER_RADIUS + 20:
                    lives -= 1  # Reduce lives
                    threats.pop(i)
                    if held_threat_id == t["id"]: 
                        held_threat_id = None
                    hit_anim_timer = current_time
                    
                    # Play sound
                    if hit_sound:
                        hit_sound.play()
                    
                    # FIXED: Check if game over
                    if lives <= 0:  # Changed from < 0 to <= 0
                        game_over = True
                        win = False

            # Win condition
            if grab_start_time and (current_time - grab_start_time) >= GAME_TIME: 
                game_over = True
                win = True

        # ============================================================
        # RENDERING
        # ============================================================
        
        # Render Hit Animation
        disp_pos = treasure_pos.copy()
        is_hit = (current_time - hit_anim_timer) < HIT_FLASH_DURATION
        if is_hit:
            disp_pos += np.array([random.randint(-SHAKE_INTENSITY, SHAKE_INTENSITY), 
                                 random.randint(-SHAKE_INTENSITY, SHAKE_INTENSITY)])
            hit_text = font.render("HIT!", True, RED)
            screen.blit(hit_text, (int(treasure_pos[0]-25), int(treasure_pos[1]-80)))
            c_col = RED
        else: 
            c_col = GREEN if grab_start_time else YELLOW

        pygame.draw.circle(screen, c_col, disp_pos.astype(int), BASE_BORDER_RADIUS, 4)
        screen.blit(chest_img, (int(disp_pos[0]-45), int(disp_pos[1]-45)))
        
        # Draw threats
        for t in threats:
            rot = pygame.transform.rotate(threat_img, t.get("angle", 0))
            screen.blit(rot, rot.get_rect(center=(int(t["pos"][0]), int(t["pos"][1]))).topleft)
        
        # Draw hands
        if not p_tracking_lost: 
            pygame.draw.circle(screen, CYAN, p_hand_smooth.astype(int), 15, 3)
        if not a_tracking_lost: 
            pygame.draw.circle(screen, MAGENTA if held_threat_id else WHITE, 
                             a_hand_smooth.astype(int), 15, 3)

        # UI Elements
        if not grab_start_time and not game_over:
            instr = big_font.render("GRAB CHEST TO START", True, GREEN)
            screen.blit(instr, (WIDTH//2 - instr.get_width()//2, HEIGHT//2))
        elif not game_over:
            rem = max(0, int(GAME_TIME - (current_time - grab_start_time)))
            
            # FIXED: Lives display
            hearts = "♥ " * lives  # Removed the +1
            lives_text = font.render(f"Lives: {hearts}", True, RED)
            screen.blit(lives_text, (20, 20))
            
            # Timer
            time_text = font.render(f"Time: {rem}s", True, WHITE)
            screen.blit(time_text, (20, 60))
            
            # NEW: Score and stats
            score_text = font.render(f"Score: {score}", True, GOLD)
            screen.blit(score_text, (20, 100))
            
            dodged_text = font.render(f"Dodged: {threats_dodged}", True, GREEN)
            screen.blit(dodged_text, (20, 140))
        
        # GAME OVER SCREEN - NEW
        if game_over:
            time_survived = current_time - grab_start_time if grab_start_time else 0
            draw_game_over_screen(win, score, time_survived, threats_dodged)

        pygame.display.flip()
        
except KeyboardInterrupt:
    print("\nGame interrupted by user")
finally:
    cap.release()
    pygame.quit()
    print("Game closed successfully")