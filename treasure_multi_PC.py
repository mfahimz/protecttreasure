# 1. IMPORTS & SETUP
# ============================================================
import os, sys, time, random, math, cv2, pygame
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

# ---- Windows Hardware Configuration ----
os.environ["SDL_VIDEO_CENTERED"] = "1"

# ============================================================
# 2. CONFIGURATION & AUTO-SCALING
# ============================================================
pygame.init()
# Get monitor information for adaptive window sizing
screen_info = pygame.display.Info()
WIDTH = screen_info.current_w
HEIGHT = screen_info.current_h

CAMERA_INDEX = 1        
GAME_TIME = 60          
MODEL_PATH_HAND = "models/hand_landmarker.task"

# Asset Paths (Windows)
CHEST_IMAGE_PATH = r"C:\Users\pc\protecttreasure\CHEST_IMAGE_PATH.png"
THREAT_IMAGE_PATH = r"C:\Users\pc\protecttreasure\THREAT_IMAGE_PATH.png"
BACKGROUND_IMAGE_PATH = r"C:\Users\pc\protecttreasure\BACKGROUND_IMAGE_PATH.png"
BACKGROUND_MUSIC_PATH = r"C:\Users\pc\protecttreasure\BACKGROUND_MUSIC_PATH.mp3"
HIT_SOUND_PATH = r"C:\Users\pc\protecttreasure\HIT_SOUND_PATH.wav"

# ============================================================
# 3. PHYSICS & STABILITY TUNING
# ============================================================
TREASURE_SIZE = 90
THREAT_SIZE = 60
BASE_BORDER_RADIUS = 70
BASE_GRAB_RADIUS = 180  
MAX_LIVES = 3  

HIT_FLASH_DURATION = 0.4  
SHAKE_INTENSITY = 12      

P_GRAB_THRESH = 1.3; P_DROP_THRESH = 1.9
A_GRAB_THRESH = 1.15; A_DROP_THRESH = 1.85    
RELEASE_BUFFER_MAX = 2
MOVE_SMOOTHING = 0.15
THROW_SPEED = 55.0
FLING_SPEED_TRIGGER = 110.0 
GRAB_LOCK_TIME = 0.3 

WHITE = (255, 255, 255); RED = (220, 60, 60); GREEN = (80, 220, 120)
YELLOW = (255, 200, 100); BLUE = (100, 200, 255); CYAN = (80, 200, 255)
MAGENTA = (220, 80, 220)

# ============================================================
# 4. INITIALIZE SYSTEMS
# ============================================================
options_hand = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH_HAND),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=4,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5        
)
landmarker_hand = vision.HandLandmarker.create_from_options(options_hand)

pygame.mixer.init()
# Adaptive screen mode (Fullscreen-like borderless window)
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)
pygame.display.set_caption("Treasure Guard – Debug & Auto-Scale Build")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 35); big_font = pygame.font.Font(None, 100)

try:
    background_img = pygame.image.load(BACKGROUND_IMAGE_PATH).convert()
    background_img = pygame.transform.smoothscale(background_img, (WIDTH, HEIGHT))
except: background_img = None 

try:
    chest_img = pygame.transform.smoothscale(pygame.image.load(CHEST_IMAGE_PATH).convert_alpha(), (TREASURE_SIZE, TREASURE_SIZE))
    threat_img = pygame.transform.smoothscale(pygame.image.load(THREAT_IMAGE_PATH).convert_alpha(), (THREAT_SIZE, THREAT_SIZE))
except:
    chest_img = pygame.Surface((TREASURE_SIZE, TREASURE_SIZE)); chest_img.fill(YELLOW)
    threat_img = pygame.Surface((THREAT_SIZE, THREAT_SIZE)); threat_img.fill(RED)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened(): cap = cv2.VideoCapture(0)
start_time_ref = time.time() 

# ============================================================
# 5. STATE VARIABLES
# ============================================================
treasure_pos = np.array([WIDTH // 4, HEIGHT // 2], dtype=float)
p_hand_smooth = np.array([WIDTH // 4, HEIGHT // 2], dtype=float)
p_velocity = np.array([0.0, 0.0], dtype=float) 
p_grip = 1.0; p_tracking_lost = True

a_hand_smooth = np.array([WIDTH * 0.75, HEIGHT // 2], dtype=float)
a_velocity = np.array([0.0, 0.0], dtype=float)
a_grip = 1.0; a_tracking_lost = True
held_threat_id = None; held_start_time = 0.0 
a_release_counter = 0

threats = []; chest_state = "IDLE"; grab_start_time = None 
game_over = False; win = False; lives = MAX_LIVES; message_text = ""; message_time = 0 
hit_anim_timer = 0.0

# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================
def get_grip_value(lm):
    wrist = lm[0]; mcp = lm[9]
    hand_size = math.dist((wrist.x, wrist.y), (mcp.x, mcp.y))
    if hand_size < 0.01: return 1.0
    tips = [8, 12, 16, 20]
    avg_finger_dist = sum(math.dist((wrist.x, wrist.y), (lm[i].x, lm[i].y)) for i in tips) / 4.0
    return avg_finger_dist / hand_size 

def spawn_threat():
    x = random.randint(int(WIDTH * 0.6), WIDTH - 80)
    y = random.randint(HEIGHT // 2 - 150, HEIGHT // 2 + 150)
    return {
        "id": time.time() + random.random(),
        "pos": np.array([x, y], dtype=float), "vel": np.array([0.0, 0.0]), 
        "state": "IDLE", "angle": 0, "rotation_speed": random.uniform(-2, 2)
    }

def update_hand_physics(raw_pos, current_smooth, current_vel):
    if raw_pos is not None:
        new_velocity = raw_pos - current_smooth
        smooth = (raw_pos * 0.15) + (current_smooth * 0.85)
        vel = (new_velocity * 0.3) + (current_vel * 0.7)
        return smooth, vel
    return current_smooth + current_vel, current_vel * 0.80 

# ============================================================
# 7. MAIN GAME LOOP
# ============================================================
try:
    while True:
        clock.tick(60)
        current_time = time.time()
        
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1) 

        # CLAHE Light Processing
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, ac, bc = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
        l = clahe.apply(l); limg = cv2.merge((l, ac, bc))
        frame_proc = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int((current_time - start_time_ref) * 1000)
        
        res_hand = landmarker_hand.detect_for_video(mp_image, timestamp)
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

        if background_img: screen.blit(background_img, (0, 0))
        else: screen.fill((20, 20, 30))

        for event in pygame.event.get():
            if event.type == pygame.QUIT: cap.release(); pygame.quit(); sys.exit()

        if not game_over:
            if grab_start_time and len(threats) < 6 and random.random() < 0.04: 
                threats.append(spawn_threat())

            # Protector Logic
            is_p_holding = (chest_state == "IDLE" and p_grip < P_GRAB_THRESH) or (chest_state == "GRABBED" and p_grip < P_DROP_THRESH)
            if chest_state == "IDLE" and is_p_holding and not p_tracking_lost:
                if np.linalg.norm(p_hand_smooth - treasure_pos) < BASE_GRAB_RADIUS:
                    chest_state = "GRABBED"
                    if grab_start_time is None: grab_start_time = current_time
            elif chest_state == "GRABBED":
                if is_p_holding or p_tracking_lost:
                    treasure_pos += (p_hand_smooth - treasure_pos) * MOVE_SMOOTHING
                else: chest_state = "IDLE"

            # Attacker Logic
            if held_threat_id is None and not a_tracking_lost and a_grip < A_GRAB_THRESH: 
                target = next((t for t in threats if t["state"] == "IDLE" and math.dist(a_hand_smooth, t["pos"]) < 150), None)
                if target:
                    target["state"] = "HELD"; held_threat_id = target["id"]; held_start_time = current_time; a_release_counter = 0

            for i in range(len(threats) - 1, -1, -1):
                t = threats[i]
                if t["state"] == "HELD" and t["id"] == held_threat_id:
                    t["pos"] = a_hand_smooth.copy() 
                    if (current_time - held_start_time) > GRAB_LOCK_TIME:
                        speed = np.linalg.norm(a_velocity)
                        if (a_grip > A_DROP_THRESH or speed > FLING_SPEED_TRIGGER) and not a_tracking_lost:
                            a_release_counter += 1
                            if a_release_counter >= RELEASE_BUFFER_MAX:
                                t["state"] = "FIRED"; t["vel"] = np.array([-1.0, 0.0]) * THROW_SPEED; held_threat_id = None
                        else: a_release_counter = max(0, a_release_counter - 1)
                elif t["state"] == "FIRED":
                    t["pos"] += t["vel"]; t["angle"] = math.degrees(math.atan2(-t["vel"][1], t["vel"][0]))
                
                if np.linalg.norm(t["pos"] - treasure_pos) < BASE_BORDER_RADIUS + 20:
                    lives -= 1; threats.pop(i)
                    if held_threat_id == t["id"]: held_threat_id = None
                    hit_anim_timer = current_time 
                    if lives < 0: game_over = True; win = False

            if grab_start_time and (current_time - grab_start_time) >= GAME_TIME:
                game_over = True; win = True

        # --- DEBUG CAMERA VIEW ---
        # Resize OpenCV frame for debug view (small overlay top-right)
        debug_w, debug_h = 320, 180
        debug_frame = cv2.resize(frame, (debug_w, debug_h))
        debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
        debug_surf = pygame.surfarray.make_surface(debug_frame.swapaxes(0, 1))
        screen.blit(debug_surf, (WIDTH - debug_w - 20, 20))
        pygame.draw.rect(screen, CYAN, (WIDTH - debug_w - 20, 20, debug_w, debug_h), 2)

        # --- RENDER GAME ASSETS ---
        display_pos = treasure_pos.copy()
        if (current_time - hit_anim_timer) < HIT_FLASH_DURATION:
            display_pos += np.array([random.randint(-SHAKE_INTENSITY, SHAKE_INTENSITY), random.randint(-SHAKE_INTENSITY, SHAKE_INTENSITY)])
            screen.blit(font.render("HIT!", True, RED), (int(treasure_pos[0]-25), int(treasure_pos[1]-80)))
            chest_color = RED
        else: chest_color = GREEN if grab_start_time else YELLOW

        pygame.draw.circle(screen, chest_color, display_pos.astype(int), BASE_BORDER_RADIUS, 4)
        screen.blit(chest_img, (int(display_pos[0]-45), int(display_pos[1]-45)))

        for t in threats:
            rot = pygame.transform.rotate(threat_img, t.get("angle", 0))
            screen.blit(rot, rot.get_rect(center=(int(t["pos"][0]), int(t["pos"][1]))).topleft)

        if not p_tracking_lost: pygame.draw.circle(screen, CYAN, p_hand_smooth.astype(int), 15, 3)
        if not a_tracking_lost: pygame.draw.circle(screen, MAGENTA if held_threat_id else WHITE, a_hand_smooth.astype(int), 15, 3)

        if not grab_start_time:
            instr = big_font.render("GRAB CHEST TO START", True, GREEN)
            screen.blit(instr, (WIDTH//2 - instr.get_width()//2, HEIGHT//2))
        else:
            rem = max(0, int(GAME_TIME - (current_time - grab_start_time)))
            screen.blit(font.render(f"Lives: {'♥ ' * (lives + 1)}", True, RED), (20, 20))
            screen.blit(font.render(f"Time: {rem}s", True, WHITE), (20, 60))

        pygame.display.flip()
except KeyboardInterrupt:
    cap.release(); pygame.quit()