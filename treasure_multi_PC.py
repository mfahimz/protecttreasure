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
# 2. CONFIGURATION
# ============================================================
WIDTH, HEIGHT = 1400, 950
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
# 3. PHYSICS TUNING
# ============================================================
TREASURE_SIZE = 90
THREAT_SIZE = 60
BASE_BORDER_RADIUS = 70
BASE_GRAB_RADIUS = 180  
MAX_LIVES = 3  

# --- PROTECTOR PHYSICS ---
MOVE_SMOOTHING = 0.15    # Factors into how "heavy" the chest feels
P_GRAB_THRESH = 0.22     # Fist closed to grab
P_DROP_THRESH = 0.45     # Hand open to drop
GRACE_PERIOD_DURATION = 1.0 

# --- ATTACKER PHYSICS ---
ATTACKER_GRAB_RANGE = 75   
THROW_SPEED = 55.0        
A_GRAB_THRESH = 0.18       
A_DROP_THRESH = 0.38       
FLING_SPEED_TRIGGER = 55.0 
GRAB_LOCK_TIME = 0.1       

# Colors
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
    min_hand_detection_confidence=0.2,
    min_tracking_confidence=0.2        
)
landmarker_hand = vision.HandLandmarker.create_from_options(options_hand)

pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Treasure Guard – Full Movement Build")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)
big_font = pygame.font.Font(None, 100)

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
cap.set(cv2.CAP_PROP_FPS, 60) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 
start_time_ref = time.time() 

# ============================================================
# 5. STATE VARIABLES
# ============================================================
treasure_pos = np.array([WIDTH // 4, HEIGHT // 2], dtype=float)
p_hand_smooth = np.array([WIDTH // 4, HEIGHT // 2], dtype=float)
p_velocity = np.array([0.0, 0.0], dtype=float)
p_grip = 1.0; p_tracking_lost = True; p_grace_timer = 0.0

a_hand_smooth = np.array([WIDTH * 0.75, HEIGHT // 2], dtype=float)
a_velocity = np.array([0.0, 0.0], dtype=float)
a_grip = 1.0; a_tracking_lost = True
held_threat_id = None; held_start_time = 0.0 

threats = []; chest_state = "IDLE"; grab_frames = 0; grab_start_time = None 
game_over = False; win = False; lives = MAX_LIVES; message_text = ""; message_time = 0 

# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================
def get_grip_value(lm):
    palm = lm[0]; tips = [8, 12, 16, 20] 
    return sum(math.dist((palm.x, palm.y), (lm[i].x, lm[i].y)) for i in tips) / 4.0 

def spawn_threat():
    x = random.randint(int(WIDTH * 0.6), WIDTH - 80)
    y = random.randint(HEIGHT // 2 - 100, HEIGHT // 2 + 100)
    return {
        "id": time.time() + random.random(),
        "pos": np.array([x, y], dtype=float), "vel": np.array([0.0, 0.0]), 
        "state": "IDLE", "angle": 0, "rotation_speed": random.uniform(-2, 2)
    }

def update_hand_physics(raw_pos, current_smooth, current_vel):
    if raw_pos is not None:
        new_velocity = raw_pos - current_smooth
        smooth = (raw_pos * 0.25) + (current_smooth * 0.75)
        vel = (new_velocity * 0.5) + (current_vel * 0.5)
        return smooth, vel
    return current_smooth + current_vel, current_vel * 0.9 

# ============================================================
# 7. MAIN GAME LOOP
# ============================================================
try:
    while True:
        clock.tick(60)
        current_time = time.time()
        
        if background_img: screen.blit(background_img, (0, 0))
        else: screen.fill((20, 20, 30))

        # Zone Divider & Top Labels
        zone_line_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.line(zone_line_surf, (100, 100, 100, 40), (WIDTH//2, 0), (WIDTH//2, HEIGHT), 4)
        screen.blit(zone_line_surf, (0,0))
        
        if not grab_start_time:
            screen.blit(font.render("PROTECTOR ZONE (LEFT)", True, CYAN), (100, 20))
            screen.blit(font.render("ATTACKER ZONE (RIGHT)", True, MAGENTA), (WIDTH - 350, 20))

        for event in pygame.event.get():
            if event.type == pygame.QUIT: cap.release(); pygame.quit(); sys.exit()

        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1) 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int((current_time - start_time_ref) * 1000)
        
        res_hand = landmarker_hand.detect_for_video(mp_image, timestamp)
        raw_p_pos, raw_a_pos = None, None
        
        if res_hand.hand_landmarks:
            for lm in res_hand.hand_landmarks:
                wrist = lm[0]
                px = np.array([int(wrist.x * WIDTH), int(wrist.y * HEIGHT)], dtype=float)
                grip = get_grip_value(lm)
                if px[0] < WIDTH * 0.48:
                    raw_p_pos, p_grip, p_tracking_lost = px, grip, False
                elif px[0] > WIDTH * 0.52:
                    raw_a_pos, a_grip, a_tracking_lost = px, grip, False
        
        if raw_p_pos is None: p_tracking_lost = True
        if raw_a_pos is None: a_tracking_lost = True

        p_hand_smooth, p_velocity = update_hand_physics(raw_p_pos, p_hand_smooth, p_velocity)
        a_hand_smooth, a_velocity = update_hand_physics(raw_a_pos, a_hand_smooth, a_velocity)

        if not game_over:
            if grab_start_time and len(threats) < 6 and random.random() < 0.03: 
                threats.append(spawn_threat())

            # --- PROTECTOR LOGIC (THE FIX) ---
            # Determine if hand is "holding" based on grip strength
            is_p_holding = (chest_state == "IDLE" and p_grip < P_GRAB_THRESH) or (chest_state == "GRABBED" and p_grip < P_DROP_THRESH)
            
            if chest_state == "IDLE":
                if is_p_holding and not p_tracking_lost:
                    if np.linalg.norm(p_hand_smooth - treasure_pos) < BASE_GRAB_RADIUS:
                        grab_frames += 1
                        if grab_frames > 2:
                            chest_state = "GRABBED"
                            grab_frames = 0; p_grace_timer = 0
                            if grab_start_time is None: grab_start_time = time.time()
                else: grab_frames = 0
            
            elif chest_state == "GRABBED":
                if is_p_holding:
                    p_grace_timer = 0
                    # SMOOTH MOVEMENT: Move chest toward hand
                    treasure_pos += (p_hand_smooth - treasure_pos) * MOVE_SMOOTHING
                else:
                    # Grace period before dropping
                    if p_grace_timer == 0: p_grace_timer = current_time
                    if (current_time - p_grace_timer) > GRACE_PERIOD_DURATION:
                        chest_state = "IDLE"; p_grace_timer = 0

            # --- ATTACKER LOGIC ---
            if held_threat_id is None and not a_tracking_lost and a_grip < A_GRAB_THRESH: 
                target = next((t for t in threats if t["state"] == "IDLE" and math.dist(a_hand_smooth, t["pos"]) < ATTACKER_GRAB_RANGE), None)
                if target:
                    target["state"] = "HELD"; held_threat_id = target["id"]; held_start_time = current_time 

            for i in range(len(threats) - 1, -1, -1):
                t = threats[i]
                if t["state"] == "HELD" and t["id"] == held_threat_id:
                    t["pos"] = a_hand_smooth.copy() 
                    if (current_time - held_start_time) > GRAB_LOCK_TIME:
                        if a_grip > A_DROP_THRESH or np.linalg.norm(a_velocity) > FLING_SPEED_TRIGGER:
                            t["state"] = "FIRED"; t["vel"] = np.array([-1.0, 0.0]) * THROW_SPEED; held_threat_id = None
                
                elif t["state"] == "FIRED":
                    t["pos"] += t["vel"] 
                    t["angle"] = math.degrees(math.atan2(-t["vel"][1], t["vel"][0]))
                
                if np.linalg.norm(t["pos"] - treasure_pos) < BASE_BORDER_RADIUS + 20:
                    lives -= 1; threats.pop(i)
                    if held_threat_id == t["id"]: held_threat_id = None
                    if lives < 0: game_over = True; win = False

            if grab_start_time and (current_time - grab_start_time) >= GAME_TIME:
                game_over = True; win = True

        # RENDER CHEST
        # 
        pygame.draw.circle(screen, GREEN if chest_state == "GRABBED" else YELLOW, treasure_pos.astype(int), BASE_BORDER_RADIUS, 4)
        screen.blit(chest_img, (int(treasure_pos[0]-45), int(treasure_pos[1]-45)))

        # RENDER THREATS
        for t in threats:
            rot = pygame.transform.rotate(threat_img, t.get("angle", 0))
            screen.blit(rot, rot.get_rect(center=(int(t["pos"][0]), int(t["pos"][1]))).topleft)

        # RENDER HANDS (Cyan for Protector, Magenta/White for Attacker)
        if not p_tracking_lost: 
            pygame.draw.circle(screen, CYAN, p_hand_smooth.astype(int), 15, 3)
            if not grab_start_time:
                screen.blit(font.render("Protector", True, CYAN), (p_hand_smooth[0], p_hand_smooth[1]-30))

        if not a_tracking_lost: 
            cursor_col = MAGENTA if held_threat_id else WHITE
            pygame.draw.circle(screen, cursor_col, a_hand_smooth.astype(int), 15, 3)
            if not grab_start_time:
                screen.blit(font.render("Attacker", True, cursor_col), (a_hand_smooth[0], a_hand_smooth[1]-30))

        # UI OVERLAYS
        if not grab_start_time:
            instr = big_font.render("GRAB CHEST TO START", True, GREEN)
            screen.blit(instr, (WIDTH//2 - instr.get_width()//2, HEIGHT//2))
        else:
            lives_txt = font.render(f"Lives: {'♥ ' * (lives + 1)}", True, RED)
            screen.blit(lives_txt, (20, 20))
            rem = max(0, int(GAME_TIME - (current_time - grab_start_time)))
            screen.blit(font.render(f"Time: {rem}s", True, WHITE), (20, 60))

        if game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT)); overlay.set_alpha(200); overlay.fill((0, 0, 0)); screen.blit(overlay, (0, 0))
            txt = big_font.render("PROTECTOR WINS!" if win else "ATTACKER WINS!", True, GREEN if win else RED)
            screen.blit(txt, (WIDTH//2 - txt.get_width()//2, HEIGHT//2 - 50))

        pygame.display.flip()
except KeyboardInterrupt:
    cap.release(); pygame.quit()