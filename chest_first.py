# ============================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================
import os, sys, time, random, math, cv2, pygame
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

# ---- macOS SDL safety ----
os.environ["SDL_VIDEO_CENTERED"] = "1"
os.environ["SDL_AUDIODRIVER"] = "coreaudio"

# ============================================================
# CONFIGURATION (UPDATED)
# ============================================================
WIDTH, HEIGHT = 1400, 950
CAMERA_INDEX = 1        # [FIX] Set to 1 as requested
GAME_TIME = 30          # [FIX] Set to 30 seconds
MODEL_PATH_HAND = "models/hand_landmarker.task"
MODEL_PATH_FACE = "models/face_landmarker.task"

# Paths
CHEST_IMAGE_PATH = "/Users/fahim/Desktop/Landmark_detection/CHEST_IMAGE_PATH.png"
THREAT_IMAGE_PATH = "/Users/fahim/Desktop/Landmark_detection/THREAT_IMAGE_PATH.png"
BACKGROUND_IMAGE_PATH = "/Users/fahim/Desktop/Landmark_detection/BACKGROUND_IMAGE_PATH.png"
BACKGROUND_MUSIC_PATH = "/Users/fahim/Desktop/Landmark_detection/BACKGROUND_MUSIC_PATH.mp3"
HIT_SOUND_PATH = "/Users/fahim/Desktop/Landmark_detection/HIT_SOUND_PATH.wav"
ROAR_SOUND_PATH = "/Users/fahim/Desktop/Landmark_detection/ROAR_SOUND_PATH.wav"

# ============================================================
# GAME SETTINGS
# ============================================================
TREASURE_SIZE = 90
BASE_BORDER_RADIUS = 70
BASE_GRAB_RADIUS = 180  
MOVE_SMOOTHING = 0.9    
MAX_LIVES = 3  

NORMAL_THREAT_SIZE = 60
JUGGERNAUT_SIZE = 120   

# Hand Physics
GRAB_THRESHOLD = 0.22   
DROP_THRESHOLD = 0.45   
GRACE_PERIOD_DURATION = 1.0 

# Roar
MOUTH_OPEN_THRESHOLD = 0.05
ROAR_COOLDOWN = 15.0 

# Difficulty
SPAWN_RATE = 0.02       
START_SPEED = 7.0
TURN_SPEED = 0.04       

# Colors
WHITE = (255, 255, 255)
RED = (220, 60, 60)
GREEN = (80, 220, 120)
YELLOW = (255, 200, 100)
BLUE = (100, 200, 255) 
CYAN = (80, 200, 255)
GOLD = (255, 215, 0)
ORANGE = (255, 100, 0)

# ============================================================
# INITIALIZE SYSTEMS
# ============================================================
options_hand = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH_HAND),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.1, 
    min_tracking_confidence=0.1        
)
landmarker_hand = vision.HandLandmarker.create_from_options(options_hand)

options_face = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH_FACE),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
landmarker_face = vision.FaceLandmarker.create_from_options(options_face)

pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Treasure Guard – ARCADE FINAL")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)
big_font = pygame.font.Font(None, 100)

try:
    background_img = pygame.image.load(BACKGROUND_IMAGE_PATH).convert()
    background_img = pygame.transform.smoothscale(background_img, (WIDTH, HEIGHT))
except: background_img = None

chest_img = pygame.transform.smoothscale(pygame.image.load(CHEST_IMAGE_PATH).convert_alpha(), (TREASURE_SIZE, TREASURE_SIZE))
threat_img = pygame.transform.smoothscale(pygame.image.load(THREAT_IMAGE_PATH).convert_alpha(), (NORMAL_THREAT_SIZE, NORMAL_THREAT_SIZE))
juggernaut_img = pygame.transform.smoothscale(threat_img, (JUGGERNAUT_SIZE, JUGGERNAUT_SIZE))

try: pygame.mixer.music.load(BACKGROUND_MUSIC_PATH); pygame.mixer.music.set_volume(0.6); pygame.mixer.music.play(-1)
except: pass
hit_sound = None; roar_sound = None
try: hit_sound = pygame.mixer.Sound(HIT_SOUND_PATH); hit_sound.set_volume(0.9)
except: pass
try: roar_sound = pygame.mixer.Sound(ROAR_SOUND_PATH); roar_sound.set_volume(1.0)
except: pass

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
start_time_ref = time.time()

# ============================================================
# STATE
# ============================================================
treasure_pos = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
smooth_hand_pos = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)
hand_velocity = np.array([0.0, 0.0], dtype=float)

threats = []
coins = [] 
state = "IDLE"
grab_frames = 0
game_start_time = time.time()
grab_start_time = None
elapsed = 0
score = 0
game_over = False
win = False
lives = MAX_LIVES
message_text = ""
message_time = 0
tracking_lost = False 
grace_timer_start = 0.0 
current_grip_value = 1.0

last_roar_time = -999
roar_active = False
roar_radius = 0

# ============================================================
# LOGIC
# ============================================================

def get_grip_value(lm):
    palm = lm[0]; tips = [8, 12, 16, 20]
    avg_dist = sum(math.dist((palm.x, palm.y), (lm[i].x, lm[i].y)) for i in tips)
    return avg_dist / 4.0 

def is_mouth_open(face_landmarks):
    upper = face_landmarks[13]; lower = face_landmarks[14]
    return math.dist((upper.x, upper.y), (lower.x, lower.y)) > MOUTH_OPEN_THRESHOLD

def spawn_threat(difficulty):
    angle = random.uniform(0, 2 * math.pi)
    # [FIX] Spawn slightly closer so they appear faster
    dist = max(WIDTH, HEIGHT) * 0.7 
    pos = np.array([WIDTH//2 + math.cos(angle)*dist, HEIGHT//2 + math.sin(angle)*dist], dtype=float)
    
    target_dir = (treasure_pos - pos)
    target_dir /= np.linalg.norm(target_dir)
    
    is_boss = (random.random() < 0.1)
    
    if is_boss:
        speed = START_SPEED * 0.6 
        size = JUGGERNAUT_SIZE
        hp = 3
        type_tag = "JUGGERNAUT"
    else:
        speed = random.uniform(START_SPEED, START_SPEED + (difficulty * 4))
        size = NORMAL_THREAT_SIZE
        hp = 1
        type_tag = "NORMAL"

    return {
        "pos": pos, "vel": target_dir * speed, "speed": speed, 
        "type": type_tag, "size": size, "hp": hp
    }

def spawn_coin():
    x = random.randint(100, WIDTH - 100)
    y = random.randint(100, HEIGHT - 100)
    return {
        "pos": np.array([x, y], dtype=float),
        "spawn_time": time.time()
    }

# ============================================================
# MAIN LOOP
# ============================================================
try:
    while True:
        clock.tick(60)
        current_time = time.time()
        
        if grab_start_time:
            uptime = current_time - grab_start_time
            intensity = min(1.0, uptime / 30.0) # Scale difficulty over 30s
        else: intensity = 0.0

        # BG
        if message_text == "HIT!" and (current_time - message_time) < 0.1: screen.fill(RED) 
        elif message_text == "+1000!" and (current_time - message_time) < 0.1: screen.fill(GOLD)
        elif background_img: screen.blit(background_img, (0, 0))
        else: screen.fill((20, 20, 30))

        for event in pygame.event.get():
            if event.type == pygame.QUIT: cap.release(); pygame.quit(); sys.exit()

        # --- 1. VISION ---
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int((current_time - start_time_ref) * 1000)
        
        res_hand = landmarker_hand.detect_for_video(mp_image, timestamp)
        res_face = landmarker_face.detect_for_video(mp_image, timestamp)

        # Hand Logic
        raw_hand = None
        if res_hand.hand_landmarks:
            lm = res_hand.hand_landmarks[0]
            palm = lm[0]
            raw_hand = np.array([int(palm.x * WIDTH), int(palm.y * HEIGHT)], dtype=float)
            current_grip_value = get_grip_value(lm)
            tracking_lost = False
        else:
            tracking_lost = True 

        # High Speed Momentum
        if raw_hand is not None:
            new_velocity = raw_hand - smooth_hand_pos
            dist = np.linalg.norm(new_velocity)
            if dist > 400: 
                smooth_hand_pos = raw_hand 
                hand_velocity = np.array([0.0, 0.0]) 
            else:
                smooth_hand_pos = (raw_hand * 0.7) + (smooth_hand_pos * 0.3)
                hand_velocity = (new_velocity * 0.6) + (hand_velocity * 0.4)
        else:
            smooth_hand_pos += hand_velocity
            hand_velocity *= 0.92 
            if state == "GRABBED": current_grip_value = 0.0 

        active_hand_pos = smooth_hand_pos 

        # Face Logic
        if res_face.face_landmarks and not game_over and grab_start_time:
            if (current_time - last_roar_time) > ROAR_COOLDOWN:
                if is_mouth_open(res_face.face_landmarks[0]):
                    last_roar_time = current_time
                    roar_active = True; roar_radius = 50
                    if roar_sound: roar_sound.play()
                    message_text = "SONIC ROAR!"
                    message_time = current_time

        # --- 2. GAMEPLAY ---
        if not game_over:
            
            # Roar
            if roar_active:
                roar_radius += 45
                pygame.draw.circle(screen, BLUE, treasure_pos.astype(int), roar_radius, 20)
                for i in range(len(threats) - 1, -1, -1):
                    if math.dist(threats[i]["pos"], treasure_pos) < roar_radius:
                        threats.pop(i); score += 100
                if roar_radius > max(WIDTH, HEIGHT): roar_active = False

            # Spawners
            if grab_start_time:
                if random.random() < (SPAWN_RATE + (0.04 * intensity)):
                    threats.append(spawn_threat(intensity))
                if random.random() < 0.01: 
                    coins.append(spawn_coin())
                score += 1

            # Grip Physics
            is_holding_signal = False
            if not tracking_lost or state == "GRABBED":
                if state == "IDLE":
                    if current_grip_value < GRAB_THRESHOLD: is_holding_signal = True
                elif state == "GRABBED":
                    if current_grip_value < DROP_THRESHOLD: is_holding_signal = True

            if state == "IDLE":
                if is_holding_signal:
                    if np.linalg.norm(active_hand_pos - treasure_pos) < BASE_GRAB_RADIUS:
                        grab_frames += 1
                        if grab_frames > 2:
                            state = "GRABBED"
                            if grab_start_time is None: grab_start_time = time.time()
                            grab_frames = 0; grace_timer_start = 0
                else: grab_frames = 0
            
            elif state == "GRABBED":
                if is_holding_signal:
                    grace_timer_start = 0
                    treasure_pos += (active_hand_pos - treasure_pos) * MOVE_SMOOTHING
                else:
                    if grace_timer_start == 0: grace_timer_start = current_time
                    if (current_time - grace_timer_start) > GRACE_PERIOD_DURATION:
                        state = "IDLE"; grace_timer_start = 0

            if state == "GRABBED": elapsed = time.time() - grab_start_time

            # Threat Physics
            for i in range(len(threats) - 1, -1, -1):
                t = threats[i]
                desired_dir = treasure_pos - t["pos"]
                dist = np.linalg.norm(desired_dir)
                if dist > 0: desired_dir /= dist
                
                turn_rate = TURN_SPEED
                if t["type"] == "JUGGERNAUT": turn_rate = 0.02
                
                current_dir = t["vel"] / np.linalg.norm(t["vel"])
                new_dir = (current_dir * (1 - turn_rate)) + (desired_dir * turn_rate)
                new_dir /= np.linalg.norm(new_dir)
                
                t["vel"] = new_dir * t["speed"]
                t["pos"] += t["vel"]
                t["angle"] = math.degrees(math.atan2(-t["vel"][1], t["vel"][0]))

                # Collision
                hit_radius = BASE_BORDER_RADIUS + (t["size"] // 2)
                if np.linalg.norm(t["pos"] - treasure_pos) < hit_radius:
                    lives -= 1; threats.pop(i)
                    message_text = "HIT!"; message_time = time.time()
                    if hit_sound: hit_sound.play()
                    if lives < 0: game_over = True; win = False
                    continue

                # Cleanup
                px, py = t["pos"]
                if px < -200 or px > WIDTH + 200 or py < -200 or py > HEIGHT + 200:
                    threats.pop(i)

            # Coin Physics
            for i in range(len(coins) - 1, -1, -1):
                c = coins[i]
                if (current_time - c["spawn_time"]) > 5.0:
                    coins.pop(i); continue
                
                if np.linalg.norm(c["pos"] - treasure_pos) < BASE_BORDER_RADIUS + 30:
                    score += 1000; message_text = "+1000!"; message_time = current_time
                    coins.pop(i)

            if grab_start_time and (time.time() - grab_start_time) >= GAME_TIME:
                game_over = True; win = True

        # --- 3. RENDER ---
        color = GREEN if state == "GRABBED" else YELLOW
        if state == "GRABBED" and grace_timer_start > 0: color = ORANGE
        pygame.draw.circle(screen, color, treasure_pos.astype(int), BASE_BORDER_RADIUS, 4)
        screen.blit(chest_img, (int(treasure_pos[0]-45), int(treasure_pos[1]-45)))

        for c in coins:
            pulse = math.sin(current_time * 10) * 5
            pygame.draw.circle(screen, GOLD, c["pos"].astype(int), 25 + pulse)
            pygame.draw.circle(screen, WHITE, c["pos"].astype(int), 25 + pulse, 2)
            screen.blit(font.render("$", True, (100, 80, 0)), (c["pos"][0]-8, c["pos"][1]-10))

        for t in threats:
            img = juggernaut_img if t["type"] == "JUGGERNAUT" else threat_img
            rotated_threat = pygame.transform.rotate(img, t.get("angle", 0))
            rect = rotated_threat.get_rect(center=(int(t["pos"][0]), int(t["pos"][1])))
            screen.blit(rotated_threat, rect.topleft)

        cursor_col = CYAN if not tracking_lost else YELLOW 
        pygame.draw.circle(screen, cursor_col, active_hand_pos.astype(int), 15, 3)

        # UI - Always Visible
        lives_txt = "♥ " * (lives + 1)
        screen.blit(font.render(f"Lives: {lives_txt}", True, RED), (20, 20))
        screen.blit(font.render(f"Score: {score}", True, GOLD), (20, 55))
        
        # [FIX] Timer Logic
        if grab_start_time:
            rem = max(0, int(GAME_TIME - (current_time - grab_start_time)))
        else:
            rem = GAME_TIME
        screen.blit(font.render(f"Time: {rem}s", True, WHITE), (20, 90))
        
        # Roar Bar
        cooldown_pct = min(1.0, (current_time - last_roar_time) / ROAR_COOLDOWN)
        bar_col = BLUE if cooldown_pct >= 1.0 else (50, 50, 100)
        pygame.draw.rect(screen, bar_col, (WIDTH//2 - 150, 20, int(300 * cooldown_pct), 20))
        pygame.draw.rect(screen, WHITE, (WIDTH//2 - 150, 20, 300, 20), 2)
        screen.blit(font.render("SONIC ROAR", True, WHITE), (WIDTH//2 - 60, 22))

        if message_text and time.time() - message_time < 1.0:
            txt = big_font.render(message_text, True, YELLOW)
            screen.blit(txt, (WIDTH//2 - txt.get_width()//2, HEIGHT//2 - 100))

        # [FIX] Start Instructions
        if not grab_start_time:
            # Flashing Text
            if int(current_time * 2) % 2 == 0:
                instr = big_font.render("GRAB CHEST TO START", True, GREEN)
                screen.blit(instr, (WIDTH//2 - instr.get_width()//2, HEIGHT//2 - 50))

        if game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT)); overlay.set_alpha(200); overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            msg = "PROTECTOR WINS!" if win else "GAME OVER"
            col = GREEN if win else RED
            txt = big_font.render(msg, True, col)
            screen.blit(txt, (WIDTH//2 - txt.get_width()//2, HEIGHT//2 - 50))
            score_final = font.render(f"Final Score: {score}", True, WHITE)
            screen.blit(score_final, (WIDTH//2 - score_final.get_width()//2, HEIGHT//2 + 50))

        pygame.display.flip()

except KeyboardInterrupt:
    cap.release()
    pygame.quit()