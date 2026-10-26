"""
Sign Language-Based Intelligent Robot Control
---------------------------------------------
Webcam hand gestures (MediaPipe) -> control a virtual humanoid robot (Pygame).

Gestures (show ONE hand to the camera):
    Index finger pointing UP          -> FORWARD
    Index finger pointing LEFT        -> TURN LEFT
    Index finger pointing RIGHT       -> TURN RIGHT
    Fist or open palm                 -> STOP
    Pinch (thumb + index touching)    -> PICK UP nearest object
    Peace sign (index + middle up)    -> DROP the object

Keyboard fallback (works even without a webcam):
    Arrow keys = move/turn, P = pick, D = drop, ESC = quit

Install:  pip install opencv-python mediapipe pygame
Run:      python sign_robot.py
"""

import math
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import pygame
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

W, H = 960, 640          # window size
HORIZON = 230            # y where the wall meets the floor
CAM_W, CAM_H = 256, 192  # webcam preview size inside the window

MOVE_SPEED = 170         # pixels per second
TURN_SPEED = 1.9         # radians per second
PICK_RANGE = 95          # how close the robot must be to an object

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# --------------------------------------------------------------------------
# Model file
# --------------------------------------------------------------------------
def ensure_model():
    """Download hand_landmarker.task if it is missing."""
    if MODEL_PATH.exists():
        return
    print("hand_landmarker.task not found. Downloading (about 8 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
        print("Model downloaded.")
    except Exception as exc:
        print("Could not download the model:", exc)
        print("Download it manually from:\n", MODEL_URL)
        print("and save it next to sign_robot.py as hand_landmarker.task")
        sys.exit(1)


# --------------------------------------------------------------------------
# Gesture recognition
# --------------------------------------------------------------------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_up(pts, tip, pip):
    """A finger is 'up' when its tip is farther from the wrist than its middle joint."""
    return dist(pts[tip], pts[0]) > dist(pts[pip], pts[0]) * 1.05


def classify(pts):
    """pts = list of 21 (x, y) pixel points. Returns a gesture label."""
    hand_size = dist(pts[0], pts[9]) + 1e-6

    index = finger_up(pts, 8, 6)
    middle = finger_up(pts, 12, 10)
    ring = finger_up(pts, 16, 14)
    pinky = finger_up(pts, 20, 18)
    count = sum([index, middle, ring, pinky])

    # Pinch: thumb tip and index tip very close
    if dist(pts[4], pts[8]) < 0.32 * hand_size:
        return "PICK"

    # Peace sign: index + middle up, ring + pinky down
    if index and middle and not ring and not pinky:
        return "DROP"

    # Fist or open palm
    if count == 0 or count == 4:
        return "STOP"

    # Only index finger up: direction decides forward / left / right
    if index and not middle and not ring and not pinky:
        vx = pts[8][0] - pts[5][0]
        vy = pts[8][1] - pts[5][1]
        length = math.hypot(vx, vy) + 1e-6
        if vx / length > 0.55:
            return "RIGHT"
        if vx / length < -0.55:
            return "LEFT"
        return "FORWARD"

    return "NONE"


# --------------------------------------------------------------------------
# Room background (drawn once)
# --------------------------------------------------------------------------
def draw_plant(s, x, y, scale=1.0):
    pot_w, pot_h = int(34 * scale), int(30 * scale)
    pygame.draw.polygon(
        s, (160, 82, 45),
        [(x - pot_w // 2, y - pot_h), (x + pot_w // 2, y - pot_h),
         (x + pot_w // 2 - 5, y), (x - pot_w // 2 + 5, y)],
    )
    for dx, dy, r in [(0, -62, 20), (-16, -48, 16), (16, -48, 16), (0, -46, 18), (-8, -78, 13), (9, -76, 12)]:
        pygame.draw.circle(s, (46, 125, 50), (x + int(dx * scale), y + int(dy * scale)), int(r * scale))
    pygame.draw.circle(s, (76, 160, 80), (x - int(4 * scale), y - int(60 * scale)), int(8 * scale))


def build_background():
    bg = pygame.Surface((W, H))

    # Wall with soft vertical gradient
    for y in range(HORIZON):
        t = y / HORIZON
        c = (int(226 - 22 * t), int(216 - 20 * t), int(196 - 22 * t))
        pygame.draw.line(bg, c, (0, y), (W, y))

    # Floor with planks and perspective shading
    for y in range(HORIZON, H):
        t = (y - HORIZON) / (H - HORIZON)
        c = (int(176 - 36 * t), int(128 - 30 * t), int(86 - 24 * t))
        pygame.draw.line(bg, c, (0, y), (W, y))
    vanish = (W // 2, HORIZON - 120)
    for i in range(-14, 15):
        x_bottom = W // 2 + i * 120
        # line from the floor edge at the horizon toward the bottom
        x_top = W // 2 + (x_bottom - W // 2) * 0.25
        pygame.draw.line(bg, (120, 82, 52), (int(x_top), HORIZON), (x_bottom, H), 1)
    for k in range(1, 9):
        y = HORIZON + int((k / 8) ** 1.6 * (H - HORIZON))
        pygame.draw.line(bg, (128, 88, 56), (0, y), (W, y), 1)

    # Baseboard
    pygame.draw.rect(bg, (245, 240, 230), (0, HORIZON - 10, W, 12))
    pygame.draw.line(bg, (180, 170, 155), (0, HORIZON + 2), (W, HORIZON + 2), 2)

    # Window with sky, sun and frame
    win = pygame.Rect(90, 40, 210, 140)
    pygame.draw.rect(bg, (120, 180, 235), win)
    pygame.draw.circle(bg, (255, 236, 140), (win.x + 150, win.y + 40), 20)
    pygame.draw.ellipse(bg, (255, 255, 255), (win.x + 20, win.y + 60, 70, 26))
    pygame.draw.ellipse(bg, (255, 255, 255), (win.x + 45, win.y + 50, 60, 28))
    pygame.draw.rect(bg, (250, 250, 250), win, 8)
    pygame.draw.line(bg, (250, 250, 250), (win.centerx, win.y), (win.centerx, win.bottom), 6)
    pygame.draw.line(bg, (250, 250, 250), (win.x, win.centery), (win.right, win.centery), 6)
    pygame.draw.rect(bg, (215, 205, 190), (win.x - 10, win.bottom, win.w + 20, 10))

    # Shelves with books
    for sy in (70, 135):
        pygame.draw.rect(bg, (110, 72, 40), (620, sy, 240, 10))
        bx = 628
        colors = [(200, 60, 60), (60, 110, 200), (70, 160, 90), (230, 180, 60), (150, 80, 170), (60, 60, 60)]
        for i in range(8):
            bw = 16 + (i * 7) % 10
            bh = 36 + (i * 13) % 22
            pygame.draw.rect(bg, colors[i % len(colors)], (bx, sy - bh, bw, bh))
            bx += bw + 3
    pygame.draw.rect(bg, (110, 72, 40), (620, 70, 8, 80))
    pygame.draw.rect(bg, (110, 72, 40), (852, 70, 8, 80))

    # Wall clock
    pygame.draw.circle(bg, (60, 60, 60), (470, 70), 30)
    pygame.draw.circle(bg, (250, 250, 250), (470, 70), 26)
    pygame.draw.line(bg, (40, 40, 40), (470, 70), (470, 52), 3)
    pygame.draw.line(bg, (40, 40, 40), (470, 70), (483, 74), 3)

    # Desk with computer
    desk_top = pygame.Rect(330, HORIZON - 40, 250, 16)
    pygame.draw.rect(bg, (140, 96, 58), desk_top)
    pygame.draw.rect(bg, (110, 72, 40), (desk_top.x + 8, desk_top.bottom, 12, 60))
    pygame.draw.rect(bg, (110, 72, 40), (desk_top.right - 20, desk_top.bottom, 12, 60))
    pygame.draw.rect(bg, (30, 30, 36), (desk_top.x + 70, desk_top.y - 78, 110, 70), border_radius=4)
    pygame.draw.rect(bg, (70, 140, 210), (desk_top.x + 76, desk_top.y - 72, 98, 58))
    pygame.draw.rect(bg, (30, 30, 36), (desk_top.x + 118, desk_top.y - 8, 14, 8))
    pygame.draw.rect(bg, (210, 210, 215), (desk_top.x + 78, desk_top.y - 6, 94, 6), border_radius=2)
    pygame.draw.rect(bg, (230, 230, 235), (desk_top.x + 190, desk_top.y - 14, 22, 14), border_radius=3)

    # Chair next to the desk
    pygame.draw.rect(bg, (160, 50, 50), (270, HORIZON - 70, 12, 70))
    pygame.draw.rect(bg, (160, 50, 50), (264, HORIZON - 20, 50, 12), border_radius=3)
    pygame.draw.rect(bg, (60, 40, 30), (268, HORIZON - 8, 6, 40))
    pygame.draw.rect(bg, (60, 40, 30), (304, HORIZON - 8, 6, 40))

    # Side table (right) with a lamp
    pygame.draw.rect(bg, (140, 96, 58), (800, HORIZON - 10, 120, 14))
    pygame.draw.rect(bg, (110, 72, 40), (810, HORIZON + 4, 10, 56))
    pygame.draw.rect(bg, (110, 72, 40), (900, HORIZON + 4, 10, 56))
    pygame.draw.rect(bg, (90, 90, 90), (855, HORIZON - 40, 6, 30))
    pygame.draw.polygon(bg, (250, 220, 120), [(838, HORIZON - 40), (878, HORIZON - 40), (868, HORIZON - 70), (848, HORIZON - 70)])

    # Plants
    draw_plant(bg, 40, HORIZON + 30, 1.1)
    draw_plant(bg, 930, HORIZON + 120, 1.3)

    # Rug under the middle of the room
    rug = pygame.Rect(250, 380, 420, 170)
    pygame.draw.ellipse(bg, (120, 60, 70), rug)
    pygame.draw.ellipse(bg, (170, 100, 100), rug.inflate(-40, -30), 4)

    return bg


# --------------------------------------------------------------------------
# Robot and world objects
# --------------------------------------------------------------------------
def depth_scale(y):
    return 0.55 + 0.75 * ((y - HORIZON) / (H - HORIZON))


class Robot:
    def __init__(self):
        self.x = W / 2
        self.y = 470.0
        self.heading = 0.0   # 0 = away from camera, positive = clockwise (to the right)
        self.phase = 0.0
        self.moving = False
        self.held = None

    def update(self, cmd, dt):
        self.moving = False
        if cmd == "FORWARD":
            self.x += math.sin(self.heading) * MOVE_SPEED * dt
            self.y -= math.cos(self.heading) * MOVE_SPEED * dt * 0.6
            self.moving = True
        elif cmd == "LEFT":
            self.heading -= TURN_SPEED * dt
        elif cmd == "RIGHT":
            self.heading += TURN_SPEED * dt

        self.x = max(40, min(W - 40, self.x))
        self.y = max(HORIZON + 40, min(H - 20, self.y))
        if self.moving:
            self.phase += dt * 9


DROP_ZONE = pygame.Rect(740, 470, 180, 100)


def new_objects():
    return [
        {"x": 200.0, "y": 520.0, "color": (220, 60, 60), "done": False},
        {"x": 420.0, "y": 340.0, "color": (60, 120, 220), "done": False},
        {"x": 640.0, "y": 410.0, "color": (240, 190, 40), "done": False},
    ]


def draw_box(surf, x, y, color, size):
    r = pygame.Rect(0, 0, size, size)
    r.midbottom = (int(x), int(y))
    pygame.draw.rect(surf, (60, 50, 45), r.move(4, 3), border_radius=3)
    pygame.draw.rect(surf, color, r, border_radius=3)
    pygame.draw.rect(surf, (255, 255, 255), (r.x + 3, r.y + 3, size // 3, size // 5), border_radius=2)
    pygame.draw.rect(surf, (30, 30, 30), r, 2, border_radius=3)


def draw_robot(surf, r):
    s = depth_scale(r.y)
    x, y = int(r.x), int(r.y)

    def sc(v):
        return int(v * s)

    # Shadow and heading arrow on the floor
    pygame.draw.ellipse(surf, (70, 50, 38), (x - sc(34), y - sc(8), sc(68), sc(16)))
    ax = x + math.sin(r.heading) * sc(60)
    ay = y - math.cos(r.heading) * sc(26)
    pygame.draw.line(surf, (255, 235, 90), (x, y), (ax, ay), 4)
    pygame.draw.circle(surf, (255, 235, 90), (int(ax), int(ay)), 6)

    swing = math.sin(r.phase) * 10 * s if r.moving else 0.0
    leg_h = sc(42)
    torso_h = sc(62)
    torso_top = y - leg_h - torso_h

    # Legs
    leg_w = sc(15)
    lift_l = max(0, swing) * 0.7
    lift_r = max(0, -swing) * 0.7
    pygame.draw.rect(surf, (70, 80, 100), (x - sc(22), y - leg_h - lift_l, leg_w, leg_h), border_radius=4)
    pygame.draw.rect(surf, (70, 80, 100), (x + sc(7), y - leg_h - lift_r, leg_w, leg_h), border_radius=4)
    pygame.draw.rect(surf, (35, 40, 50), (x - sc(25), y - sc(8) - lift_l, sc(21), sc(8)), border_radius=3)
    pygame.draw.rect(surf, (35, 40, 50), (x + sc(4), y - sc(8) - lift_r, sc(21), sc(8)), border_radius=3)

    # Arms (drawn behind torso)
    sh_l = (x - sc(30), torso_top + sc(12))
    sh_r = (x + sc(30), torso_top + sc(12))
    if r.held is not None:
        hand_l = (x - sc(18), torso_top + sc(42))
        hand_r = (x + sc(18), torso_top + sc(42))
    else:
        hand_l = (x - sc(36), torso_top + sc(54) + swing)
        hand_r = (x + sc(36), torso_top + sc(54) - swing)
    pygame.draw.line(surf, (190, 195, 205), sh_l, hand_l, sc(11))
    pygame.draw.line(surf, (190, 195, 205), sh_r, hand_r, sc(11))
    pygame.draw.circle(surf, (230, 90, 70), hand_l, sc(7))
    pygame.draw.circle(surf, (230, 90, 70), hand_r, sc(7))

    # Torso
    torso = pygame.Rect(x - sc(27), torso_top, sc(54), torso_h)
    pygame.draw.rect(surf, (60, 120, 210), torso, border_radius=sc(10))
    pygame.draw.rect(surf, (30, 60, 120), torso, 2, border_radius=sc(10))
    pygame.draw.circle(surf, (120, 240, 255), (x, torso_top + sc(22)), sc(8))
    pygame.draw.rect(surf, (30, 60, 120), (x - sc(18), torso_top + sc(38), sc(36), sc(5)), border_radius=2)

    # Head
    head = pygame.Rect(x - sc(21), torso_top - sc(38), sc(42), sc(34))
    pygame.draw.rect(surf, (200, 205, 215), head, border_radius=sc(9))
    pygame.draw.rect(surf, (90, 95, 110), head, 2, border_radius=sc(9))
    pygame.draw.line(surf, (90, 95, 110), (x, head.y), (x, head.y - sc(10)), 3)
    pygame.draw.circle(surf, (230, 60, 60), (x, head.y - sc(11)), sc(4))

    facing_away = math.cos(r.heading) > 0.25
    if facing_away:
        for i in range(3):
            pygame.draw.line(surf, (120, 125, 140),
                             (head.x + sc(9), head.y + sc(10 + i * 6)),
                             (head.right - sc(9), head.y + sc(10 + i * 6)), 2)
    else:
        off = int(math.sin(r.heading) * sc(8))
        for ex in (-sc(9), sc(9)):
            pygame.draw.circle(surf, (20, 30, 50), (x + ex + off, head.y + sc(15)), sc(6))
            pygame.draw.circle(surf, (110, 230, 255), (x + ex + off, head.y + sc(15)), sc(3))

    # Held object in front of the chest
    if r.held is not None:
        draw_box(surf, x, torso_top + sc(50), r.held["color"], sc(24))


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------
def do_pick(robot, objects):
    if robot.held is not None:
        return "Already holding an object"
    best, best_d = None, 1e9
    for o in objects:
        if o["done"]:
            continue
        d = math.hypot(o["x"] - robot.x, (o["y"] - robot.y) * 1.6)
        if d < best_d:
            best, best_d = o, d
    if best is not None and best_d <= PICK_RANGE:
        robot.held = best
        objects.remove(best)
        return "Picked up!"
    return "Nothing close enough to pick"


def do_drop(robot, objects):
    if robot.held is None:
        return "Nothing to drop", 0
    obj = robot.held
    robot.held = None
    obj["x"], obj["y"] = robot.x, robot.y + 6
    scored = 0
    if DROP_ZONE.collidepoint(obj["x"], obj["y"]):
        obj["done"] = True
        scored = 1
    objects.append(obj)
    return ("Dropped in the zone!" if scored else "Dropped"), scored


# --------------------------------------------------------------------------
# Camera + hand tracking
# --------------------------------------------------------------------------
def open_camera():
    for index in (0, 1):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
        cap.release()
    return None


def create_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def read_gesture(cap, landmarker, last_ts):
    """Returns (label, preview_surface, new_timestamp)."""
    ok, frame = cap.read()
    if not ok:
        return "NO CAMERA", None, last_ts

    frame = cv2.flip(frame, 1)  # mirror so left/right feel natural
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    ts = max(int(time.time() * 1000), last_ts + 1)
    result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

    label = "NO HAND"
    if result.hand_landmarks:
        pts = [(lm.x * w, lm.y * h) for lm in result.hand_landmarks[0]]
        label = classify(pts)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])), (0, 255, 120), 2)
        for p in pts:
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, (0, 120, 255), -1)

    small = cv2.resize(frame, (CAM_W, CAM_H))
    small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    surface = pygame.image.frombuffer(small.tobytes(), (CAM_W, CAM_H), "RGB")
    return label, surface, ts


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ensure_model()

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Sign Language Robot Control")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20, bold=True)
    small_font = pygame.font.SysFont("consolas", 15)

    background = build_background()
    robot = Robot()
    objects = new_objects()
    score = 0
    message = "Show your hand to the camera"
    message_until = 0.0

    cap = open_camera()
    landmarker = create_landmarker() if cap is not None else None
    if cap is None:
        message = "No webcam found - use keyboard controls"
    last_ts = 0
    history = deque(maxlen=4)
    stable = "NO HAND"
    prev_stable = "NO HAND"
    last_action_time = 0.0
    preview = None

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        now = time.time()
        key_pick = key_drop = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    key_pick = True
                elif event.key == pygame.K_d:
                    key_drop = True

        # ---- gesture input ----
        if cap is not None:
            label, preview, last_ts = read_gesture(cap, landmarker, last_ts)
            history.append(label)
            if len(history) == history.maxlen and len(set(history)) == 1:
                stable = label
        else:
            stable = "NO HAND"

        # ---- keyboard fallback ----
        keys = pygame.key.get_pressed()
        cmd = None
        if keys[pygame.K_UP]:
            cmd = "FORWARD"
        elif keys[pygame.K_LEFT]:
            cmd = "LEFT"
        elif keys[pygame.K_RIGHT]:
            cmd = "RIGHT"
        elif stable in ("FORWARD", "LEFT", "RIGHT"):
            cmd = stable

        # ---- pick / drop (triggered once per gesture) ----
        want_pick = key_pick or (stable == "PICK" and prev_stable != "PICK")
        want_drop = key_drop or (stable == "DROP" and prev_stable != "DROP")
        if (want_pick or want_drop) and now - last_action_time > 0.6:
            last_action_time = now
            if want_pick:
                message = do_pick(robot, objects)
            else:
                message, gained = do_drop(robot, objects)
                score += gained
            message_until = now + 2.0
        prev_stable = stable

        robot.update(cmd, dt)

        # ---- draw ----
        screen.blit(background, (0, 0))

        # drop zone (on the floor)
        zone_surf = pygame.Surface(DROP_ZONE.size, pygame.SRCALPHA)
        zone_surf.fill((60, 200, 90, 110))
        screen.blit(zone_surf, DROP_ZONE.topleft)
        pygame.draw.rect(screen, (30, 140, 60), DROP_ZONE, 3)
        screen.blit(small_font.render("DROP ZONE", True, (20, 90, 40)), (DROP_ZONE.x + 38, DROP_ZONE.y + 38))

        # draw objects and robot sorted by depth so nothing overlaps wrongly
        drawables = [("obj", o, o["y"]) for o in objects] + [("robot", robot, robot.y)]
        drawables.sort(key=lambda d: d[2])
        for kind, item, _ in drawables:
            if kind == "obj":
                draw_box(screen, item["x"], item["y"], item["color"], int(26 * depth_scale(item["y"])))
            else:
                draw_robot(screen, item)

        # ---- HUD ----
        panel = pygame.Surface((330, 118), pygame.SRCALPHA)
        panel.fill((20, 24, 34, 200))
        screen.blit(panel, (10, 10))
        screen.blit(font.render(f"Gesture : {stable}", True, (120, 255, 170)), (22, 20))
        screen.blit(font.render(f"Holding : {'yes' if robot.held else 'no'}", True, (255, 255, 255)), (22, 46))
        screen.blit(font.render(f"Score   : {score}", True, (255, 220, 100)), (22, 72))
        screen.blit(small_font.render("ESC quit | Arrows move | P pick | D drop", True, (190, 200, 215)), (22, 100))

        if now < message_until:
            msg = font.render(message, True, (255, 255, 255))
            box = msg.get_rect(midbottom=(W // 2, H - 14)).inflate(24, 12)
            pygame.draw.rect(screen, (20, 24, 34), box, border_radius=8)
            screen.blit(msg, msg.get_rect(center=box.center))

        if preview is not None:
            screen.blit(preview, (W - CAM_W - 10, 10))
            pygame.draw.rect(screen, (255, 255, 255), (W - CAM_W - 10, 10, CAM_W, CAM_H), 2)

        guide = [
            "Index up = forward", "Index left/right = turn",
            "Fist / palm = stop", "Pinch = pick", "Peace sign = drop",
        ]
        for i, line in enumerate(guide):
            screen.blit(small_font.render(line, True, (30, 30, 40)), (W - CAM_W - 10, CAM_H + 18 + i * 20))

        pygame.display.flip()

    if cap is not None:
        cap.release()
    if landmarker is not None:
        landmarker.close()
    pygame.quit()


if __name__ == "__main__":
    main()
