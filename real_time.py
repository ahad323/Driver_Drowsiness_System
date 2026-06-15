"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         DRIVER DROWSINESS DETECTION — REAL TIME v2                          ║
║                                                                              ║
║  Eye detection  : EAR gate  +  eye_model.h5 (double confirmation)           ║
║                   EAR = geometric ratio — never fooled by eye shape          ║
║                   eye_model = your trained CNN — confirms real closure       ║
║  Yawn detection : yawn_model.h5 — your trained CNN                          ║
║  Face           : MediaPipe Face Mesh — fast, accurate                      ║
║                                                                              ║
║  WHY EAR GATE:                                                               ║
║    Narrow eyes → low EAR → model thinks closed → false alarm                ║
║    Fix: only call eye_model when EAR DROPS from the person's OWN baseline   ║
║    So narrow eyes that are OPEN never trigger. Only actual closure does.     ║
║                                                                              ║
║  INSTALL: pip install opencv-python mediapipe tensorflow pygame numpy        ║
║  RUN:     python real_time.py                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import time
import sys
import os

# Arduino hardware alert (Buzzer)
ARDUINO_PORT = "COM9"
ARDUINO_BAUD = 9600
arduino = None

try:
    import serial
except ImportError:
    serial = None
    print("[WARNING] pyserial is not installed. Arduino buzzer will be disabled.")
    print("          Run: pip install pyserial")

def connect_arduino():
    global arduino
    if serial is None:
        return None
    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
        time.sleep(2)  # wait for Arduino reset
        arduino.write(b'0')
        print(f"  [OK] Arduino connected on {ARDUINO_PORT}")
        return arduino
    except Exception as e:
        arduino = None
        print(f"[WARNING] Could not connect Arduino on {ARDUINO_PORT}: {e}")
        print("          The software alarm will still work.")
        return None

def set_buzzer(on):
    if arduino is not None:
        try:
            arduino.write(b'1' if on else b'0')
        except Exception as e:
            print(f"[WARNING] Arduino write failed: {e}")


try:
    import mediapipe as mp
except ImportError:
    print("[ERROR] Run: pip install mediapipe"); sys.exit(1)
try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("[ERROR] Run: pip install tensorflow"); sys.exit(1)
try:
    import pygame
except ImportError:
    print("[ERROR] Run: pip install pygame"); sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
EYE_MODEL_PATH     = "eye_model.h5"
YAWN_MODEL_PATH    = "yawn_model.h5"
ALARM_PATH         = "alarm.mp3"
CAMERA_INDEX       = 0

# EAR gate — adapts to the person's own eye shape automatically
# Eye is only "possibly closed" if EAR drops to this fraction of their baseline
# 0.80 means: trigger if eye opens 20% less than YOUR normal open state
EAR_DROP_RATIO     = 0.60
EAR_CALIBRATE_SECS = 5.0     # seconds at startup to learn baseline (keep eyes open)

# CNN confidence — eye_model must also agree before we say "closed"
EYE_CNN_THRESHOLD  = 0.72    # closed confidence must exceed this (higher = stricter)
YAWN_CNN_THRESHOLD = 0.55

# Timing
EYE_CLOSED_SECONDS = 1.2
YAWN_SECONDS       = 1.5
ALARM_COOLDOWN     = 5.0

# Performance
SMOOTH_FRAMES      = 5
YAWN_EVERY_N       = 3       # run yawn CNN every N frames (saves CPU/FPS)
EYE_EVERY_N        = 2       # run eye CNN every N frames when EAR gate triggers


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-LOCATE FILES
# ─────────────────────────────────────────────────────────────────────────────
def find_file(primary, fallbacks, label):
    for path in [primary] + fallbacks:
        if path and os.path.exists(path):
            return path
    print(f"\n[ERROR] Cannot find {label}")
    print(f"  Searched: {[primary] + fallbacks}")
    print(f"  Place the file in the same folder as this script.\n")
    sys.exit(1)

print("\n" + "=" * 62)
print("  DRIVER DROWSINESS DETECTION v2")
print("=" * 62)

eye_path   = find_file(EYE_MODEL_PATH,  ["models/eye_model.h5",  "models/eye.h5"],  "eye_model.h5")
yawn_path  = find_file(YAWN_MODEL_PATH, ["models/yawn_model.h5", "models/yawn.h5"], "yawn_model.h5")
alarm_path = find_file(ALARM_PATH,      ["alarm.wav", "assets/alarm.mp3"],           "alarm sound")

print(f"  [OK] Eye model  : {eye_path}")
print(f"  [OK] Yawn model : {yawn_path}")
print(f"  [OK] Alarm      : {alarm_path}")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Loading models...")
eye_model  = load_model(eye_path,  compile=False)
yawn_model = load_model(yawn_path, compile=False)

EYE_H,  EYE_W  = eye_model.input_shape[1],  eye_model.input_shape[2]
YAWN_H, YAWN_W = yawn_model.input_shape[1], yawn_model.input_shape[2]
EYE_CH          = eye_model.input_shape[3]
YAWN_CH         = yawn_model.input_shape[3]
EYE_CLASSES     = eye_model.output_shape[-1]
YAWN_CLASSES    = yawn_model.output_shape[-1]

print(f"  [OK] Eye  model — input {eye_model.input_shape}  output: {EYE_CLASSES} class(es)")
print(f"  [OK] Yawn model — input {yawn_model.input_shape}  output: {YAWN_CLASSES} class(es)")


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(crop, h, w, channels):
    if crop is None or crop.size == 0:
        return None
    resized = cv2.resize(crop, (w, h))
    if channels == 1:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        arr  = gray.astype("float32") / 255.0
        arr  = np.expand_dims(arr, axis=-1)
    else:
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = rgb.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# CNN PREDICTIONS
# Keras alphabetical class ordering:
#   eyes/  → closed=0, open=1   → prob_closed = pred[0] (softmax)
#   mouth/ → no_yawn=0, yawn=1  → prob_yawn   = pred[1] (softmax)
# ─────────────────────────────────────────────────────────────────────────────
def cnn_eye_closed_prob(crop):
    inp = preprocess(crop, EYE_H, EYE_W, EYE_CH)
    if inp is None:
        return 0.0
    pred = eye_model.predict(inp, verbose=0)[0]
    if EYE_CLASSES == 1:
        return 1.0 - float(pred[0])   # sigmoid: high = open → invert
    return float(pred[0])              # softmax: index 0 = closed


def cnn_yawn_prob(crop):
    inp = preprocess(crop, YAWN_H, YAWN_W, YAWN_CH)
    if inp is None:
        return 0.0
    pred = yawn_model.predict(inp, verbose=0)[0]
    if YAWN_CLASSES == 1:
        return float(pred[0])          # sigmoid: high = yawn
    return float(pred[1])              # softmax: index 1 = yawn


# ─────────────────────────────────────────────────────────────────────────────
# EAR — Eye Aspect Ratio
# EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
# Measures eye OPENNESS as a ratio → same formula works for all eye shapes
# We compare to the person's OWN calibrated baseline, not a fixed number
# ─────────────────────────────────────────────────────────────────────────────

# MediaPipe landmark indices
LEFT_EYE_EAR  = [33, 160, 158, 133, 153, 144]   # p1,p2,p3,p4,p5,p6
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
LEFT_EYE_BOX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_BOX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX     = [61, 39, 0, 267, 291, 405, 17, 181]


def ear(landmarks, indices, W, H):
    """Compute Eye Aspect Ratio from 6 landmark points."""
    pts = np.array([
        [landmarks[i].x * W, landmarks[i].y * H]
        for i in indices
    ], dtype="float32")
    # vertical distances
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    # horizontal distance
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def get_crop_box(landmarks, indices, H, W, pad=0.35):
    pts = np.array([
        [int(landmarks[i].x * W), int(landmarks[i].y * H)]
        for i in indices
    ])
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    pw = int((x2 - x1) * pad)
    ph = int((y2 - y1) * pad)
    return (max(0, x1 - pw), max(0, y1 - ph),
            min(W - 1, x2 + pw), min(H - 1, y2 + ph))


# ─────────────────────────────────────────────────────────────────────────────
# SMOOTHING
# ─────────────────────────────────────────────────────────────────────────────
class Smoother:
    def __init__(self, size):
        self._buf  = []
        self._size = size

    def update(self, val):
        self._buf.append(float(val))
        if len(self._buf) > self._size:
            self._buf.pop(0)
        return sum(self._buf) / len(self._buf)

    def value(self):
        return sum(self._buf) / len(self._buf) if self._buf else 0.0

eye_smooth  = Smoother(SMOOTH_FRAMES)
yawn_smooth = Smoother(SMOOTH_FRAMES)


# ─────────────────────────────────────────────────────────────────────────────
# ALARM
# ─────────────────────────────────────────────────────────────────────────────
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(alarm_path)
connect_arduino()

def play_alarm():
    if not pygame.mixer.get_busy():
        alarm_sound.play()


# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE
# ─────────────────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────────────────────────────────────
FONT   = cv2.FONT_HERSHEY_SIMPLEX
RED    = (0,   0,   220)
GREEN  = (0,   200, 60)
WHITE  = (255, 255, 255)
YELLOW = (0,   200, 255)
ORANGE = (0,   140, 255)
GRAY   = (70,  70,  70)


def draw_bar(frame, x, y, w, h, value, label, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)
    fill = int(w * min(max(value, 0.0), 1.0))
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), GRAY, 1)
    cv2.putText(frame, f"{label}  {value:.0%}",
                (x + 5, y + h - 3), FONT, 0.36, WHITE, 1, cv2.LINE_AA)


def draw_hud(frame, eye_conf, yawn_conf, ear_val, ear_baseline,
             eye_closed, yawning, eye_dur, yawn_dur,
             fps, drowsy, alarming, calibrating, calib_pct):

    H, W = frame.shape[:2]

    # ── Top banner ────────────────────────────────────────────────────────────
    ov = frame.copy()
    banner_col = (0, 0, 160) if drowsy else (20, 20, 20)
    cv2.rectangle(ov, (0, 0), (W, 52), banner_col, -1)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)

    if calibrating:
        pct_text = f"Calibrating... keep eyes open  {calib_pct:.0%}"
        cv2.putText(frame, pct_text, (12, 35), FONT, 0.7, YELLOW, 2, cv2.LINE_AA)
    else:
        status     = "!  DROWSINESS DETECTED  !" if drowsy else "ALERT — DRIVING SAFE"
        status_col = (80, 80, 255) if drowsy else GREEN
        cv2.putText(frame, status, (12, 35), FONT, 0.78, status_col, 2, cv2.LINE_AA)

    cv2.putText(frame, f"FPS {fps:.0f}", (W - 85, 35), FONT, 0.58, WHITE, 1, cv2.LINE_AA)

    # ── Bottom panel ──────────────────────────────────────────────────────────
    panel_h = 130
    py = H - panel_h
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, py), (W, H), (12, 12, 12), -1)
    cv2.addWeighted(ov2, 0.80, frame, 0.20, 0, frame)
    cv2.line(frame, (0, py), (W, py), GRAY, 1)

    # Eye row
    elabel = "EYES CLOSED" if eye_closed else "EYES OPEN"
    ecol   = RED if eye_closed else GREEN
    cv2.putText(frame, elabel, (12, py + 26), FONT, 0.65, ecol, 2, cv2.LINE_AA)
    if eye_closed and eye_dur > 0:
        cv2.putText(frame, f"{eye_dur:.1f}s / {EYE_CLOSED_SECONDS:.0f}s",
                    (210, py + 26), FONT, 0.55, YELLOW, 1, cv2.LINE_AA)

    # EAR value display
    ear_ratio = ear_val / max(ear_baseline, 0.01)
    cv2.putText(frame, f"EAR {ear_val:.3f} (base {ear_baseline:.3f})",
                (W - 230, py + 26), FONT, 0.42, GRAY, 1, cv2.LINE_AA)

    # Yawn row
    ylabel = "YAWNING" if yawning else "NO YAWN"
    ycol   = ORANGE if yawning else GREEN
    cv2.putText(frame, ylabel, (12, py + 55), FONT, 0.65, ycol, 2, cv2.LINE_AA)
    if yawning and yawn_dur > 0:
        cv2.putText(frame, f"{yawn_dur:.1f}s / {YAWN_SECONDS:.0f}s",
                    (210, py + 55), FONT, 0.55, YELLOW, 1, cv2.LINE_AA)

    if alarming:
        cv2.putText(frame, "ALARM", (W - 90, py + 55), FONT, 0.65, RED, 2, cv2.LINE_AA)

    # Bars
    half = (W - 24) // 2
    draw_bar(frame, 12,        py + 68, half - 4, 16,
             eye_conf,  "Eye closed CNN", RED if eye_closed else GREEN)
    draw_bar(frame, 12 + half, py + 68, half - 4, 16,
             yawn_conf, "Yawn CNN",       ORANGE if yawning else GREEN)

    # EAR openness bar (1.0 = fully open at baseline)
    draw_bar(frame, 12, py + 90, W - 24, 16,
             ear_ratio, "Eye openness (EAR)", GREEN if ear_ratio > EAR_DROP_RATIO else RED)

    # Drowsiness score
    score = min(1.0, max(
        eye_dur / EYE_CLOSED_SECONDS,
        yawn_dur / YAWN_SECONDS
    ))
    score_col = RED if score > 0.7 else (ORANGE if score > 0.35 else GREEN)
    draw_bar(frame, 12, py + 112, W - 24, 14,
             score, "Drowsiness score", score_col)


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Opening camera...")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"\n[ERROR] Cannot open camera {CAMERA_INDEX}.")
    print("  Try CAMERA_INDEX = 1 at the top of this script.\n")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("  [OK] Camera ready.\n")
print("  *** KEEP YOUR EYES OPEN for the first 3 seconds to calibrate ***\n")
print("  Press ESC to quit.")
print("=" * 62 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────
ear_baseline      = 0.0
ear_samples       = []
calibrated        = False
calibrate_start   = time.time()

eye_closed_since  = None
yawn_since        = None
last_alarm_time   = 0.0
last_eye_conf     = 0.0
last_yawn_conf    = 0.0

frame_count       = 0
fps               = 0.0
fps_t             = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_count += 1
        now = time.time()

        if frame_count % 30 == 0:
            fps   = 30.0 / max(now - fps_t, 1e-6)
            fps_t = now

        H, W = frame.shape[:2]

        # MediaPipe
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        eye_conf   = last_eye_conf
        yawn_conf  = last_yawn_conf
        eye_closed = False
        yawning    = False
        cur_ear    = ear_baseline if ear_baseline > 0 else 0.25

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # ── Compute EAR for both eyes ─────────────────────────────────────
            left_ear  = ear(lm, LEFT_EYE_EAR,  W, H)
            right_ear = ear(lm, RIGHT_EYE_EAR, W, H)
            cur_ear   = (left_ear + right_ear) / 2.0

            # ── Calibration phase — learn this person's open-eye EAR ──────────
            calib_elapsed = now - calibrate_start
            if not calibrated:
                ear_samples.append(cur_ear)
                if calib_elapsed >= EAR_CALIBRATE_SECS:
                    ear_baseline = np.percentile(ear_samples, 70)  # 70th pct = relaxed open
                    ear_baseline = max(ear_baseline, EAR_MIN_BASELINE := 0.18)
                    calibrated   = True
                    print(f"  [OK] EAR baseline calibrated: {ear_baseline:.4f}")

            # ── EAR gate — is this eye geometrically possibly closed? ─────────
            # BOTH eyes must drop — a wink only drops one eye so it won't trigger
            left_gate  = calibrated and (left_ear  < ear_baseline * EAR_DROP_RATIO)
            right_gate = calibrated and (right_ear < ear_baseline * EAR_DROP_RATIO)
            ear_gate_triggered = left_gate and right_gate

            # ── Eye CNN — only runs when EAR gate triggers ────────────────────
            if ear_gate_triggered and (frame_count % EYE_EVERY_N == 0):
                lx1, ly1, lx2, ly2 = get_crop_box(lm, LEFT_EYE_BOX,  H, W)
                rx1, ry1, rx2, ry2 = get_crop_box(lm, RIGHT_EYE_BOX, H, W)
                prob_l = cnn_eye_closed_prob(frame[ly1:ly2, lx1:lx2])
                prob_r = cnn_eye_closed_prob(frame[ry1:ry2, rx1:rx2])
                raw    = max(prob_l, prob_r)         # worst eye wins
                eye_conf     = eye_smooth.update(raw)
                last_eye_conf = eye_conf
            elif not ear_gate_triggered:
                # EAR says eye is open → reset CNN score toward 0
                eye_conf      = eye_smooth.update(0.0)
                last_eye_conf = eye_conf

            # Eye is closed only if BOTH EAR gate AND CNN agree
            eye_closed = ear_gate_triggered and (eye_conf > EYE_CNN_THRESHOLD)

            # ── Yawn CNN ─────────────────────────────────────────────────────
            if frame_count % YAWN_EVERY_N == 0:
                mx1, my1, mx2, my2 = get_crop_box(lm, MOUTH_IDX, H, W, pad=0.2)
                raw_yawn  = cnn_yawn_prob(frame[my1:my2, mx1:mx2])
                yawn_conf = yawn_smooth.update(raw_yawn)
                last_yawn_conf = yawn_conf
            else:
                yawn_conf = last_yawn_conf

            yawning = yawn_conf > YAWN_CNN_THRESHOLD

            # ── Draw boxes ───────────────────────────────────────────────────
            eye_col   = RED if eye_closed else (ORANGE if ear_gate_triggered else GREEN)
            mouth_col = ORANGE if yawning else GREEN

            lx1, ly1, lx2, ly2 = get_crop_box(lm, LEFT_EYE_BOX,  H, W)
            rx1, ry1, rx2, ry2 = get_crop_box(lm, RIGHT_EYE_BOX, H, W)
            mx1, my1, mx2, my2 = get_crop_box(lm, MOUTH_IDX, H, W, pad=0.2)

            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), eye_col,   1)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), eye_col,   1)
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), mouth_col, 1)

        else:
            eye_smooth.update(0.0)
            yawn_smooth.update(0.0)
            last_eye_conf  = 0.0
            last_yawn_conf = 0.0
            cv2.putText(frame, "No face detected",
                        (W // 2 - 110, H // 2), FONT, 0.75, YELLOW, 2, cv2.LINE_AA)

        # ── Duration tracking ─────────────────────────────────────────────────
        if eye_closed:
            if eye_closed_since is None:
                eye_closed_since = now
        else:
            eye_closed_since = None

        if yawning:
            if yawn_since is None:
                yawn_since = now
        else:
            yawn_since = None

        eye_dur  = (now - eye_closed_since) if eye_closed_since else 0.0
        yawn_dur = (now - yawn_since)       if yawn_since       else 0.0

        # ── Drowsiness verdict ────────────────────────────────────────────────
        drowsy = (eye_dur >= EYE_CLOSED_SECONDS) or (yawn_dur >= YAWN_SECONDS)

        # ── Alarm ─────────────────────────────────────────────────────────────
        alarming = False
        if drowsy:
            set_buzzer(True)
            if (now - last_alarm_time) >= ALARM_COOLDOWN:
                play_alarm()
                last_alarm_time = now
            alarming = True
        else:
            set_buzzer(False)

        # ── HUD ───────────────────────────────────────────────────────────────
        calib_pct = min((now - calibrate_start) / EAR_CALIBRATE_SECS, 1.0)
        draw_hud(
            frame,
            eye_conf, yawn_conf,
            cur_ear, ear_baseline,
            eye_closed, yawning,
            eye_dur, yawn_dur,
            fps, drowsy, alarming,
            calibrating=not calibrated,
            calib_pct=calib_pct
        )

        cv2.imshow("Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            print("\n  ESC — shutting down.")
            break

finally:
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    set_buzzer(False)
    if arduino is not None:
        try:
            arduino.close()
        except Exception:
            pass
    pygame.mixer.quit()
    print("  Done.\n")