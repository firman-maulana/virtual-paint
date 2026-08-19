import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
 
# Inisialisasi MediaPipe Hand Tracking (Tasks API) - VIDEO Mode
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=10, # Deteksi hingga 10 telapak tangan
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)
landmarker = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def draw_landmarks(image, landmarks, connections):
    h, w, _ = image.shape
    for connection in connections:
        start_idx, end_idx = connection[0], connection[1]
        start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)
    for landmark in landmarks:
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 4, (0, 0, 255), cv2.FILLED)

def get_fingers_up(hand_landmarks):
    tipIds = [4, 8, 12, 16, 20]
    fingers = []
    # Ibu Jari
    if hand_landmarks[tipIds[0]].x > hand_landmarks[tipIds[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # 4 Jari lainnya
    for id in range(1, 5):
        if hand_landmarks[tipIds[id]].y < hand_landmarks[tipIds[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# Konfigurasi Layar & Kamera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Variabel State Canvas Global
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
undo_stack = [imgCanvas.copy()]
redo_stack = []

# MULTI-HAND TRACKING STATES
# Variabel yang disimpan per masing-masing tangan (maksimal 10 tangan)
MAX_HANDS = 10
xp = [0] * MAX_HANDS
yp = [0] * MAX_HANDS
smooth_xp = [0] * MAX_HANDS
smooth_yp = [0] * MAX_HANDS
drawColor = [(0, 255, 0)] * MAX_HANDS
is_drawing = [False] * MAX_HANDS

brushThickness = 15
eraserThickness = 80

# Konfigurasi UI Menu (Top Bar)
header_rects = [
    {"label": "RED", "rect": (50, 20, 150, 100), "color": (0, 0, 255), "type": "color"},
    {"label": "GREEN", "rect": (170, 20, 270, 100), "color": (0, 255, 0), "type": "color"},
    {"label": "BLUE", "rect": (290, 20, 390, 100), "color": (255, 0, 0), "type": "color"},
    {"label": "YELLOW", "rect": (410, 20, 510, 100), "color": (0, 255, 255), "type": "color"},
    {"label": "ERASER", "rect": (530, 20, 680, 100), "color": (0, 0, 0), "type": "eraser"},
    {"label": "UNDO", "rect": (700, 20, 850, 100), "color": (150, 150, 150), "type": "action"},
    {"label": "REDO", "rect": (870, 20, 1020, 100), "color": (150, 150, 150), "type": "action"},
    {"label": "CLEAR", "rect": (1040, 20, 1190, 100), "color": (50, 50, 200), "type": "action"}
]

def push_undo_state(canvas):
    global undo_stack, redo_stack
    undo_stack.append(canvas.copy())
    if len(undo_stack) > 10: # Batas maksimal history memori
        undo_stack.pop(0)
    redo_stack.clear()

def draw_header(img):
    cv2.rectangle(img, (0, 0), (1280, 120), (30, 30, 30), cv2.FILLED)
    for item in header_rects:
        x1, y1, x2, y2 = item["rect"]
        color = item["color"]
        if item["type"] == "color":
            cv2.rectangle(img, (x1, y1), (x2, y2), color, cv2.FILLED)
        elif item["type"] == "eraser":
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), cv2.FILLED)
            cv2.putText(img, "ERASER", (x1+15, y1+50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        else: # Action (Undo, Redo, Clear)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, cv2.FILLED)
            cv2.putText(img, item["label"], (x1+20, y1+50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Anti-bounce untuk tombol action
last_action_time = 0

print("Aplikasi Virtual Paint (Multi-Hand) Siap!")
print("Tekan 'q' untuk keluar.")

while True:
    success, img = cap.read()
    if not success:
        break
        
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    
    timestamp = int(time.time_ns() // 1_000_000)
    results = landmarker.detect_for_video(mp_image, timestamp)
    
    draw_header(img)
    
    num_detected = 0
    if results.hand_landmarks:
        num_detected = min(len(results.hand_landmarks), MAX_HANDS)
        
        for i in range(num_detected):
            hand_landmarks = results.hand_landmarks[i]
            draw_landmarks(img, hand_landmarks, HAND_CONNECTIONS)
            
            h, w, c = img.shape
            x1, y1 = int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h) # Telunjuk
            
            # Smoothing filter per masing-masing tangan (indeks i)
            if smooth_xp[i] == 0 and smooth_yp[i] == 0:
                smooth_xp[i], smooth_yp[i] = x1, y1
            else:
                smooth_xp[i] = smooth_xp[i] + 0.6 * (x1 - smooth_xp[i])
                smooth_yp[i] = smooth_yp[i] + 0.6 * (y1 - smooth_yp[i])
                
            cx, cy = int(smooth_xp[i]), int(smooth_yp[i])
            fingers = get_fingers_up(hand_landmarks)
            
            # Mode Seleksi / Hover (Telunjuk dan Tengah terangkat)
            if fingers[1] and fingers[2]:
                if is_drawing[i]:
                    # Stroke selesai
                    push_undo_state(imgCanvas)
                    is_drawing[i] = False
                    
                xp[i], yp[i] = 0, 0 # Reset posisi kursor gambar
                cv2.circle(img, (x1, y1), 15, drawColor[i], cv2.FILLED)
                
                # Cek apakah telunjuk menyentuh menu di atas
                if y1 < 120:
                    for item in header_rects:
                        rx1, ry1, rx2, ry2 = item["rect"]
                        if rx1 < x1 < rx2 and ry1 < y1 < ry2:
                            if item["type"] == "color":
                                drawColor[i] = item["color"]
                            elif item["type"] == "eraser":
                                drawColor[i] = (0, 0, 0)
                            elif item["type"] == "action" and (time.time() - last_action_time) > 1.0:
                                last_action_time = time.time()
                                if item["label"] == "UNDO":
                                    if len(undo_stack) > 1:
                                        redo_stack.append(undo_stack.pop())
                                        imgCanvas = undo_stack[-1].copy()
                                elif item["label"] == "REDO":
                                    if len(redo_stack) > 0:
                                        imgCanvas = redo_stack.pop()
                                        undo_stack.append(imgCanvas.copy())
                                elif item["label"] == "CLEAR":
                                    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                                    push_undo_state(imgCanvas)
            
            # Mode Menggambar (Hanya Telunjuk yang terangkat)
            elif fingers[1] and not fingers[2]:
                if not is_drawing[i]:
                    is_drawing[i] = True
                    
                cv2.circle(img, (cx, cy), 15, drawColor[i], cv2.FILLED)
                
                if xp[i] == 0 and yp[i] == 0:
                    xp[i], yp[i] = cx, cy
                    
                thickness = eraserThickness if drawColor[i] == (0, 0, 0) else brushThickness
                
                cv2.line(img, (xp[i], yp[i]), (cx, cy), drawColor[i], thickness)
                cv2.line(imgCanvas, (xp[i], yp[i]), (cx, cy), drawColor[i], thickness)
                
                xp[i], yp[i] = cx, cy
                
            else:
                # Mode idle (jari lain)
                if is_drawing[i]:
                    push_undo_state(imgCanvas)
                    is_drawing[i] = False
                xp[i], yp[i] = 0, 0

    # Bersihkan / Reset memory posisi tangan jika tangan hilang dari layar
    for j in range(num_detected, MAX_HANDS):
        if is_drawing[j]:
            push_undo_state(imgCanvas)
            is_drawing[j] = False
        xp[j], yp[j] = 0, 0
        smooth_xp[j], smooth_yp[j] = 0, 0

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    
    cv2.imshow("Virtual Air Drawing", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
