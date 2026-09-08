import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os

# Inisialisasi MediaPipe Hand Tracking (Tasks API) - VIDEO Mode
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=10,
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
    if hand_landmarks[tipIds[0]].x > hand_landmarks[tipIds[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
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
MAX_HANDS = 10
xp = [0] * MAX_HANDS
yp = [0] * MAX_HANDS
smooth_xp = [0] * MAX_HANDS
smooth_yp = [0] * MAX_HANDS
drawColor = [(0, 255, 0)] * MAX_HANDS
is_drawing = [False] * MAX_HANDS

brushThickness = 15
eraserThickness = 80

# Header Buttons (Warna, Undo, Redo, Camera khusus KLIK MANUAL. Eraser & Clear khusus GESTURE)
header_rects = [
    {"label": "", "rect": (30, 15, 75, 65), "color": (0, 0, 255), "type": "manual_color"},
    {"label": "", "rect": (90, 15, 135, 65), "color": (0, 255, 0), "type": "manual_color"},
    {"label": "", "rect": (150, 15, 195, 65), "color": (255, 0, 0), "type": "manual_color"},
    {"label": "", "rect": (210, 15, 255, 65), "color": (0, 255, 255), "type": "manual_color"},
    {"label": "", "rect": (270, 15, 315, 65), "color": (0, 127, 255), "type": "manual_color"},
    {"label": "", "rect": (330, 15, 375, 65), "color": (182, 89, 155), "type": "manual_color"},
    {"label": "", "rect": (390, 15, 435, 65), "color": (147, 20, 255), "type": "manual_color"},
    {"label": "ERASER", "rect": (450, 15, 545, 65), "color": (255, 255, 255), "type": "gesture_eraser"},
    {"label": "UNDO", "rect": (555, 15, 605, 65), "color": (100, 115, 130), "type": "manual_undo"},
    {"label": "REDO", "rect": (615, 15, 665, 65), "color": (100, 115, 130), "type": "manual_redo"},
    {"label": "CLEAR", "rect": (675, 15, 750, 65), "color": (40, 40, 200), "type": "gesture_clear"},
    {"label": "CAMERA", "rect": (760, 15, 825, 65), "color": (199, 132, 2), "type": "manual_camera"}
]

# State Foto / Countdown & Freeze Canvas
countdown_start_time = None
is_counting_down = False
is_photo_taken = False
frozen_photo = None

def push_undo_state(canvas):
    global undo_stack, redo_stack
    undo_stack.append(canvas.copy())
    if len(undo_stack) > 10:
        undo_stack.pop(0)
    redo_stack.clear()

def draw_undo_redo_icon(img, cx, cy, is_undo):
    color = (255, 255, 255)
    if is_undo:
        cv2.ellipse(img, (cx+2, cy+2), (9, 9), 0, 210, 70, color, 2)
        cv2.line(img, (cx-6, cy-9), (cx-6, cy-2), color, 2)
        cv2.line(img, (cx-6, cy-9), (cx+1, cy-7), color, 2)
    else:
        cv2.ellipse(img, (cx-2, cy+2), (9, 9), 0, 110, 330, color, 2)
        cv2.line(img, (cx+6, cy-9), (cx+6, cy-2), color, 2)
        cv2.line(img, (cx+6, cy-9), (cx-1, cy-7), color, 2)

def draw_checkmark_icon(img, cx, cy):
    cv2.line(img, (cx - 5, cy), (cx - 1, cy + 5), (255, 255, 255), 2)
    cv2.line(img, (cx - 1, cy + 5), (cx + 6, cy - 5), (255, 255, 255), 2)

def draw_camera_or_retake_icon(img, cx, cy, is_retake):
    if not is_retake:
        cv2.rectangle(img, (cx - 10, cy - 6), (cx + 10, cy + 8), (255, 255, 255), 2)
        cv2.circle(img, (cx, cy + 1), 4, (255, 255, 255), 2)
        cv2.rectangle(img, (cx - 4, cy - 9), (cx + 4, cy - 6), (255, 255, 255), -1)
    else:
        cv2.ellipse(img, (cx, cy), (9, 9), 0, 30, 320, (255, 255, 255), 2)
        cv2.line(img, (cx + 6, cy - 9), (cx + 10, cy - 5), (255, 255, 255), 2)
        cv2.line(img, (cx + 6, cy - 9), (cx + 10, cy - 12), (255, 255, 255), 2)

def draw_header(img, current_color):
    cv2.rectangle(img, (0, 0), (1280, 80), (20, 20, 25), cv2.FILLED)
    
    # Tombol Download Icon (Pojok Kanan Atas)
    if is_photo_taken:
        cv2.rectangle(img, (1200, 15), (1260, 65), (45, 55, 72), cv2.FILLED)
        cv2.rectangle(img, (1200, 15), (1260, 65), (200, 200, 200), 1)
        cv2.arrowedLine(img, (1230, 25), (1230, 48), (255, 255, 255), 2, tipLength=0.4)
        cv2.line(img, (1215, 52), (1245, 52), (255, 255, 255), 2)

    for item in header_rects:
        # Sembunyikan Undo & Redo saat sedang Take Foto / Freeze Foto
        if is_photo_taken and (item["type"] == "manual_undo" or item["type"] == "manual_redo"):
            continue

        x1, y1, x2, y2 = item["rect"]
        color = item["color"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if item["type"] == "manual_color":
            cv2.circle(img, (cx, cy), 18, color, cv2.FILLED)
            if current_color == color:
                cv2.circle(img, (cx, cy), 20, (255, 255, 255), 2)
                draw_checkmark_icon(img, cx, cy)
        elif item["type"] == "manual_undo" or item["type"] == "manual_redo":
            cv2.rectangle(img, (x1, y1), (x2, y2), color, cv2.FILLED)
            draw_undo_redo_icon(img, cx, cy, item["type"] == "manual_undo")
        elif item["type"] == "manual_camera":
            btn_color = (20, 120, 220) if is_photo_taken else (199, 132, 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), btn_color, cv2.FILLED)
            draw_camera_or_retake_icon(img, cx, cy, is_photo_taken)
        else:
            # Gesture Eraser & Clear
            cv2.rectangle(img, (x1, y1), (x2, y2), color, cv2.FILLED)
            text_color = (0, 0, 0) if item["type"] == "gesture_eraser" else (255, 255, 255)
            cv2.putText(img, item["label"], (x1+10, y1+33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

def save_image(img_to_save, prefix="virtual_paint"):
    filename = f"{prefix}_{int(time.time())}.png"
    cv2.imwrite(filename, img_to_save)
    print(f"File berhasil disimpan: {os.path.abspath(filename)}")
    return filename

# Callback klik mouse manual di OpenCV window
def on_mouse_click(event, x, y, flags, param):
    global is_photo_taken, frozen_photo, is_counting_down, countdown_start_time, drawColor, imgCanvas, undo_stack, redo_stack
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check Klik Header (Y: 15 - 65)
        if 15 <= y <= 65:
            # Download Foto
            if is_photo_taken and 1200 <= x <= 1260:
                save_image(frozen_photo, "download_photo_snapshot")
                return

            for item in header_rects:
                rx1, ry1, rx2, ry2 = item["rect"]
                if rx1 <= x <= rx2:
                    if item["type"] == "manual_color":
                        drawColor = [item["color"]] * MAX_HANDS
                    elif item["type"] == "manual_camera":
                        if is_photo_taken:
                            is_photo_taken = False
                            frozen_photo = None
                        else:
                            if not is_counting_down:
                                is_counting_down = True
                                countdown_start_time = time.time()
                    elif not is_photo_taken and item["type"] == "manual_undo":
                        if len(undo_stack) > 1:
                            redo_stack.append(undo_stack.pop())
                            imgCanvas = undo_stack[-1].copy()
                    elif not is_photo_taken and item["type"] == "manual_redo":
                        if len(redo_stack) > 0:
                            imgCanvas = redo_stack.pop()
                            undo_stack.append(imgCanvas.copy())

last_action_time = 0

cv2.namedWindow("Virtual Air Drawing")
cv2.setMouseCallback("Virtual Air Drawing", on_mouse_click)

print("Aplikasi Virtual Paint (Multi-Hand) Minimalis Siap!")
print("Catatan: Warna, Undo, Redo, & Kamera klik manual dengan Mouse!")
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
    
    num_detected = 0
    if results.hand_landmarks:
        num_detected = min(len(results.hand_landmarks), MAX_HANDS)
        
        for i in range(num_detected):
            hand_landmarks = results.hand_landmarks[i]
            draw_landmarks(img, hand_landmarks, HAND_CONNECTIONS)
            
            h, w, c = img.shape
            x1, y1 = int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h)
            
            if smooth_xp[i] == 0 and smooth_yp[i] == 0:
                smooth_xp[i], smooth_yp[i] = x1, y1
            else:
                smooth_xp[i] = smooth_xp[i] + 0.6 * (x1 - smooth_xp[i])
                smooth_yp[i] = smooth_yp[i] + 0.6 * (y1 - smooth_yp[i])
                
            cx, cy = int(smooth_xp[i]), int(smooth_yp[i])
            fingers = get_fingers_up(hand_landmarks)
            
            # Mode Seleksi (Telunjuk & Tengah) - HANYA UNTUK GESTURE ERASER DAN CLEAR
            if fingers[1] and fingers[2]:
                if is_drawing[i]:
                    push_undo_state(imgCanvas)
                    is_drawing[i] = False
                    
                xp[i], yp[i] = 0, 0
                cv2.circle(img, (x1, y1), 12, drawColor[i], cv2.FILLED)
                
                # Cek Hover Menu GESTURE (Hanya ERASER & CLEAR)
                if y1 < 80:
                    for item in header_rects:
                        if item["type"] in ["gesture_eraser", "gesture_clear"]:
                            rx1, ry1, rx2, ry2 = item["rect"]
                            if rx1 < x1 < rx2 and ry1 < y1 < ry2:
                                if item["type"] == "gesture_eraser":
                                    drawColor[i] = (0, 0, 0)
                                elif item["type"] == "gesture_clear" and (time.time() - last_action_time) > 1.0:
                                    last_action_time = time.time()
                                    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                                    push_undo_state(imgCanvas)
                                    is_photo_taken = False
                                    frozen_photo = None
            
            # Mode Menggambar (Hanya Telunjuk - Nonaktif jika foto sedang di-freeze)
            elif fingers[1] and not fingers[2] and not is_photo_taken:
                if not is_drawing[i]:
                    is_drawing[i] = True
                    
                cv2.circle(img, (cx, cy), 10, drawColor[i], cv2.FILLED)
                
                if xp[i] == 0 and yp[i] == 0:
                    xp[i], yp[i] = cx, cy
                    
                thickness = eraserThickness if drawColor[i] == (0, 0, 0) else brushThickness
                
                cv2.line(img, (xp[i], yp[i]), (cx, cy), drawColor[i], thickness)
                cv2.line(imgCanvas, (xp[i], yp[i]), (cx, cy), drawColor[i], thickness)
                
                xp[i], yp[i] = cx, cy
                
            else:
                if is_drawing[i]:
                    push_undo_state(imgCanvas)
                    is_drawing[i] = False
                xp[i], yp[i] = 0, 0

    for j in range(num_detected, MAX_HANDS):
        if is_drawing[j]:
            push_undo_state(imgCanvas)
            is_drawing[j] = False
        xp[j], yp[j] = 0, 0
        smooth_xp[j], smooth_yp[j] = 0, 0

    # Gabungkan Live Camera & Canvas jika belum ada foto di-freeze
    if is_photo_taken and frozen_photo is not None:
        display_img = frozen_photo.copy()
    else:
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        
        display_img = cv2.bitwise_and(img, imgInv)
        display_img = cv2.bitwise_or(display_img, imgCanvas)

    # Proses Countdown 3 Detik Take Foto (Warna Angka Putih)
    if is_counting_down:
        elapsed = time.time() - countdown_start_time
        remaining = 3 - int(elapsed)
        if remaining > 0:
            cv2.putText(display_img, str(remaining), (600, 400), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)
        else:
            is_counting_down = False
            is_photo_taken = True
            frozen_photo = display_img.copy()

    # Gambar Header UI di atas Display
    active_color = drawColor[0] if MAX_HANDS > 0 else (0, 255, 0)
    draw_header(display_img, active_color)

    cv2.imshow("Virtual Air Drawing", display_img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        if is_photo_taken:
            is_photo_taken = False
            frozen_photo = None
        else:
            if not is_counting_down:
                is_counting_down = True
                countdown_start_time = time.time()
    elif key == ord('s') and is_photo_taken:
        save_image(frozen_photo, "download_photo_snapshot")

landmarker.close()
cap.release()
cv2.destroyAllWindows()

