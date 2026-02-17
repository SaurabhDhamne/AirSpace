import cv2
import numpy as np
import mediapipe as mp
import pytesseract
import time
import re
import webbrowser  # <--- Add this
import os          # <--- Add this (for opening apps like calculator)
# If needed on Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

print("----------------------------------------")
print("           AIRSPACE CONTROLS")
print("---------------------------------------")
print("☝️  Index Finger:      DRAW")
print("✌️  Index + Middle:    ERASE")
print("✊  Fist:              HOVER")
print("👍  Thumb Up:          OCR Scan")
print("✋  Open Palm:         Unlock OCR")
print("Press Q or ESC to Exit")
print("---------------------------------------")

# ===============================
# Mediapipe Setup
# ===============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
mp_draw = mp.solutions.drawing_utils

# ===============================
# Camera Setup
# ===============================
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w, _ = frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

# ===============================
# Drawing Variables
# ===============================
prev_x, prev_y = None, None
draw_color = (147, 20, 255)
eraser_color = (0, 0, 0)
thickness = 6
eraser_thickness = 40

# ===============================
# OCR Control
# ===============================
ocr_locked = False
last_ocr_time = 0
cooldown = 2

# ===============================
# Finger Detection
# ===============================
def get_finger_status(hand_landmarks):
    tips = [8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers  # [thumb, index, middle, ring, pinky]

# ===============================
# Main Loop
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        prev_x, prev_y = None, None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

            fingers = get_finger_status(hand_landmarks)

            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # ===============================
            # DRAW ☝️
            # ===============================
            if fingers[1] == 1 and fingers[2] == 0 and sum(fingers) == 1:
                cv2.putText(frame, "DRAW MODE", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

                if prev_x is None:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y),
                         draw_color, thickness)

                prev_x, prev_y = x, y

            # ===============================
            # OCR 👍
            # ===============================
            elif fingers[0] == 1 and sum(fingers) == 1:

                current_time = time.time()

                if not ocr_locked and (current_time - last_ocr_time > cooldown):

                    cv2.putText(frame, "OCR SCANNING", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)

                    thresh = cv2.adaptiveThreshold(
                        blur, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV,
                        11, 2
                    )

                    kernel = np.ones((3, 3), np.uint8)
                    processed = cv2.dilate(thresh, kernel, iterations=1)

                    cv2.imshow("OCR Preview", processed)

                    custom_config = r'--oem 3 --psm 6'
                    text = pytesseract.image_to_string(
                        processed, config=custom_config)

                    text = text.strip().upper()
                    print("OCR Raw Output:", text)

                    if text != "":

                        if "CAL" in text:
                            cv2.putText(frame, "OPENING CALCULATOR...", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            print("🚀 Action: Opening Calculator")
                            os.system('calc')
                            ocr_locked = True
                            ast_ocr_time = current_time

                        elif "GG" in text:
                            cv2.putText(frame, "OPENING GOOGLE...", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            print("🚀 Action: Opening Google")
                            webbrowser.open("https://www.google.com")
                            ocr_locked = True
                            ast_ocr_time = current_time
                        elif "YOU" in text:
                            webbrowser.open("https://www.youtube.com")  
                        elif "MOM" in text:
                            webbrowser.open("https://wa.me/7276773021")
                            print("🚀 Action: Chating with MOM")


                        elif "R" in text:
                            draw_color = (0, 0, 255)
                            print("Red color brush!")  
                        elif "B" in text:
                            draw_color = (255, 0, 0 )
                        elif "P" in text:
                            draw_color = (0,255,255)    


                        ocr_locked = True
                        last_ocr_time = current_time

                        canvas = np.zeros((h, w, 3), dtype=np.uint8)

                else:
                    cv2.putText(frame, "OCR LOCKED", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                prev_x, prev_y = None, None

            # ===============================
            # UNLOCK ✋ (All fingers up)
            # ===============================
            elif sum(fingers) == 5:
                cv2.putText(frame, "OCR UNLOCKED", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

                ocr_locked = False
                prev_x, prev_y = None, None

            # ===============================
            # ERASE ✌️
            # ===============================
            elif fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
                cv2.putText(frame, "ERASE MODE", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                cv2.circle(canvas, (x, y),
                           eraser_thickness, eraser_color, -1)

                prev_x, prev_y = None, None

            # ===============================
            # HOVER ✊
            # ===============================
            elif sum(fingers) == 0:
                cv2.putText(frame, "HOVER MODE", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

                prev_x, prev_y = None, None

            else:
                prev_x, prev_y = None, None

    # ===============================
    # Merge Canvas
    # ===============================
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Airspace Controls", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
