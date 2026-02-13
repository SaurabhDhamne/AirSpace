import cv2
import numpy as np
import av
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AirSpace Web", layout="wide")
st.title("✨ AirSpace - Gesture Drawing (Web Demo)")

st.markdown("Control the canvas using hand gestures.")

# ===============================
# SIDEBAR CONTROLS
# ===============================
st.sidebar.header("🎨 Brush Settings")

color_option = st.sidebar.selectbox(
    "Select Brush Color",
    ["Purple", "Red", "Blue", "Green"]
)

brush_thickness = st.sidebar.slider("Brush Thickness", 1, 20, 6)
eraser_thickness = st.sidebar.slider("Eraser Size", 10, 80, 40)

color_map = {
    "Purple": (147, 20, 255),
    "Red": (0, 0, 255),
    "Blue": (255, 0, 0),
    "Green": (0, 255, 0),
}

draw_color = color_map[color_option]

# ===============================
# MEDIAPIPE SETUP
# ===============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
mp_draw = mp.solutions.drawing_utils

# ===============================
# VIDEO PROCESSOR
# ===============================
class VideoProcessor:
    def __init__(self):
        self.canvas = None
        self.prev_x = None
        self.prev_y = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        if self.canvas is None or self.canvas.shape != img.shape:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

                lm_list = []
                for id, lm in enumerate(hand_lms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])

                if len(lm_list) != 0:
                    x, y = lm_list[8][1], lm_list[8][2]

                    fingers = []

                    # Thumb (compare x coordinates correctly)
                    if lm_list[4][1] > lm_list[3][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    # Other fingers
                    for id in [8, 12, 16, 20]:
                        if lm_list[id][2] < lm_list[id - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    total_fingers = fingers.count(1)

                    # ===============================
                    # DRAW MODE (Index only)
                    # ===============================
                    if fingers[1] == 1 and total_fingers == 1:
                        if self.prev_x is None:
                            self.prev_x, self.prev_y = x, y

                        cv2.line(
                            self.canvas,
                            (self.prev_x, self.prev_y),
                            (x, y),
                            draw_color,
                            brush_thickness
                        )
                        self.prev_x, self.prev_y = x, y

                    # ===============================
                    # ERASE MODE (Index + Middle)
                    # ===============================
                    elif fingers[1] == 1 and fingers[2] == 1 and total_fingers == 2:
                        cv2.circle(
                            self.canvas,
                            (x, y),
                            eraser_thickness,
                            (0, 0, 0),
                            -1
                        )
                        self.prev_x, self.prev_y = None, None

                    else:
                        self.prev_x, self.prev_y = None, None

        # ===============================
        # MERGE CANVAS
        # ===============================
        img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, self.canvas)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===============================
# BUTTON CONTROLS
# ===============================
start = st.button("🚀 Start Camera")
clear = st.button("🧹 Clear Canvas")

if "run" not in st.session_state:
    st.session_state.run = False

if start:
    st.session_state.run = True

if clear:
    st.session_state.run = False
    st.experimental_rerun()

# ===============================
# WEBRTC STREAM
# ===============================
if st.session_state.run:
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="airspace",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

st.info("☝️ Index = Draw | ✌️ Index+Middle = Erase")
