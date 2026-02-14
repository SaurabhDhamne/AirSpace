import cv2
import numpy as np
import av
import mediapipe as mp
import streamlit as st
import os
import urllib.request

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AirSpace Web", layout="wide")
st.title("✨ AirSpace - Gesture Drawing (Optimized Web Demo)")
st.markdown("Control the canvas using hand gestures.")

# ===============================
# SIDEBAR CONTROLS
# ===============================
st.sidebar.header("🎨 Brush Settings")

color_option = st.sidebar.selectbox(
    "Select Brush Color",
    ["Purple", "Red", "Blue", "Green"]
)

brush_thickness = st.sidebar.slider("Brush Thickness", 1, 15, 5)
eraser_thickness = st.sidebar.slider("Eraser Size", 10, 60, 30)

color_map = {
    "Purple": (147, 20, 255),
    "Red": (0, 0, 255),
    "Blue": (255, 0, 0),
    "Green": (0, 255, 0),
}

draw_color = color_map[color_option]

# ===============================
# DOWNLOAD MODEL ONCE
# ===============================
MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode


@st.cache_resource
def load_model():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1
    )
    return HandLandmarker.create_from_options(options)


landmarker = load_model()

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

        # 🔥 Reduce resolution for memory saving
        img = cv2.resize(img, (480, 360))

        h, w, _ = img.shape

        if self.canvas is None or self.canvas.shape != img.shape:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb
        )

        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            hand_lms = result.hand_landmarks[0]

            lm_list = []
            for id, lm in enumerate(hand_lms):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            if len(lm_list) != 0:
                x, y = lm_list[8][1], lm_list[8][2]

                fingers = []

                # Thumb
                fingers.append(1 if lm_list[4][1] > lm_list[3][1] else 0)

                # Other fingers
                for fid in [8, 12, 16, 20]:
                    fingers.append(
                        1 if lm_list[fid][2] < lm_list[fid - 2][2] else 0
                    )

                total_fingers = fingers.count(1)

                # DRAW
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

                # ERASE
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

        # Merge canvas
        img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, self.canvas)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ===============================
# BUTTONS
# ===============================
col1, col2 = st.columns(2)

with col1:
    start = st.button("🚀 Start Camera")

with col2:
    stop = st.button("⛔ Stop Camera")

if "run" not in st.session_state:
    st.session_state.run = False

if start:
    st.session_state.run = True

if stop:
    st.session_state.run = False
    st.experimental_rerun()

# ===============================
# WEBRTC
# ===============================
if st.session_state.run:
    rtc_configuration = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {
                    "urls": [
                        "turn:openrelay.metered.ca:80",
                        "turn:openrelay.metered.ca:443",
                        "turn:openrelay.metered.ca:443?transport=tcp",
                    ],
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
                },
            ]
        }
    )

    webrtc_streamer(
        key="airspace",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {"frameRate": 15},
            "audio": False
        },
        async_processing=True,
    )

st.info("☝️ Index = Draw | ✌️ Index+Middle = Erase")
