import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pygame
import time
import os
import threading

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
SONGS_DIR  = "saved_songs"
ALARM_FILE = "alarm.wav"

if not os.path.exists(SONGS_DIR):
    os.makedirs(SONGS_DIR)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
.status-box { padding:0.8rem; border-radius:0.5rem; margin:0.4rem 0; font-size:14px; }
.alert      { background:#ffebee; color:#c62828; }
.warning    { background:#fff3cd; color:#856404; }
.safe       { background:#e8f5e9; color:#2e7d32; }
.inactive   { background:#f5f5f5; color:#555; }
</style>
""", unsafe_allow_html=True)

st.title("🚗 Driver Drowsiness Detection System")
st.markdown("Real-time drowsiness monitoring with 2-level alert system.")

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
defaults = {
    "ear_threshold":   0.25,
    "song_secs":       5,
    "alarm_secs":      20,
    "blink_threshold": 5,
    "alarm_volume":    0.7,
    "song_volume":     0.5,
    "selected_song":   None,
    "running":         False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v



# ─────────────────────────────────────────────
# Audio Engine
# pygame.mixer.music  → plays MP3/WAV songs
# pygame.mixer.Sound  → plays WAV alarm
# Both work from the main thread (cv2 loop)
# ─────────────────────────────────────────────
class AudioEngine:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 1024)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(1)
        self._alarm_ch     = pygame.mixer.Channel(0)
        self._current_type = None

    def play_alarm(self, path, vol):
        if not path or not os.path.exists(path):
            return
        if self._current_type == "alarm":
            return
        try:
            pygame.mixer.music.stop()   # stop song if playing
            snd = pygame.mixer.Sound(path)
            snd.set_volume(vol)
            self._alarm_ch.play(snd, loops=-1)
            self._current_type = "alarm"
        except Exception as e:
            print(f"[Audio] alarm error: {e}")

    def play_song(self, path, vol):
        if not path or not os.path.exists(path):
            return
        if self._current_type == "song":
            return
        try:
            self._alarm_ch.stop()       # stop alarm if playing
            pygame.mixer.music.load(path)
            pygame.mixer.music.set_volume(vol)
            pygame.mixer.music.play(-1) # loop forever
            self._current_type = "song"
        except Exception as e:
            print(f"[Audio] song error: {e}")

    def stop(self):
        try:
            self._alarm_ch.stop()
            pygame.mixer.music.stop()
            self._current_type = None
        except Exception as e:
            print(f"[Audio] stop error: {e}")

    def current_type(self):
        return self._current_type

@st.cache_resource
def get_audio():
    return AudioEngine()

audio = get_audio()

# ─────────────────────────────────────────────
# MediaPipe
# ─────────────────────────────────────────────
@st.cache_resource
def init_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

face_mesh = init_face_mesh()

LEFT_EYE_TOP     = [159, 158, 157]
LEFT_EYE_BOTTOM  = [145, 153, 154]
LEFT_EYE_LEFT    = 33
LEFT_EYE_RIGHT   = 133
RIGHT_EYE_TOP    = [386, 385, 384]
RIGHT_EYE_BOTTOM = [374, 380, 381]
RIGHT_EYE_LEFT   = 362
RIGHT_EYE_RIGHT  = 263

def get_ear(landmarks, top_ids, bottom_ids, left_id, right_id, iw, ih):
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * iw, lm.y * ih])
    top        = np.mean([pt(i) for i in top_ids],    axis=0)
    bottom     = np.mean([pt(i) for i in bottom_ids], axis=0)
    vertical   = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(pt(left_id) - pt(right_id))
    return (vertical / horizontal) if horizontal > 0 else 0.0

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.markdown("**🎯 Detection Sensitivity**")
    st.session_state["ear_threshold"] = st.slider(
        "EAR Threshold", 0.15, 0.35,
        st.session_state["ear_threshold"], 0.01,
        help="Open eye ≈ 0.30+ | Closed eye < 0.25"
    )
    st.session_state["blink_threshold"] = st.slider(
        "Blink Ignore (frames)", 1, 10,
        st.session_state["blink_threshold"],
        help="Blinks shorter than this are ignored."
    )

    st.markdown("**⏱️ Alert Timing**")
    st.session_state["song_secs"] = st.slider(
        "Level 1 — Song starts after (secs)", 3, 60,
        st.session_state["song_secs"]
    )
    st.session_state["alarm_secs"] = st.slider(
        "Level 2 — Alarm starts after (secs)", 5, 120,
        st.session_state["alarm_secs"]
    )

    st.markdown("**🔊 Audio**")
    st.session_state["alarm_volume"] = st.slider(
        "Alarm Volume", 0.0, 1.0, st.session_state["alarm_volume"], 0.05)
    st.session_state["song_volume"] = st.slider(
        "Song Volume", 0.0, 1.0, st.session_state["song_volume"], 0.05)

    st.markdown("---")
    st.markdown("**🎵 Favourite Song**")
    uploaded = st.file_uploader("Upload MP3 or WAV", type=["mp3", "wav"])
    if uploaded:
        path = os.path.join(SONGS_DIR, uploaded.name)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"✅ Saved: {uploaded.name}")

    saved_songs = [f for f in os.listdir(SONGS_DIR) if f.endswith((".mp3", ".wav"))]
    if saved_songs:
        chosen = st.selectbox("Select song for Level 1 alert", saved_songs)
        st.session_state["selected_song"] = os.path.join(SONGS_DIR, chosen)
    else:
        st.warning("No songs found. Upload one above.")

    st.markdown("---")
    st.markdown("""
**📋 How it works**
1. Click **▶ Start Detection**
2. Tracks real seconds eyes are closed
3. Normal blinks ignored automatically

**🔔 Alert Levels**
- 🟡 **Level 1** → song plays
- 🔴 **Level 2** → alarm + WAKE UP
- ✅ Eyes open 1s → resets
""")

# ─────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📷 Live Camera Feed")
    frame_placeholder = st.empty()

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        start = st.button("▶ Start Detection", use_container_width=True)
    with btn_col2:
        stop  = st.button("⏹ Stop Detection",  use_container_width=True)

    if start:
        st.session_state["running"] = True
    if stop:
        st.session_state["running"] = False
        audio.stop()

    st.warning("**Disclaimer:** For demonstration only. Do not rely on this for actual driving safety.")

with col2:
    st.subheader("📊 Live Status")
    status_ph  = st.empty()
    level_ph   = st.empty()
    timing_ph  = st.empty()
    prog_ph    = st.empty()
    stats_ph   = st.empty()

LEVEL_LABELS = {
    0: ("🟢 Normal",              "safe"),
    1: ("🟡 Level 1: Song Alert", "warning"),
    2: ("🔴 Level 2: Alarm",      "alert"),
}

# ─────────────────────────────────────────────
# Detection Loop — runs directly in Streamlit
# Uses cv2.VideoCapture (no WebRTC needed)
# This is the most reliable approach on Windows
# ─────────────────────────────────────────────
if st.session_state["running"]:

    # Get song path directly — pygame.mixer.music handles MP3 natively
    song_path = st.session_state.get("selected_song", None)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot open camera. Make sure your webcam is connected.")
        st.session_state["running"] = False
        st.stop()

    # State variables
    closed_start     = None
    open_start       = None
    closed_secs      = 0.0
    open_secs        = 0.0
    drowsiness_level = 0
    total_frames     = 0
    drowsy_secs      = 0.0
    last_time        = time.time()

    while st.session_state["running"]:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to read from camera.")
            break

        now = time.time()
        dt  = now - last_time
        last_time = now

        ear_thr    = st.session_state.get("ear_threshold",   0.25)
        song_secs  = st.session_state.get("song_secs",       5)
        alarm_secs = st.session_state.get("alarm_secs",      20)
        blink_secs = st.session_state.get("blink_threshold", 5) * 0.033
        alarm_vol  = st.session_state.get("alarm_volume",    0.7)
        song_vol   = st.session_state.get("song_volume",     0.5)

        ih, iw = frame.shape[:2]
        total_frames += 1

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ear       = 0.0
        eye_label = "Unknown"

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left_ear  = get_ear(lm, LEFT_EYE_TOP,  LEFT_EYE_BOTTOM,
                                 LEFT_EYE_LEFT,  LEFT_EYE_RIGHT,  iw, ih)
            right_ear = get_ear(lm, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                                 RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT, iw, ih)
            ear       = (left_ear + right_ear) / 2.0
            eye_label = "Closed" if ear < ear_thr else "Open"
            eye_color = (0, 0, 255) if eye_label == "Closed" else (0, 255, 0)
            cv2.putText(frame, f"Eyes: {eye_label}  EAR: {ear:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, eye_color, 2)

            if eye_label == "Closed":
                if closed_start is None:
                    closed_start = now
                closed_secs = now - closed_start
                open_start  = None
                open_secs   = 0.0
                if closed_secs > blink_secs:
                    drowsy_secs += dt
            else:
                if open_start is None:
                    open_start = now
                open_secs   = now - open_start
                closed_start = None
                closed_secs  = 0.0

            if closed_secs >= alarm_secs:
                drowsiness_level = 2
            elif closed_secs >= song_secs:
                drowsiness_level = max(drowsiness_level, 1)

            if open_secs >= 1.0:
                drowsiness_level = 0
        else:
            cv2.putText(frame, "No face detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            if open_start is None:
                open_start = now
            open_secs   = now - open_start
            closed_start = None
            closed_secs  = 0.0
            if open_secs >= 1.0:
                drowsiness_level = 0

        # ── Audio ──────────────────────────────────────────
        if drowsiness_level == 2:
            audio.play_alarm(ALARM_FILE, alarm_vol)
        elif drowsiness_level == 1:
            audio.play_song(song_path, song_vol)
        else:
            if audio.current_type() is not None:
                audio.stop()

        # ── Visuals ────────────────────────────────────────
        LEVEL_TEXT = {0: "Normal", 1: "Level 1: Song", 2: "Level 2: Alarm"}
        if drowsiness_level == 2:
            status = "Drowsy! WAKE UP"
            cv2.rectangle(frame, (0, 0), (iw-1, ih-1), (0, 0, 255), 8)
            cv2.putText(frame, "WAKE UP!", (iw//2 - 100, ih//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
        elif drowsiness_level == 1:
            status = "Slight Drowsiness"
            cv2.rectangle(frame, (0, 0), (iw-1, ih-1), (0, 165, 255), 4)
            cv2.putText(frame, "Stay Alert!", (iw//2 - 110, ih//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 165, 255), 3)
        else:
            status = "Awake"

        s_color = (0, 255, 0) if status == "Awake" else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, s_color, 2)
        cv2.putText(frame, LEVEL_TEXT[drowsiness_level],
                    (10, ih - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(frame,
                    f"Closed: {closed_secs:.1f}s / Song:{song_secs}s / Alarm:{alarm_secs}s",
                    (10, ih - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show frame
        frame_placeholder.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True
        )

        # ── Update Live Status ──────────────────────────────
        label, css   = LEVEL_LABELS.get(drowsiness_level, ("🟢 Normal", "safe"))
        session_mins = round(total_frames / 15 / 60, 1)

        status_ph.markdown(f"""
            <div class="status-box {css}">
                <b>Status:</b> {status}<br>
                <b>Eyes:</b> {eye_label}<br>
                <b>EAR Score:</b> {round(ear, 3)}
            </div>""", unsafe_allow_html=True)

        level_ph.markdown(f"""
            <div class="status-box {css}">
                <b>Alert Level:</b> {label}
            </div>""", unsafe_allow_html=True)

        timing_ph.markdown(f"""
            <div class="status-box safe">
                <b>Eyes closed for:</b> {closed_secs:.1f}s<br>
                <b>Eyes open for:</b> {open_secs:.1f}s<br>
                <b>Song at:</b> {song_secs}s &nbsp;|&nbsp; <b>Alarm at:</b> {alarm_secs}s
            </div>""", unsafe_allow_html=True)

        prog_ph.progress(
            min(closed_secs / max(alarm_secs, 1), 1.0),
            text=f"Drowsiness: {closed_secs:.1f}s / {alarm_secs}s"
        )

        stats_ph.markdown(f"""
            <div class="status-box safe">
                <b>📈 Session Stats</b><br>
                Session: ~{session_mins} min<br>
                Drowsy time: {drowsy_secs:.1f}s<br>
                Total frames: {total_frames}
            </div>""", unsafe_allow_html=True)

    cap.release()
    audio.stop()

else:
    frame_placeholder.markdown("""
        <div style='background:#1e1e1e; color:#aaa; text-align:center;
                    padding:80px 20px; border-radius:8px; font-size:18px;'>
            📷 Camera not started<br>
            <small>Click <b>▶ Start Detection</b> to begin</small>
        </div>""", unsafe_allow_html=True)

    status_ph.markdown("""
        <div class="status-box inactive">
            ⚪ <b>Not active</b><br>
            Click <b>▶ Start Detection</b> to begin.
        </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("🚗 Driver Drowsiness Detection | Powered by MediaPipe & Streamlit")