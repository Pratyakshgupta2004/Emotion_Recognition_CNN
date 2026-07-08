
import os
import time
from collections import deque, Counter

import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "models/emotion_model.keras"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
OUTPUT_DIR = "output"
CAM_WIDTH, CAM_HEIGHT = 640, 480
DISPLAY_WIDTH, DISPLAY_HEIGHT = 960, 720  
PANEL_WIDTH = 300  
FACE_INPUT_SIZE = (48, 48)
MIN_FACE_SIZE = (60, 60)
SMOOTHING_WINDOW = 8            
CONFIDENCE_LEVELS = [0, 40, 60, 80]  

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

EMOTION_COLORS = {
    "Angry": (0, 0, 255),
    "Disgust": (128, 0, 128),
    "Fear": (255, 0, 255),
    "Happy": (0, 255, 0),
    "Sad": (255, 0, 0),
    "Surprise": (0, 255, 255),
    "Neutral": (255, 255, 0),
    "Uncertain": (180, 180, 180),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at '{MODEL_PATH}'")

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.")

face_detector = cv2.CascadeClassifier(CASCADE_PATH)
if face_detector.empty():
    raise RuntimeError("Failed to load Haar cascade classifier")

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

if not camera.isOpened():
    raise RuntimeError("Could not open camera (index 0). Check connection/permissions.")


prev_time = time.time()
smoothed_fps = 0.0
show_help = True
show_fps = True
mirror = True
conf_level_idx = 1 
recording = False
video_writer = None
session_emotion_log = deque(maxlen=2000)  


face_histories = {}


last_prediction_vector = None
last_display_emotion = None


def bucket_key(x, y, w, h, grid=40):
    """Coarse spatial bucket so the same physical face keeps its history
    across frames even with small jitter in detected coordinates."""
    return (x // grid, y // grid)


def draw_help_overlay(frame):
    lines = [
        "S: Screenshot   R: Record   M: Mirror",
        "H: Help   F: FPS   C: Confidence thresh",
        "Q / ESC: Quit",
    ]
    y0 = frame.shape[0] - 20 - (len(lines) - 1) * 22
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0 - 15), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_session_summary(frame):
    """Small live bar of the most common emotions this session, top-right."""
    if not session_emotion_log:
        return
    counts = Counter(session_emotion_log)
    total = sum(counts.values())
    x0 = frame.shape[1] - 180
    y0 = 15
    cv2.putText(frame, "Session mix:", (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, (emo, cnt) in enumerate(counts.most_common(4)):
        pct = cnt / total * 100
        bar_len = int(pct / 100 * 100)
        y = y0 + 18 + i * 18
        color = EMOTION_COLORS.get(emo, (200, 200, 200))
        cv2.rectangle(frame, (x0, y - 8), (x0 + bar_len, y), color, -1)
        cv2.putText(frame, f"{emo[:3]} {pct:.0f}%", (x0 + 105, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def draw_confidence_panel(height, prediction_vector, display_emotion, fps_value):
    """Builds the dark side panel with a title, current emotion + FPS readout,
    and a horizontal bar for every emotion class's confidence — mirrors the
    'Emotion Confidence' mockup layout."""
    panel = np.full((height, PANEL_WIDTH, 3), (24, 24, 24), dtype=np.uint8)

    cv2.putText(panel, "Emotion Recognition", (14, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(panel, (14, 42), (PANEL_WIDTH - 14, 42), (70, 70, 70), 1)

    y = 72
    if display_emotion is not None:
        color = EMOTION_COLORS.get(display_emotion, (255, 255, 255))
        cv2.putText(panel, display_emotion, (14, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2, cv2.LINE_AA)
    else:
        cv2.putText(panel, "No face detected", (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)
    y += 26
    cv2.putText(panel, f"FPS: {int(fps_value)}", (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    y += 30
    cv2.putText(panel, "Emotion Confidence", (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    y += 14

    bar_x0 = 100
    bar_max_w = PANEL_WIDTH - bar_x0 - 55
    row_h = 32

    for i, label in enumerate(EMOTION_LABELS):
        row_y = y + 20 + i * row_h
        pct = float(prediction_vector[i] * 100) if prediction_vector is not None else 0.0
        bar_w = int(bar_max_w * pct / 100)
        color = EMOTION_COLORS.get(label, (200, 200, 200))

        cv2.putText(panel, label, (14, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (bar_x0, row_y - 12), (bar_x0 + bar_max_w, row_y - 2),
                      (55, 55, 55), 1)
        if bar_w > 0:
            cv2.rectangle(panel, (bar_x0, row_y - 12), (bar_x0 + bar_w, row_y - 2),
                          color, -1)
        cv2.putText(panel, f"{pct:.0f}%", (bar_x0 + bar_max_w + 8, row_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

    return panel


print("Starting webcam feed. Press 'H' any time to toggle the help overlay.")

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to read frame from camera. Exiting.")
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray) 
        faces = face_detector.detectMultiScale(
            gray_eq,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=MIN_FACE_SIZE,
        )

        min_confidence = CONFIDENCE_LEVELS[conf_level_idx]

        last_prediction_vector = None
        last_display_emotion = None

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, FACE_INPUT_SIZE)
            face_roi = face_roi.astype("float32") / 255.0
            face_roi = np.expand_dims(face_roi, axis=-1)
            face_roi = np.expand_dims(face_roi, axis=0)

            prediction = model.predict(face_roi, verbose=0)[0]
            raw_idx = int(np.argmax(prediction))
            raw_emotion = EMOTION_LABELS[raw_idx]
            confidence = float(np.max(prediction) * 100)

            
            key = bucket_key(x, y, w, h)
            history = face_histories.setdefault(key, deque(maxlen=SMOOTHING_WINDOW))
            history.append(raw_emotion)
            smoothed_emotion = Counter(history).most_common(1)[0][0]

            if confidence < min_confidence:
                display_emotion = "Uncertain"
            else:
                display_emotion = smoothed_emotion
                session_emotion_log.append(smoothed_emotion)

          
            if last_prediction_vector is None:
                last_prediction_vector = prediction
                last_display_emotion = display_emotion

            color = EMOTION_COLORS.get(display_emotion, (255, 255, 255))

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.rectangle(frame, (x, y - 35), (x + w, y), color, -1)
            cv2.putText(
                frame,
                f"{display_emotion} ({confidence:.1f}%)",
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        if len(face_histories) > 50:
            face_histories.clear()

        current_time = time.time()
        instant_fps = 1 / max(current_time - prev_time, 1e-6)
        smoothed_fps = 0.9 * smoothed_fps + 0.1 * instant_fps if smoothed_fps else instant_fps
        prev_time = current_time

        if show_fps:
            cv2.putText(frame, f"FPS : {int(smoothed_fps)}", (10, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        if recording:
            cv2.circle(frame, (frame.shape[1] - 20, 20), 8, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (frame.shape[1] - 60, 27),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)

        draw_session_summary(frame)

        if show_help:
            draw_help_overlay(frame)

        video_display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        panel = draw_confidence_panel(DISPLAY_HEIGHT, last_prediction_vector,
                                       last_display_emotion, smoothed_fps)
        display = np.hstack((video_display, panel))

        if recording and video_writer is not None:
            video_writer.write(display)

        cv2.imshow("Emotion Recognition - Mini-Xception", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            filename = os.path.join(OUTPUT_DIR, f"screenshot_{int(time.time())}.jpg")
            ok = cv2.imwrite(filename, display)
            print(f"Screenshot saved to {filename}" if ok else "Screenshot failed to save")

        elif key == ord("r"):
            if not recording:
                video_path = os.path.join(OUTPUT_DIR, f"recording_{int(time.time())}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_path, fourcc, 20.0,
                                                (display.shape[1], display.shape[0]))
                recording = True
                print(f"Recording started -> {video_path}")
            else:
                recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                print("Recording stopped.")

        elif key == ord("m"):
            mirror = not mirror

        elif key == ord("h"):
            show_help = not show_help

        elif key == ord("f"):
            show_fps = not show_fps

        elif key == ord("c"):
            conf_level_idx = (conf_level_idx + 1) % len(CONFIDENCE_LEVELS)
            print(f"Confidence threshold set to {CONFIDENCE_LEVELS[conf_level_idx]}%")

        elif key in (ord("q"), 27):  
            break

finally:
    if video_writer is not None:
        video_writer.release()
    camera.release()
    cv2.destroyAllWindows()

    if session_emotion_log:
        summary = Counter(session_emotion_log).most_common()
        print("\nSession summary:")
        for emo, cnt in summary:
            print(f"  {emo}: {cnt} frames")
