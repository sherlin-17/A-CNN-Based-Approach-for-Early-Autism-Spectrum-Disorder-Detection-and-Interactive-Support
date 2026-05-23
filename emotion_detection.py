from keras.models import load_model
import cv2
import numpy as np
import time
from collections import deque, Counter
import pyttsx3

# =========================
# LOAD MODEL
# =========================
model = load_model("emotion_model.h5", compile=False)

model_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ASD-friendly mapping
emotion_map = {
    'Happy': ('Happy', '😊'),
    'Sad': ('Sad', '😢'),
    'Angry': ('Angry', '😠'),
    'Neutral': ('Neutral', '😐'),
    'Fear': ('Tension', '😟'),
    'Surprise': ('Confused', '😕'),
    'Disgust': ('Angry', '😠')
}

# =========================
# VOICE ENGINE (WINDOWS)
# =========================
engine = pyttsx3.init('sapi5')

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Female voice (Zira)

engine.setProperty('rate', 140)
engine.setProperty('volume', 0.9)

# English + Tamil messages
voice_messages = {
    'Happy': (
        "You look happy.",
        "நீ மகிழ்ச்சியாக இருக்கிறாய்."
    ),
    'Sad': (
        "You seem sad. It's okay.",
        "நீ சோகமாக இருக்கிறாய். பரவாயில்லை."
    ),
    'Angry': (
        "You look angry. Let's calm down.",
        "நீ கோபமாக இருக்கிறாய். அமைதியாகலாம்."
    ),
    'Neutral': (
        "You look calm.",
        "நீ அமைதியாக இருக்கிறாய்."
    ),
    'Tension': (
        "You seem tense. Take a deep breath.",
        "நீ பதற்றமாக இருக்கிறாய். மெதுவாக மூச்சு விடு."
    ),
    'Confused': (
        "You look confused. Take your time.",
        "நீ குழப்பமாக இருக்கிறாய். மெதுவாக யோசி."
    )
}

last_spoken_emotion = None

# =========================
# STABILITY + COUNTERS
# =========================
emotion_history = deque(maxlen=7)

emotion_count = {
    'Happy': 0,
    'Sad': 0,
    'Angry': 0,
    'Neutral': 0,
    'Tension': 0,
    'Confused': 0
}

# =========================
# FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

last_detection_time = 0
detection_interval = 10  # seconds
current_emotion_text = "Detecting..."

print("\nASD Emotion Detection Started")
print("Emotion updates every 10 seconds\n")

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    current_time = time.time()

    # Predict only every 10 seconds
    if current_time - last_detection_time >= detection_interval and len(faces) > 0:
        x, y, w, h = faces[0]

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        preds = model.predict(face, verbose=0)[0]
        neutral_index = model_labels.index("Neutral")

        if preds[neutral_index] > 0.60:
            label = "Neutral"
        else:
            label = model_labels[np.argsort(preds)[-2]]

        asd_emotion, emoji = emotion_map[label]

        emotion_history.append(asd_emotion)
        stable_emotion = Counter(emotion_history).most_common(1)[0][0]

        emotion_count[stable_emotion] += 1
        current_emotion_text = f"{stable_emotion} {emoji}"

        print(f"[{time.strftime('%H:%M:%S')}] Emotion Detected → {current_emotion_text}")

        # 🔊 Speak only if emotion changes
        if stable_emotion != last_spoken_emotion:
            english_msg, tamil_msg = voice_messages[stable_emotion]

            engine.say(english_msg)
            engine.runAndWait()

            time.sleep(0.5)

            engine.say(tamil_msg)
            engine.runAndWait()

            last_spoken_emotion = stable_emotion

        last_detection_time = current_time

    # Draw face + emotion
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            current_emotion_text,
            (x, max(y-10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("ASD Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()

print("\nEmotion Trend Summary (ASD)\n")
for emotion, count in emotion_count.items():
    bar = "█" * count
    print(f"{emotion:<10} | {bar} ({count})")

print("\nSession Ended")
