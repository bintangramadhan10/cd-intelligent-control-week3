import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# === Path model & dataset ===
MODEL_PATH = "scene_classification.h5"
TRAIN_DIR = r"G:\Semester 7\Kontrol Cerdas\Praktikum\week2\cd intelligent-control-week3\seg_train"

# Load model
model = load_model(MODEL_PATH)

# Ambil nama kelas dari folder dataset
class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
print("Class names:", class_names)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Night vision
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    night_vision = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Preprocessing
    img = cv2.resize(frame, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediksi semua kelas
    preds = model.predict(img)[0]

    # Normalisasi biar nggak semua 100%
    preds = preds / np.sum(preds)

    # Ambil kelas dengan probabilitas tertinggi
    idx = np.argmax(preds)
    label = class_names[idx]
    prob = preds[idx] * 100

    # Tulis hasil di layar
    text = f"{label} ({prob:.2f}%)"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Print semua kelas + prob di terminal
    print({cls: f"{p*100:.2f}%" for cls, p in zip(class_names, preds)})

    cv2.imshow("Prediction", frame)
    cv2.imshow("Night Vision", night_vision)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
