import cv2
import numpy as np
import tensorflow as tf
import os

# Load model terbaik
MODEL_PATH = "best_scene_model.h5"   # pakai model terbaik
model = tf.keras.models.load_model(MODEL_PATH)

# Path dataset training untuk ambil nama kelas
TRAIN_DIR = r"G:\Semester 7\Kontrol Cerdas\Praktikum\week2\cd intelligent-control-week3\seg_train"
class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])

print("Class labels:", class_names)

# Open kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing ke ukuran sesuai training
    img = cv2.resize(frame, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediksi
    preds = model.predict(img, verbose=0)[0]
    class_idx = np.argmax(preds)
    label = class_names[class_idx]
    confidence = preds[class_idx] * 100  # ubah ke persen

    # Buat night vision
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    night_vision = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Tampilkan prediksi di frame asli
    cv2.putText(frame, f"{label}: {confidence:.2f}%", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Tampilkan semua kemungkinan kelas + akurasi
    y0, dy = 100, 30
    for i, cls in enumerate(class_names):
        text = f"{cls}: {preds[i]*100:.2f}%"
        cv2.putText(frame, text, (30, y0 + i*dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Tampilkan hasil
    cv2.imshow("Scene Classification", frame)
    cv2.imshow("Night Vision", night_vision)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
