import cv2
from fer import FER

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
detector = FER()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi emosi dari wajah di frame
    emotions = detector.detect_emotions(frame)

    if emotions:
        emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
        # Menampilkan emotikon sesuai emosi
        if emotion == 'happy':
            cv2.putText(frame, '😊', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        elif emotion == 'sad':
            cv2.putText(frame, '😢', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        # Tambahkan emotikon lainnya sesuai kebutuhan

    # Tampilkan hasil deteksi
    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
