import streamlit as st
import cv2
from fer import FER
import numpy as np
from PIL import Image, ImageDraw, ImageFont
    
st.title("Emotion Recognition App with Emoticon Output")

# Inisialisasi detektor emosi
detector = FER()

# Function untuk menambahkan emoji ke frame
def add_emoji(frame, emotion):
    # Konversi frame CV2 ke PIL Image untuk menambahkan emoji
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Dictionary emoji untuk setiap emosi
    emoji_dict = {
        'happy': '😊',
        'sad': '😢',
        'angry': '😠',
        'surprise': '😲',
        'disgust': '🤢',
        'fear': '😨',
        'neutral': '😐'
    }
    
    # Tambahkan emoji yang sesuai
    emoji = emoji_dict.get(emotion, '🤔')
    position = (50, 50)
    draw.text(position, emoji, font=ImageFont.truetype("DejaVuSans.ttf", 40))
    
    # Konversi kembali ke format CV2
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def detect_emotion():
    # Buat placeholder untuk webcam feed
    frame_placeholder = st.empty()
    stop_button = st.button('Stop')
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Webcam tidak ditemukan!")
        return
        
    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal membaca frame dari webcam.")
            break
            
        # Deteksi emosi dari frame
        emotions = detector.detect_emotions(frame)
        
        if emotions:
            # Ambil emosi dengan confidence tertinggi
            emotion_dict = emotions[0]['emotions']
            emotion = max(emotion_dict, key=emotion_dict.get)
            confidence = emotion_dict[emotion]
            
            # Tambahkan emoji jika confidence melebihi threshold
            if confidence > 0.2:
                frame = add_emoji(frame, emotion)
                
            # Tambahkan teks confidence
            confidence_text = f"Emotion: {emotion} ({confidence:.2f})"
            cv2.putText(frame, confidence_text, (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Konversi frame untuk ditampilkan di Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")
        
    cap.release()

# Sidebar untuk kontrol
st.sidebar.title("Controls")
if st.sidebar.button('Start Emotion Detection'):
    detect_emotion()

# Tambahkan informasi cara penggunaan
st.markdown("""
### Cara Penggunaan:
1. Klik tombol 'Start Emotion Detection' di sidebar
2. Tunggu webcam menyala
3. Program akan mendeteksi emosi dan menampilkan emoji yang sesuai
4. Klik 'Stop' untuk menghentikan deteksi
""")