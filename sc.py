import streamlit as st
import cv2
from fer import FER
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

st.title("Emotion Recognition App with Emoticon Output")

# Inisialisasi detektor emosi
detector = FER()

# Konfigurasi WebRTC
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Class untuk memproses video
class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = FER()
    
    def add_emoji(self, frame, emotion):
        try:
            # Konversi frame CV2 ke PIL Image
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
            
            try:
                font = ImageFont.load_default()
                draw.text(position, emoji, font=font)
            except Exception as e:
                print(f"Font error: {str(e)}")
                
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error in add_emoji: {str(e)}")
            return frame

    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # Deteksi emosi
            emotions = self.detector.detect_emotions(img)
            
            if emotions:
                # Ambil emosi dengan confidence tertinggi
                emotion_dict = emotions[0]['emotions']
                emotion = max(emotion_dict, key=emotion_dict.get)
                confidence = emotion_dict[emotion]
                
                # Tambahkan emoji jika confidence melebihi threshold
                if confidence > 0.2:
                    img = self.add_emoji(img, emotion)
                    
                # Tambahkan teks confidence
                confidence_text = f"Emotion: {emotion} ({confidence:.2f})"
                cv2.putText(img, confidence_text, (10, img.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            print(f"Error in transform: {str(e)}")
            return frame

# Sidebar untuk kontrol
st.sidebar.title("Controls")

# Tampilkan webcam feed dengan emotion detection
webrtc_ctx = webrtc_streamer(
    key="emotion-detection",
    mode=webrtc_streamer.transform_frames,
    rtc_configuration=rtc_configuration,
    video_transformer_factory=EmotionTransformer,
    async_transform=True
)

# Tambahkan informasi cara penggunaan
st.markdown("""
### Cara Penggunaan:
1. Izinkan browser mengakses webcam Anda ketika diminta
2. Klik tombol 'START' untuk memulai deteksi
3. Program akan mendeteksi emosi dan menampilkan emoji yang sesuai secara real-time
4. Klik 'STOP' untuk menghentikan deteksi
""")

# Tambahkan informasi tambahan
st.sidebar.markdown("""
### Tips:
- Pastikan pencahayaan cukup
- Posisikan wajah dengan jelas
- Jaga jarak yang sesuai dari kamera
""")
