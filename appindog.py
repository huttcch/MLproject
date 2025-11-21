import streamlit as st
import time
import numpy as np
from PIL import Image
import tensorflow as tf

# ==========================================
# 1. SETUP & STYLING
# ==========================================
st.set_page_config(
    page_title="DogDetect AI - Real vs AI",
    page_icon="🐕",
    layout="centered"
)

# CSS สำหรับ Dashboard UI และ Loading
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif;
    }

    .main-header {
        text-align: center;
        margin-bottom: 30px;
    }
    .main-header h1 {
        color: #2c3e50;
        font-weight: 700;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #7f8c8d;
        font-size: 1.1rem;
    }

    /* Dashboard Card */
    .result-card {
        background-color: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid #f0f2f5;
        margin-bottom: 20px;
    }

    .score-big {
        font-size: 4rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #3498db, #8e44ad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .label-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 10px;
    }

    .badge-ai { background-color: #ffebee; color: #c62828; }
    .badge-real { background-color: #e8f5e9; color: #2e7d32; }

    /* Cookie Banner */
    .cookie-box {
        background-color: #34495e;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Upload Area */
    .stFileUploader {
        border: 2px dashed #bdc3c7;
        border-radius: 15px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Dictionary ภาษา
translations = {
    "th": {
        "title": "🐕 DogDetect AI",
        "subtitle": "ตรวจจับภาพน้องหมาว่าเป็น 'ภาพ AI' หรือไม่",
        "upload_label": "อัปโหลดภาพน้องหมา (Drag & Drop)",
        "analyzing": "กำลังประมวลผล...",
        "result_title": "ผลการวิเคราะห์ AI",
        "ai_prob": "ความเป็น AI",
        "type": "ประเภท",
        "type_ai": "🤖 ภาพจาก AI (Generated)",
        "type_real": "📸 ภาพถ่ายจริง (Real Photo)",
        "share": "แชร์ผลลัพธ์",
        "cookie_text": "🍪 เว็บไซต์นี้ใช้คุกกี้เพื่อพัฒนาโมเดล AI หากยอมรับ ภาพของคุณจะถูกนำไปเรียนรู้",
        "accept": "ยอมรับ",
        "decline": "ไม่ยอมรับ",
        "sensitive_title": "⚠️ แจ้งเตือนภาพละเอียดอ่อน",
        "sensitive_msg": "ระบบตรวจพบว่าภาพนี้อาจมีข้อมูลส่วนบุคคล (เช่น หน้าคน, บัตรประชาชน) คุณยืนยันที่จะประมวลผลหรือไม่?",
        "btn_continue": "ยืนยัน / ทำต่อ",
        "btn_cancel": "ยกเลิก",
        "error_model": "❌ ไม่พบไฟล์ 'dog_model_binary.keras' กรุณารันไฟล์ train_model.py ก่อน"
    },
    "en": {
        "title": "🐕 DogDetect AI",
        "subtitle": "Detect if a dog image is 'AI Generated' or Real",
        "upload_label": "Upload Dog Image (Drag & Drop)",
        "analyzing": "Processing...",
        "result_title": "AI Analysis Result",
        "ai_prob": "AI Probability",
        "type": "Type",
        "type_ai": "🤖 AI Generated",
        "type_real": "📸 Real Photo",
        "share": "Share Result",
        "cookie_text": "🍪 We use cookies to improve our AI model.",
        "accept": "Accept",
        "decline": "Decline",
        "sensitive_title": "⚠️ Sensitive Content Warning",
        "sensitive_msg": "This image may contain personal data (faces, IDs). Do you want to proceed?",
        "btn_continue": "Confirm / Proceed",
        "btn_cancel": "Cancel",
        "error_model": "❌ Model file 'dog_model_binary.keras' not found. Please run train_model.py first."
    }
}


# ==========================================
# 2. LOGIC & FUNCTIONS
# ==========================================
@st.cache_resource
def load_ai_model():
    try:
        model = tf.keras.models.load_model('dog_model_binary.keras')
        return model
    except:
        return None


def predict_image(model, image):
    # Preprocess (ต้องเหมือนตอนเทรนเป๊ะๆ)
    img = image.resize((224, 224))
    img_array = np.array(img)

    # Handle RGBA
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32)  # 0-255 float

    # Predict
    prediction = model.predict(img_array)
    score = prediction[0][0]  # ค่า 0-1 (0=AI, 1=Real ตามลำดับโฟลเดอร์ ai, real)

    # Interpret Result
    # ถ้า score ต่ำกว่า 0.5 แปลว่า AI (เพราะ ai มาก่อน real ตามตัวอักษร)
    if score < 0.5:
        is_ai = True
        ai_percent = (1 - score) * 100
    else:
        is_ai = False
        ai_percent = (1 - score) * 100  # AI probability ของภาพ Real ก็คือส่วนกลับ

    return is_ai, ai_percent


# ฟังก์ชันจำลองการตรวจจับ Sensitive Data (Mockup)
def check_sensitive_content(image):
    # ในการใช้งานจริงอาจใช้ Face Detection หรือ OCR
    # ตรงนี้เราจำลองด้วยการสุ่ม หรือเช็คขนาดไฟล์เพื่อสาธิต UI Popup
    import random
    return random.random() > 0.7  # สุ่มเจอ 30% เพื่อเทสระบบ


# ==========================================
# 3. MAIN APPLICATION FLOW
# ==========================================

# Session State Management
if 'lang' not in st.session_state: st.session_state.lang = 'th'
if 'cookie_consent' not in st.session_state: st.session_state.cookie_consent = None  # None, True, False
if 'sensitive_confirmed' not in st.session_state: st.session_state.sensitive_confirmed = False

t = translations[st.session_state.lang]
model = load_ai_model()

# --- Sidebar: Language ---
with st.sidebar:
    st.header("Settings ⚙️")
    lang_choice = st.radio("Language / ภาษา", ["ภาษาไทย", "English"])
    if lang_choice == "English":
        st.session_state.lang = 'en'
    else:
        st.session_state.lang = 'th'
    st.rerun if lang_choice != ("ภาษาไทย" if st.session_state.lang == 'th' else "English") else None

# --- 1. Cookie Banner ---
if st.session_state.cookie_consent is None:
    with st.container():
        st.markdown(f"""
        <div class="cookie-box">
            <div>{t['cookie_text']}</div>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([6, 1, 1])
        if col2.button(t['accept']):
            st.session_state.cookie_consent = True
            st.rerun()
        if col3.button(t['decline']):
            st.session_state.cookie_consent = False
            st.rerun()

# --- 2. Header ---
st.markdown(f"""
<div class="main-header">
    <h1>{t['title']}</h1>
    <p>{t['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# --- 3. Model Check ---
if model is None:
    st.error(t['error_model'])
else:
    # --- 4. Upload ---
    uploaded_file = st.file_uploader(t['upload_label'], type=['jpg', 'png', 'webp', 'heic', 'jpeg'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Preview", use_container_width=True)

        # --- 5. Sensitive Content Check Logic ---
        # ตรวจสอบครั้งแรกเมื่ออัปโหลด
        if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            st.session_state.is_sensitive = check_sensitive_content(image)
            st.session_state.last_uploaded = uploaded_file.name
            st.session_state.sensitive_confirmed = False  # Reset confirmation

        # ถ้าเป็นภาพ Sensitive และยังไม่ยืนยัน
        if st.session_state.is_sensitive and not st.session_state.sensitive_confirmed:
            with st.container():
                st.warning(f"**{t['sensitive_title']}**")
                st.write(t['sensitive_msg'])
                c1, c2 = st.columns(2)
                if c1.button(t['btn_continue'], type="primary"):
                    st.session_state.sensitive_confirmed = True
                    st.rerun()
                if c2.button(t['btn_cancel']):
                    st.session_state.last_uploaded = None  # Reset
                    st.rerun()
            st.stop()  # หยุดการทำงานส่วนล่างจนกว่าจะยืนยัน

        # --- 6. Analyze Button ---
        if st.button("🚀 " + t['analyzing'].replace("...", ""), type="primary", use_container_width=True):

            # Loading Animation
            progress_text = t['analyzing']
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.01)  # Simulated delay (< 10s requirement)
                my_bar.progress(percent_complete + 1, text=progress_text)

            # Prediction
            is_ai, ai_percent = predict_image(model, image)
            my_bar.empty()

            # --- 7. Result Dashboard ---
            st.markdown("---")
            st.markdown(f"<h3 style='text-align: center;'>{t['result_title']}</h3>", unsafe_allow_html=True)

            # Determine Styles
            if is_ai:
                badge_class = "badge-ai"
                badge_text = t['type_ai']
                score_color = "#c62828"
            else:
                badge_class = "badge-real"
                badge_text = t['type_real']
                score_color = "#2e7d32"  # Green for Real

            # Layout
            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"""
                <div class="result-card">
                    <div style="color: #7f8c8d; font-weight:600;">{t['type']}</div>
                    <div class="label-badge {badge_class}">{badge_text}</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                # Requirement: Show % AI
                # If Real, we still show AI % (which is low), or we show Real %?
                # Req says: "ภาพนี้เป็น AI 72%"
                # So we always show AI Probability

                display_percent = ai_percent

                st.markdown(f"""
                <div class="result-card">
                    <div style="color: #7f8c8d; font-weight:600;">{t['ai_prob']}</div>
                    <div class="score-big" style="background: -webkit-linear-gradient(45deg, #2c3e50, {score_color}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        {display_percent:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # --- 8. Share Buttons ---
            st.markdown(f"<center style='color:#aaa; margin-top:20px;'>{t['share']}</center>", unsafe_allow_html=True)
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.button("🔗 Copy Link", use_container_width=True)
            col_s2.button("📘 Facebook", use_container_width=True)
            col_s3.button("❌ X (Twitter)", use_container_width=True)