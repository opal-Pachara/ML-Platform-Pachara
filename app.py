import streamlit as st
from PIL import Image
from datetime import datetime
from streamlit_autorefresh import st_autorefresh  # สำหรับ real-time refresh
from show_introduction import show_introduction
from show_model_development import show_model_development
from show_ml import show_ml
from show_nn import show_nn

def main():
    # ตั้งค่า auto-refresh ทุก 1 วินาที
    st_autorefresh(interval=1000, key="visitor_refresh")

    # CSS สำหรับปรับแต่งธีมและฟอนต์
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Athiti:wght@400&display=swap');
        html, body, [class*="css"] {
            font-family: 'Athiti', sans-serif;
            font-size: 16px;
            font-weight: 400;
            line-height: 1.6;
            color: #ffffff;
            background-color: #121212;
        }
        .stSidebar, .stRadio, .stButton>button, .stMarkdown, 
        .stTextInput, .stNumberInput {
            font-family: 'Athiti', sans-serif !important;
            font-size: 16px !important;
            font-weight: 400 !important;
            line-height: 1.6 !important;
            color: #ffffff !important;
        }
        @media (prefers-color-scheme: dark) {
            .stApp {
                background: #121212;
                color: white;
            }
            h1, h2, h3, h4 {
                color: #B0B0B0;
            }
            .stButton>button {
                background-color: #333333;
                color: white;
                border-radius: 8px;
                padding: 10px;
                border: 1px solid #555555;
                transition: 0.3s;
            }
            .stButton>button:hover {
                background-color: #555555;
            }
            .stTextInput>div>div>input {
                background-color: #222222;
                color: white;
                border: 1px solid #444444;
            }
        }
        @media (prefers-color-scheme: light) {
            .stApp {
                background: white;
                color: black;
            }
            h1, h2, h3, h4 {
                color: #333333;
            }
            .stButton>button {
                background-color: #E0E0E0;
                color: black;
                border-radius: 8px;
                padding: 10px;
                border: 1px solid #BBBBBB;
                transition: 0.3s;
            }
            .stButton>button:hover {
                background-color: #CCCCCC;
            }
            .stTextInput>div>div>input {
                background-color: #F0F0F0;
                color: black;
                border: 1px solid #CCCCCC;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    # ใช้ session_state เพื่อนับจำนวนผู้เข้าชม
    if 'session_ids' not in st.session_state:
        st.session_state.session_ids = set()  # เก็บ session ID ที่ไม่ซ้ำใน set

    # สร้าง session ID ถ้ายังไม่มี
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(datetime.now().timestamp())
        st.session_state.session_ids.add(st.session_state.session_id)

    # นับจำนวนผู้เข้าชมจริงจาก session_ids
    real_count = len(st.session_state.session_ids)
    # เพิ่ม offset เพื่อให้เริ่มที่ 182 (181 + 1 = 182 สำหรับคนแรก)
    visitor_count = 181 + real_count

    # ส่วนที่เหลือของแอป
    st.logo("img/logo.png", size="large")
    st.sidebar.title("Intelligent System Project")
    st.sidebar.caption("Phatchara Worrawat 6404062610324")
    st.sidebar.title("Menu")
    page = st.sidebar.radio("", ["Introduction & Data Set", "Algorithm & Model Development", "Machine Learning Model", "Neural Network Model"])

    # แสดงจำนวนผู้เข้าชมใน sidebar ด้วย timestamp แบบ real-time
    st.sidebar.markdown(f"""
        <div style='text-align: center; padding: 15px; background-color: #1e1e1e; border-radius: 10px; margin-top: 20px;'>
            <h3 style='font-family: Athiti; color: #B0B0B0; margin: 0;'>👀 จำนวนคนเข้าชม</h3>
            <p style='font-family: Athiti; font-size: 28px; font-weight: bold; color: #00b4d8; margin: 5px 0;'>{visitor_count}</p>
            <p style='font-family: Athiti; font-size: 12px; color: #888888; margin: 0;'>อัปเดต: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """, unsafe_allow_html=True)

    if page == "Introduction & Data Set":
        show_introduction()
    elif page == "Algorithm & Model Development":
        show_model_development()
    elif page == "Machine Learning Model":
        show_ml()
    elif page == "Neural Network Model":
        show_nn()

if __name__ == "__main__":
    main()