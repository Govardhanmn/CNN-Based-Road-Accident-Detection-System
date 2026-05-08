import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import base64, pathlib, time, datetime

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisionAI · Intelligent Surveillance",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Resources ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return tf.keras.models.load_model('best_cnn_model.keras')
    except:
        return None

model = load_model()

# ── Background image ──────────────────────────────────────────────────────────
@st.cache_data
def get_bg_b64():
    _path = pathlib.Path("bg_cnn.png")
    if _path.exists():
        return base64.b64encode(_path.read_bytes()).decode()
    return ""

_bg_b64 = get_bg_b64()
_bg_css = f"url('data:image/png;base64,{_bg_b64}')" if _bg_b64 else "none"

# ── Premium Styles ────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, .stApp {{
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
    background-color: #060b16;
}}

/* ── Full-page background ── */
.stApp {{
    background-image: {_bg_css};
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
.stApp::before {{
    content: '';
    position: fixed;
    inset: 0;
    background: rgba(6, 11, 22, 0.88);
    z-index: 0;
}}

#MainMenu, footer, header {{ visibility: hidden; }}

.block-container {{
    position: relative;
    z-index: 1;
    padding: 1rem 2rem 2rem 2rem !important;
    max-width: 100% !important;
}}

/* ─── SIDEBAR ────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: #080e1a !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
    width: 320px !important;
}}
section[data-testid="stSidebar"] .stMarkdown {{ margin-bottom: -1rem; }}

/* ─── TOP BAR ────────────────────────────────────────────── */
.top-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding: 0.5rem 0;
}}
.logo-area {{ display: flex; align-items: center; gap: 0.8rem; }}
.logo-icon {{ font-size: 2rem; color: #00d2ff; }}
.logo-text {{ line-height: 1.1; }}
.logo-main {{ font-size: 1.2rem; font-weight: 800; letter-spacing: 1px; }}
.logo-sub {{ font-size: 0.65rem; color: #64748b; letter-spacing: 1px; }}

.alert-banner {{
    flex-grow: 1;
    margin: 0 2rem;
    background: rgba(220, 38, 38, 0.1);
    border: 1px solid rgba(220, 38, 38, 0.3);
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: #fca5a5;
    animation: pulse-red 2s infinite;
}}
@keyframes pulse-red {{
    0% {{ box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.2); }}
    70% {{ box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }}
}}

.time-area {{ text-align: right; line-height: 1.2; }}
.time-clock {{ font-family: 'JetBrains Mono'; font-size: 1.1rem; font-weight: 600; }}
.time-date {{ font-size: 0.75rem; color: #64748b; }}

/* ─── GLASS CARDS ──────────────────────────────────────────── */
.glass-card {{
    background: rgba(13, 20, 36, 0.6);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}}
.card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.7rem;
    font-weight: 700;
    color: #00d2ff;
    letter-spacing: 1px;
    margin-bottom: 1rem;
    text-transform: uppercase;
}}

/* ─── SIDEBAR GROUPS ─────────────────────────────────────── */
.sb-group {{ margin-bottom: 2rem; }}
.sb-label {{ font-size: 0.65rem; font-weight: 700; color: #475569; letter-spacing: 1px; margin-bottom: 0.8rem; text-transform: uppercase; }}
.status-pill {{
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 6px;
    padding: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}}
.status-dot {{ width: 8px; height: 8px; background: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981; }}
.cam-item {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0.8rem;
    border-radius: 6px;
    margin-bottom: 0.3rem;
    background: rgba(255,255,255,0.02);
    font-size: 0.75rem;
    cursor: pointer;
}}
.cam-item.active {{ background: rgba(220, 38, 38, 0.1); border: 1px solid rgba(220, 38, 38, 0.2); }}
.dot-green {{ width: 6px; height: 6px; background: #10b981; border-radius: 50%; }}
.dot-red {{ width: 6px; height: 6px; background: #ef4444; border-radius: 50%; box-shadow: 0 0 6px #ef4444; }}

/* ─── DETECTION SUMMARY ───────────────────────────────────── */
.summary-alert {{
    background: rgba(220, 38, 38, 0.05);
    border: 1px solid rgba(220, 38, 38, 0.2);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1.2rem;
}}
.sum-title {{ color: #ef4444; font-weight: 800; font-size: 1.2rem; margin-bottom: 1rem; }}
.gauge-container {{
    position: relative;
    width: 80px;
    height: 80px;
    margin: 0 auto 1rem;
}}
.circular-gauge {{
    width: 100%; height: 100%;
    border-radius: 50%;
    background: conic-gradient(#ef4444 92%, rgba(255,255,255,0.05) 0);
    display: flex; align-items: center; justify-content: center;
}}
.circular-gauge::after {{
    content: '92%';
    width: 80%; height: 80%;
    background: #0d1424;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; font-weight: 800;
}}

.detail-row {{ display: flex; justify-content: space-between; font-size: 0.75rem; padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.03); }}
.detail-label {{ color: #64748b; }}
.detail-val {{ font-weight: 600; }}
.val-red {{ color: #ef4444; }}

/* ─── EVENT TIMELINE ──────────────────────────────────────── */
.timeline-item {{
    display: flex;
    gap: 1rem;
    padding: 0.8rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    font-size: 0.75rem;
}}
.tl-time {{ color: #475569; font-family: 'JetBrains Mono'; min-width: 60px; }}
.tl-icon {{ width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; }}
.tl-content {{ flex-grow: 1; }}
.tl-title {{ font-weight: 600; margin-bottom: 0.1rem; }}
.tl-desc {{ color: #64748b; font-size: 0.65rem; }}

/* ─── BUTTONS ────────────────────────────────────────────── */
.stButton > button {{
    width: 100% !important;
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    font-size: 0.8rem !important;
    height: 40px !important;
    border-radius: 8px !important;
}}
.btn-red > div > button {{
    background: linear-gradient(90deg, #991b1b, #ef4444) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
now = datetime.datetime.now()
st.markdown(f"""
<div class="top-bar">
    <div class="logo-area">
        <div class="logo-icon">👁️</div>
        <div class="logo-text">
            <div class="logo-main">VISION AI</div>
            <div class="logo-sub">INTELLIGENT SURVEILLANCE</div>
        </div>
    </div>
    <div class="alert-banner">
        <div style="display:flex; align-items:center; gap:1rem;">
            <span style="font-size:1.2rem">⚠️</span>
            <span style="font-weight:800; letter-spacing:1px">ACCIDENT DETECTED</span>
            <span style="opacity:0.6; font-size:0.8rem">Possible collision detected</span>
        </div>
        <div style="font-size:0.7rem; font-weight:700; border:1px solid rgba(252,165,165,0.4); padding:0.3rem 0.8rem; border-radius:4px; cursor:pointer">VIEW ALERT ></div>
    </div>
    <div class="time-area">
        <div class="time-clock">{now.strftime("%H:%M:%S")}</div>
        <div class="time-date">{now.strftime("%b %d, %Y")} ☀️</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-group"><div class="sb-label">System</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="status-pill">
        <div class="status-dot"></div>
        <div>
            <div style="font-weight:800; font-size:0.85rem; color:#10b981">ONLINE</div>
            <div style="font-size:0.65rem; color:#64748b">All systems operational</div>
        </div>
        <div style="margin-left:auto; color:#10b981; font-size:1.2rem">🛡️</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-group"><div class="sb-label">Settings</div>', unsafe_allow_html=True)
    st.slider("Detection Sensitivity", 0, 100, 75)
    st.toggle("Night Mode", value=True)
    st.toggle("Auto Alert", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Main Layout ───────────────────────────────────────────────────────────────
main_col, side_col = st.columns([1.6, 0.9], gap="medium")

with main_col:
    # ── Live Feed ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header"><span>LIVE FEED</span><span style="color:#ef4444">● LIVE</span></div>', unsafe_allow_html=True)
    
    # Placeholder for Video/Image
    up_file = st.file_uploader("Upload CCTV snapshot to analyze...", type=["jpg","png","jpeg"], label_visibility="collapsed")
    if up_file:
        img = Image.open(up_file)
        st.image(img, use_container_width=True)
    else:
        st.markdown('<div style="width:100%; aspect-ratio:16/9; background:rgba(0,0,0,0.3); border-radius:8px; display:flex; align-items:center; justify-content:center; color:#475569">NO ACTIVE FEED - UPLOAD IMAGE BELOW</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:1rem; font-size:0.8rem; color:#64748b">
        <div style="font-family:'JetBrains Mono'">2025-05-24  14:32:10</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with side_col:
    # ── Detection Summary ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">DETECTION SUMMARY</div>', unsafe_allow_html=True)
    
    if up_file:
        with st.spinner("AI analyzing..."):
            time.sleep(1)
            # Prediction
            img_p = img.resize((128, 128))
            arr = img_to_array(img_p)
            arr = np.expand_dims(arr, axis=0) / 255.0
            prob = model.predict(arr, verbose=0)[0][0]
            accident = prob < 0.5
            conf = (1-prob)*100 if accident else prob*100
            
            if accident:
                st.markdown(f"""
                <div class="summary-alert">
                    <div style="font-size:3rem; margin-bottom:0.5rem">⚠️</div>
                    <div class="sum-title">ACCIDENT DETECTED</div>
                    <div style="display:flex; align-items:center; justify-content:center; gap:2rem; margin-top:1.5rem">
                        <div style="text-align:left">
                            <div style="font-size:0.65rem; color:#64748b">Confidence Score</div>
                            <div style="font-size:1.8rem; font-weight:800">{conf:.0f}%</div>
                        </div>
                        <div class="gauge-container"><div class="circular-gauge" style="background:conic-gradient(#ef4444 {conf}%, rgba(255,255,255,0.05) 0)"></div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="summary-alert" style="background:rgba(16,185,129,0.05); border-color:rgba(16,185,129,0.2)">
                    <div style="font-size:3rem; margin-bottom:0.5rem">✅</div>
                    <div class="sum-title" style="color:#10b981">ROAD STATUS: SAFE</div>
                    <div style="display:flex; align-items:center; justify-content:center; gap:2rem; margin-top:1.5rem">
                        <div style="text-align:left">
                            <div style="font-size:0.65rem; color:#64748b">Confidence Score</div>
                            <div style="font-size:1.8rem; font-weight:800; color:#10b981">{conf:.0f}%</div>
                        </div>
                        <div class="gauge-container"><div class="circular-gauge" style="background:conic-gradient(#10b981 {conf}%, rgba(255,255,255,0.05) 0)"></div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="summary-alert">
            <div style="font-size:3rem; margin-bottom:0.5rem">⚠️</div>
            <div class="sum-title">ACCIDENT DETECTED</div>
            <div style="display:flex; align-items:center; justify-content:center; gap:2rem; margin-top:1.5rem">
                <div style="text-align:left">
                    <div style="font-size:0.65rem; color:#64748b">Confidence Score</div>
                    <div style="font-size:1.8rem; font-weight:800">92%</div>
                </div>
                <div class="gauge-container"><div class="circular-gauge"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    rows = [
        ("Accident Type", "Collision", "red"),
        ("Severity Level", "High", "red"),
        ("Location", "Road - East", ""),
        ("Time", "14:32:10", "")
    ]
    for label, val, cls in rows:
        val_cls = "val-red" if cls == "red" else ""
        st.markdown(f'<div class="detail-row"><span class="detail-label">{label}</span><span class="detail-val {val_cls}">{val}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Suggested Action ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">SUGGESTED ACTION</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="timeline-item" style="border:none">
        <div style="display:flex; flex-direction:column; gap:0.8rem; width:100%">
            <div style="display:flex; align-items:center; gap:0.8rem">✅ <span style="font-size:0.75rem">Notify Control Room</span></div>
            <div style="display:flex; align-items:center; gap:0.8rem">✅ <span style="font-size:0.75rem">Send Emergency Alert</span></div>
            <div style="display:flex; align-items:center; gap:0.8rem">✅ <span style="font-size:0.75rem">Record & Save Clip</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="btn-red">', unsafe_allow_html=True)
    st.button("🔔 SEND ALERT NOW")
    st.markdown('</div></div>', unsafe_allow_html=True)
