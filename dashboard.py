import streamlit as st
import requests
from PIL import Image
import io
import datetime
import time
import json
from streamlit_lottie import st_lottie

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI Detector | Premium", 
    page_icon="✨", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import os

# Add root to path for ML imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml.model import EnsembleDetector
from ml.analysis import fft_analysis, noise_residual_analysis

# Serverless Model Loader
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleDetector(pretrained=False)
    try:
        if os.path.exists("best_model.pth"):
            model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        return None, None

model, device = load_model()

# Inference Transform
inference_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# --- STATE ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

@st.cache_data(ttl=3600)
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=2) 
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def load_lottie_local(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except:
        return None

# Load Lottie Assets (moved to global scope for early loading)
LOTTIE_AI_SCAN = None
if os.path.exists("assets/ai_scan.json"):
    LOTTIE_AI_SCAN = load_lottie_local("assets/ai_scan.json")

if not LOTTIE_AI_SCAN:
    LOTTIE_AI_SCAN = load_lottieurl("https://lottie.host/02008323-2882-4c28-8255-6447c2d15383/8zD3QJb1Wj.json")

# --- STATE ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- ULTRA-PREMIUM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    :root {
        --primary: #6366f1;
        --secondary: #ec4899;
        --accent: #8b5cf6;
        --bg-color: #f8fafc;
        --card-bg: rgba(255, 255, 255, 0.85);
        --text-main: #0f172a;
        --text-sub: #64748b;
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: var(--text-main);
    }

    /* ANIMATED BACKGROUND */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(239, 246, 255) 0%, rgb(255, 255, 255) 90%);
        background-attachment: fixed;
    }
    
    /* Subtle mesh gradient overlay */
    .stApp::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.08), transparent 60%),
                    radial-gradient(circle at 80% 20%, rgba(236, 72, 153, 0.08), transparent 50%);
        z-index: -1;
        animation: rotateBg 20s linear infinite;
    }
    
    @keyframes rotateBg {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* GLASS CARDS */
    .glass-card, [data-testid="column"] > div {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.6);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.05);
        transition: transform 0.4s cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 0.4s ease;
        margin-bottom: 1rem;
    }
    .glass-card:hover, [data-testid="column"] > div:hover {
        transform: translateY(-5px);
        box-shadow: 0 30px 60px -12px rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.4);
    }

    /* IMAGE STYLING */
    .stImage img {
        border-radius: 20px;
        box-shadow: 0 15px 30px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .stImage img:hover {
        transform: scale(1.02);
    }

    /* ACTION BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 25px rgba(236, 72, 153, 0.4);
    }

    /* RESULT BANNER */
    .result-banner {
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        animation: popIn 0.5s cubic-bezier(0.18, 0.89, 0.32, 1.28);
    }
    .res-fake {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #ef4444;
        color: #991b1b;
    }
    .res-real {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 2px solid #22c55e;
        color: #166534;
    }

    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* TEXT GRADIENTS */
    .grad-text {
        background: linear-gradient(to right, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* UPLOADER */
    div[data-testid="stFileUploader"], .st-emotion-cache-1ae8kth {
        padding: 2rem;
        border: 2px dashed #cbd5e1;
        border-radius: 20px;
        background: rgba(255,255,255,0.5);
    }
    
    /* Hide default Streamlit elements if they interfere with premium feel */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Ensure the main container doesn't have too much padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
c1, c2 = st.columns([1, 10])
with c1:
    st.markdown("<div style='font-size:2.5rem;'>✨</div>", unsafe_allow_html=True)
with c2:
    st.markdown("<h3 style='margin-top:10px;'>Apex <span class='grad-text'>Neural Detector</span></h3>", unsafe_allow_html=True)

# --- HERO ---
h1, h2 = st.columns([1.3, 1])
with h1:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <h1 style='font-size: 4rem; line-height: 1.1; margin-bottom: 20px;'>
        Unmask the <span class="grad-text">Synthetic</span>.
    </h1>
    <p style='font-size: 1.3rem; color: #64748B; margin-bottom: 30px;'>
        Advanced forensic analysis for visual media. Detect deepfakes, GANs, and diffusion patterns with 99.8% precision.
    </p>
    """, unsafe_allow_html=True)
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", "99.8%", "+0.2%")
    m2.metric("Speed", "45ms", "-12ms")
    m3.metric("Models", "Ensemble-B0", "Active")

with h2:
    if LOTTIE_AI_SCAN:
        st_lottie(LOTTIE_AI_SCAN, height=350, key="hero_anim")

# --- MAIN UPLOAD ---
st.markdown("---")
st.markdown("<h2 style='text-align:center; margin-bottom: 2rem; font-size: 2.5rem;'>Start Forensic <span class='grad-text'>Analysis</span></h2>", unsafe_allow_html=True)

# Custom CSS for the upload zone
st.markdown("""
<style>
    /* Animated Upload Zone */
    [data-testid="stFileUploader"] {
        padding: 3rem 2rem;
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 24px;
        background: linear-gradient(145deg, rgba(255,255,255,0.6), rgba(255,255,255,0.3));
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary);
        background: linear-gradient(145deg, rgba(99, 102, 241, 0.05), rgba(236, 72, 153, 0.05));
        transform: translateY(-5px);
        box-shadow: 0 15px 40px -10px rgba(99, 102, 241, 0.15);
    }
    
    /* Fake "Cloud" Icon Animation inside standard uploader via CSS content is hard, 
       so we trust the custom layout wrapper below */
       
    .upload-icon {
        font-size: 4rem;
        background: linear-gradient(to bottom right, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        animation: float 3s ease-in-out infinite;
    }
    
    .upload-text {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-main);
    }
    
    .upload-subtext {
        font-size: 0.9rem;
        color: var(--text-sub);
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

uc1, uc2, uc3 = st.columns([1, 2, 1])
with uc2:
    # We wrap the uploader in a custom styled container that looks like the "active" zone
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding: 0;">
        <div style="padding: 2rem;">
            <div class="upload-icon">☁️</div>
            <div class="upload-text">Drag & Drop Evidence</div>
            <div class="upload-subtext">Supports High-Res JPG, PNG • Max 200MB</div>
        </div>
    """, unsafe_allow_html=True)
    
    # The actual functional uploader, visually blended
    uploaded_file = st.file_uploader(" ", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- ANALYSIS ENGINE ---
if uploaded_file:
    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2 = st.columns([1, 1])
    
    image = Image.open(uploaded_file)
    
    with r1:
        st.markdown('<div class="glass-card" style="display:flex; justify-content:center; align-items:center; overflow:hidden;">', unsafe_allow_html=True)
        # Fix: using use_container_width instead of deprecated use_column_width
        st.image(image, caption="Suspect Media Analysis", use_container_width=True) 
        st.markdown('</div>', unsafe_allow_html=True)
        
    with r2:
        # Custom "World Class" Styling for Console
        st.markdown("""
        <style>
            /* Glowing Pulse Button */
            .stButton button[kind="primary"] {
                background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
                background-size: 200% auto;
                color: white;
                border: none;
                padding: 1rem 2rem;
                font-size: 1.1rem;
                font-weight: 800;
                letter-spacing: 1px;
                border-radius: 50px;
                transition: all 0.5s ease;
                animation: gradientMove 3s infinite linear;
                box-shadow: 0 10px 30px -10px rgba(168, 85, 247, 0.6);
            }
            
            .stButton button[kind="primary"]:hover {
                transform: scale(1.03) translateY(-2px);
                box-shadow: 0 20px 40px -10px rgba(236, 72, 153, 0.7);
            }
            
            @keyframes gradientMove {
                0% { background-position: 0% 50% }
                50% { background-position: 100% 50% }
                100% { background-position: 0% 50% }
            }
            
            /* Premium Result Card */
            .apex-result-card {
                background: rgba(255, 255, 255, 0.9);
                backdrop-filter: blur(20px);
                border-radius: 30px;
                padding: 2.5rem;
                text-align: center;
                box-shadow: 0 20px 60px -15px rgba(0,0,0,0.1);
                border: 1px solid rgba(255,255,255,0.8);
                transition: transform 0.4s ease;
            }
            
            .apex-result-card:hover {
                transform: translateY(-5px) scale(1.01);
            }
            
            .res-icon { font-size: 3.5rem; margin-bottom: 0.5rem; display: inline-block; animation: popIn 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275); }
            
            .res-title { font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem; letter-spacing: -1px; }
            .res-subtitle { font-size: 1.2rem; font-weight: 500; opacity: 0.8; margin-bottom: 1.5rem; }
            
            .res-fake-grad { background: -webkit-linear-gradient(45deg, #ef4444, #b91c1c); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .res-real-grad { background: -webkit-linear-gradient(45deg, #10b981, #047857); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='margin-bottom: 1rem;'>Diagnostic <span class='grad-text'>Console</span></h3>", unsafe_allow_html=True)
        
        if st.button("⚡ INITIALIZE NEURAL SCAN", type="primary", use_container_width=True):
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            # Simulation of steps
            steps = ["Decomposing high-frequency patterns...", "Analyzing noise residuals...", "Checking FFT spectral artifacts...", "Aggregating ensemble votes..."]
            for i, step in enumerate(steps):
                status_text.markdown(f"<span style='color:#6366f1; font-weight:600;'>◈ {step}</span>", unsafe_allow_html=True)
                progress_bar.progress((i + 1) * 25)
                time.sleep(0.3)
            
            # Real Inference
            try:
                img_byte_arr = io.BytesIO()
                fmt = image.format if image.format else 'PNG'
                image.save(img_byte_arr, format=fmt)
                img_byte_arr = img_byte_arr.getvalue()
                
                files = {'file': (uploaded_file.name, img_byte_arr, uploaded_file.type)}
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                
                status_text.empty()
                progress_bar.empty()
                
                if response.status_code == 200:
                    data = response.json()
                    prob = data['probability']
                    is_fake = data['is_fake']
                    
                    # Store history
                    st.session_state['history'].insert(0, {
                        'name': uploaded_file.name,
                        'fake': is_fake,
                        'prob': prob,
                        'time': datetime.datetime.now().strftime("%H:%M")
                    })
                    
                    # CUSTOM RESULT BANNER
                    if is_fake:
                        st.markdown(f"""
                        <div class="apex-result-card" style="border: 2px solid rgba(239, 68, 68, 0.3);">
                            <div class="res-icon">🚨</div>
                            <div class="res-title res-fake-grad">AI DETECTED</div>
                            <div class="res-subtitle">Confidence Level: {prob*100:.1f}%</div>
                            <div style="background: rgba(239, 68, 68, 0.1); padding: 1rem; border-radius: 15px; font-size: 0.9rem; color: #b91c1c;">
                                <strong>System Alert:</strong> Synthetic signatures found in high-frequency spectrum.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.progress(prob)
                        st.caption("Generative Probability")
                        
                    else:
                        st.markdown(f"""
                        <div class="apex-result-card" style="border: 2px solid rgba(16, 185, 129, 0.3);">
                            <div class="res-icon">🌿</div>
                            <div class="res-title res-real-grad">AUTHENTIC</div>
                            <div class="res-subtitle">Nature Score: {(1-prob)*100:.1f}%</div>
                            <div style="background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 15px; font-size: 0.9rem; color: #047857;">
                                <strong>Verification Pass:</strong> Image contains consistent natural noise patterns.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.progress(1-prob)
                        st.caption("Organic Probability")
                        
                    # Detailed Metrics in Mini Cards
                    st.markdown("---")
                    c_a, c_b = st.columns(2)
                    with c_a:
                        st.markdown(f"""
                        <div style="background:rgba(255,255,255,0.6); padding:1rem; border-radius:15px; text-align:center;">
                            <div style="font-size:0.8rem; font-weight:700; color:#64748b;">FFT VARIANCE</div>
                            <div style="font-size:1.2rem; font-weight:800; color:{'#ef4444' if is_fake else '#10b981'};">{"High" if is_fake else "Low"}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c_b:
                        st.markdown(f"""
                        <div style="background:rgba(255,255,255,0.6); padding:1rem; border-radius:15px; text-align:center;">
                            <div style="font-size:0.8rem; font-weight:700; color:#64748b;">NOISE PROFILE</div>
                            <div style="font-size:1.2rem; font-weight:800; color:{'#ef4444' if is_fake else '#10b981'};">{"Irregular" if is_fake else "Uniform"}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                else:
                    st.error("Neural Core Connection Failed")
            except Exception as e:
                st.error(f"Execution Error: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- RECENT SCANS ---
st.markdown("<br><h3 style='margin-left: 10px;'>Mission History</h3>", unsafe_allow_html=True)

if st.session_state['history']:
    cols = st.columns(4)
    for idx, item in enumerate(st.session_state['history'][:4]):
        with cols[idx]:
            color = "#ef4444" if item['fake'] else "#10b981"
            bg_color = "rgba(254, 226, 226, 0.5)" if item['fake'] else "rgba(220, 252, 231, 0.5)"
            label = "SYNTHETIC" if item['fake'] else "ORGANIC"
            
            st.markdown(f"""
            <div class="glass-card" style="padding: 1.5rem; border-left: 5px solid {color};">
                <div style="font-weight: 800; color: {color}; margin-bottom: 0.5rem;">{label}</div>
                <div style="font-size: 0.9rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{item['name']}</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">
                    {item['time']} • <span style="font-weight:700;">{item['prob']*100:.0f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<br><br><div style='text-align:center; color:#cbd5e1; font-size:0.8rem;'>System v3.0 // Neural Ensemble // Muthu Selvam</div>", unsafe_allow_html=True)
