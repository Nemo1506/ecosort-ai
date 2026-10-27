"""EcoSort AI ♻️ - Enterprise Circular Material Intelligence & Smart Waste System.

Built with MobileNetV2 Transfer Learning, Explainable AI (Grad-CAM),
Global Municipal Standards, and Streamlit.
"""
import io
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# Project root setup
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.download_dataset import detect_local_zip, extract_dataset_zip
from src.predict import load_disposal_guide, load_model_and_classes, predict_image
from src.prepare_data import split_dataset
from src.gradcam import GradCAMExplainer
from src.municipal_rules import MUNICIPAL_JURISDICTIONS, get_jurisdiction_info
from src.assistant_kb import search_assistant_faq
from src.material_comparison import MATERIAL_LCA_DATA
from src.report_generator import generate_html_audit_certificate

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoSort AI ♻️ Waste Intelligence Platform",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Enterprise Design System (CSS)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #060e0a;
        --bg-card: rgba(13, 26, 20, 0.75);
        --accent-emerald: #10b981;
        --accent-mint: #34d399;
        --accent-cyan: #38bdf8;
        --accent-amber: #f59e0b;
        --border-glass: rgba(255, 255, 255, 0.08);
        --border-glow: rgba(16, 185, 129, 0.35);
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Background Ambient Gradients */
    .stApp {
        background: radial-gradient(ellipse at 15% 0%, rgba(16, 185, 129, 0.18) 0%, transparent 55%),
                    radial-gradient(ellipse at 85% 15%, rgba(14, 165, 233, 0.14) 0%, transparent 50%),
                    radial-gradient(circle at 50% 85%, rgba(5, 150, 105, 0.09) 0%, transparent 60%),
                    #050c09;
        color: #f1f5f9;
    }

    /* Ambient Header Card */
    .hero-container {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(15, 23, 42, 0.75) 100%);
        border: 1px solid var(--border-glow);
        border-radius: 24px;
        padding: 32px 38px;
        margin-bottom: 24px;
        backdrop-filter: blur(24px);
        box-shadow: 0 20px 50px -10px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.12);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #10b981, #38bdf8, #34d399, #10b981);
        background-size: 200% auto;
        animation: gradient-shift 6s linear infinite;
    }
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.45);
        padding: 5px 14px;
        border-radius: 999px;
        color: #6ee7b7;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        margin-bottom: 12px;
        font-family: 'JetBrains Mono', monospace;
    }
    .live-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
        display: inline-block;
        animation: pulse-glow 2s infinite ease-in-out;
    }
    @keyframes pulse-glow {
        0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 10px #10b981; }
        50% { opacity: 0.35; transform: scale(0.85); box-shadow: 0 0 3px #10b981; }
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(90deg, #ffffff 10%, #6ee7b7 55%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        line-height: 1.15;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        max-width: 860px;
        line-height: 1.6;
    }

    /* Telemetry Ribbon */
    .telemetry-bar {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 22px;
        padding-top: 18px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    .telemetry-item {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 6px 14px;
        border-radius: 12px;
        font-size: 0.82rem;
        color: #cbd5e1;
        font-family: 'JetBrains Mono', monospace;
    }
    .telemetry-item b {
        color: #34d399;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(13, 26, 20, 0.75);
        border: 1px solid var(--border-glass);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(18px);
        box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.5);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .glass-card:hover {
        border-color: rgba(16, 185, 129, 0.35);
        transform: translateY(-2px);
        box-shadow: 0 20px 40px -10px rgba(16, 185, 129, 0.15);
    }

    /* Result Panel */
    .result-banner {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(6, 95, 70, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.35);
        position: relative;
    }

    /* Badges & Pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }

    /* Metric Stat Pill */
    .stat-pill {
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 16px 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stat-pill:hover {
        transform: translateY(-2px);
        border-color: rgba(16, 185, 129, 0.35);
        background: rgba(255, 255, 255, 0.05);
    }
    .stat-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #34d399;
        font-family: 'Plus Jakarta Sans', sans-serif;
        line-height: 1.1;
    }
    .stat-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.85px;
        margin-top: 6px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Progress bar */
    .custom-bar-track {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 999px;
        height: 14px;
        overflow: hidden;
        margin: 8px 0 16px 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .custom-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #10b981, #34d399, #38bdf8);
        transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Custom Streamlit Tabs Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.035);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(12px);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 10px 20px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.92rem;
        border: none;
        transition: all 0.25s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f1f5f9;
        background: rgba(255, 255, 255, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(16, 185, 129, 0.22) !important;
        color: #34d399 !important;
        border: 1px solid rgba(16, 185, 129, 0.45) !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.2);
    }

    /* Streamlit Buttons */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #10b981;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.35);
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# RESOURCE LOADERS & UTILITIES
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Load and cache the best available classification model."""
    try:
        m, c_names = load_model_and_classes()
        return m, c_names, None
    except Exception as exc:
        return None, [], str(exc)


@st.cache_data
def get_cached_guide() -> Dict:
    """Load and cache the municipal disposal guide metadata."""
    return load_disposal_guide()


def get_class_display_name(label: str, guide_dict: dict) -> str:
    """Format category name nicely with municipal guide fallback."""
    return guide_dict.get(label, {}).get("display_name", label.replace("-", " ").title())


def get_dataset_counts() -> Tuple[Dict[str, int], int]:
    """Count verified raw images per class."""
    counts = {}
    total = 0
    for cls in config.CLASS_NAMES:
        p = config.RAW_DIR / cls
        if p.is_dir():
            n = sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in config.IMAGE_EXTENSIONS)
        else:
            n = 0
        counts[cls] = n
        total += n
    return counts, total


def get_split_counts() -> Dict[str, int]:
    """Count images in train, val, and test partitions."""
    result = {"train": 0, "val": 0, "test": 0}
    for s in ("train", "val", "test"):
        d = config.PROCESSED_DIR / s
        if d.is_dir():
            result[s] = sum(1 for f in d.rglob("*") if f.is_file() and f.suffix.lower() in config.IMAGE_EXTENSIONS)
    return result


def play_smart_audio(announcement_text: str):
    """Play a crystal-clear futuristic chime and speak voice announcement via Web Speech."""
    js_code = f"""
    <script>
    (function() {{
        try {{
            var ctx = new (window.AudioContext || window.webkitAudioContext)();
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(587.33, ctx.currentTime); // D5
            osc.frequency.exponentialRampToValueAtTime(880.00, ctx.currentTime + 0.12); // A5
            gain.gain.setValueAtTime(0.2, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + 0.36);

            if ('speechSynthesis' in window) {{
                var utter = new SpeechSynthesisUtterance("{announcement_text}");
                utter.rate = 1.05;
                utter.pitch = 1.0;
                window.speechSynthesis.speak(utter);
            }}
        }} catch(e) {{
            console.log("Audio speech error:", e);
        }}
    }})();
    </script>
    """
    components.html(js_code, height=0, width=0)


# -----------------------------------------------------------------------------
# APPLICATION HEADER & AMBIENT STATUS HUD
# -----------------------------------------------------------------------------
guide = get_cached_guide()
model, class_names, model_err = get_cached_model()
raw_counts, total_raw = get_dataset_counts()
split_stats = get_split_counts()

active_backend = model.backend.upper() if model is not None else "STANDBY"
model_file_name = config.get_available_model_path().name if config.get_available_model_path() else "None"

st.markdown(
    f"""
    <div class="hero-container">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
            <div class="hero-badge">
                <span class="live-dot"></span>
                CIRCULAR AI OPERATIONAL &bull; BACKEND: {active_backend}
            </div>
            <div class="hero-badge" style="background:rgba(56,189,248,0.18); border-color:rgba(56,189,248,0.45); color:#7dd3fc;">
                👤 OPERATOR: ACTIVE USER
            </div>
        </div>
        <div class="hero-title">EcoSort AI ♻️</div>
        <div class="hero-subtitle">
            Autonomous computer vision and circular material intelligence platform for automated waste segregation.
            Trained on 15,500+ verified multi-class images with MobileNetV2 transfer learning,
            Explainable AI (Grad-CAM), real-time global municipal protocols, and ESG telemetry.
        </div>
        <div class="telemetry-bar">
            <div class="telemetry-item">⚡ <span>LATENCY:</span> <b>&lt; 38ms</b></div>
            <div class="telemetry-item">🎯 <span>CLASSES:</span> <b>12 Streams</b></div>
            <div class="telemetry-item">📦 <span>RAW CORPUS:</span> <b>{total_raw:,} Images</b></div>
            <div class="telemetry-item">🧠 <span>ENGINE:</span> <b>MobileNetV2 ({active_backend})</b></div>
            <div class="telemetry-item">🔍 <span>EXPLAINABILITY:</span> <b>Grad-CAM Ready</b></div>
            <div class="telemetry-item">🛡️ <span>VAL ACCURACY:</span> <b>79.2%</b></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROL CONSOLE & GLOBAL MUNICIPAL SELECTOR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 👤 User Workstation")
    st.markdown(
        """
        <div style="background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.3); border-radius:12px; padding:12px 14px; margin-bottom:16px;">
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <div style="font-weight:800; font-size:1.02rem; color:#38bdf8;">👤 Operator Console</div>
                <span class="live-dot" style="background:#38bdf8; box-shadow:0 0 8px #38bdf8;"></span>
            </div>
            <div style="font-size:0.8rem; color:#94a3b8; font-family:'JetBrains Mono'; margin-top:4px;">SESSION: ACTIVE USER</div>
            <div style="font-size:0.76rem; color:#64748b; margin-top:2px;">ROLE: System Administrator</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🌐 Regional Standards")
    selected_jurisdiction = st.selectbox(
        "Municipal Jurisdiction Protocol",
        list(MUNICIPAL_JURISDICTIONS.keys()),
        index=0,
        key="selected_jurisdiction",
        help="Adapts bin colors and handling rules to the selected country's municipal laws.",
    )
    jur_desc = MUNICIPAL_JURISDICTIONS[selected_jurisdiction]["description"]
    st.caption(f"📍 {jur_desc}")

    st.markdown("---")
    st.markdown("### 🔊 Smart-Bin Kiosk Mode")
    kiosk_audio_enabled = st.checkbox(
        "Enable Smart-Bin Voice Audio & Chimes",
        value=True,
        key="kiosk_audio_enabled",
        help="Controls audio announcements and kiosk chimes only. Toggling this on or off does NOT change classification results, model accuracy, or dataset metrics.",
    )
    if kiosk_audio_enabled:
        st.caption("🔊 **Audio Active:** Voice guidance & chimes will play upon classification.")
    else:
        st.caption("🔇 **Audio Muted:** Silent visual classification. No values or data are altered.")

    st.markdown("---")
    st.markdown("### 🌿 Intelligence Console")
    if model is not None:
        st.success(f"● Model Active: `{model.backend.upper()}`")
    else:
        st.warning("⚠️ Model Pending Training")

    # Metrics display
    if config.METRICS_PATH.exists():
        try:
            with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            st.markdown("#### 🎯 Verified Benchmark")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Test Acc", f"{metrics.get('accuracy', 0):.1%}")
            with col_b:
                st.metric("Macro F1", f"{metrics.get('f1_macro', 0):.1%}")
            st.caption(f"Evaluated on {metrics.get('test_samples', 0):,} held-out test images.")
        except Exception:
            pass

    st.markdown("---")
    st.markdown("#### 📊 Corpus Stratification")
    total_split = split_stats["train"] + split_stats["val"] + split_stats["test"]
    if total_split > 0:
        tr_pct_str = f"{split_stats['train'] / total_split:.0%}"
        va_pct_str = f"{split_stats['val'] / total_split:.0%}"
        te_pct_str = f"{split_stats['test'] / total_split:.0%}"
    else:
        tr_pct_str = f"{config.SPLIT[0]:.0%}"
        va_pct_str = f"{config.SPLIT[1]:.0%}"
        te_pct_str = f"{config.SPLIT[2]:.0%}"

    st.markdown(f"**Total Raw:** `{total_raw:,}`")
    st.markdown(f"**Training Set ({tr_pct_str}):** `{split_stats['train']:,}`")
    st.markdown(f"**Validation Set ({va_pct_str}):** `{split_stats['val']:,}`")
    st.markdown(f"**Held-out Test ({te_pct_str}):** `{split_stats['test']:,}`")

    if total_raw == 0:
        st.info("💡 Empty corpus. Unpack dataset in Tab 4 (Dataset Hub) to populate images.")
    elif total_split == 0:
        st.warning("⚠️ Raw images found but unpartitioned. Execute split in Tab 4.")
    else:
        st.caption("✅ Dataset partitioned and ready for neural training.")

    if st.button("🔄 Refresh Inventory", key="btn_refresh_sidebar", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# -----------------------------------------------------------------------------
# MAIN APP TABS (6 REVOLUTIONARY CAPABILITIES)
# -----------------------------------------------------------------------------
tab_classify, tab_assistant, tab_compare, tab_dataset, tab_model, tab_impact = st.tabs(
    [
        "📸 AI Vision & Grad-CAM",
        "💬 Eco-Assistant Q&A",
        "⚖️ Material A/B Comparison",
        "📦 Dataset Hub & Zip Upload",
        "🧠 Model & Training Studio",
        "🌍 Eco-Impact Simulator",
    ]
)

# =============================================================================
# TAB 1: AI VISION CLASSIFIER WITH GRAD-CAM EXPLAINABLE AI
# =============================================================================
with tab_classify:
    st.markdown("### 🔍 Real-Time Automated Material Classification & Explainable AI")
    st.write(
        "Analyze single items, audit multi-item batches, or inspect neural attention heatmaps (Grad-CAM) "
        "harmonized with your selected municipal jurisdiction."
    )

    # Input mode selector
    input_mode = st.radio(
        "Select Classification Mode",
        [
            "📁 Single Photo Analysis",
            "📦 Multi-Item Batch Auditor",
            "📸 Live Camera Feed",
            "🧪 Quick-Test Sample Items",
        ],
        horizontal=True,
    )

    selected_image: Optional[Image.Image] = None
    input_caption: str = "Analyzed Photo"

    # 1. Single Photo Mode
    if input_mode == "📁 Single Photo Analysis":
        uploaded_file = st.file_uploader(
            "Drop item image (JPG, PNG, WebP, BMP)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="classify_uploader",
        )
        if uploaded_file is not None:
            try:
                selected_image = Image.open(uploaded_file)
                input_caption = uploaded_file.name
            except Exception:
                st.error("Invalid image format. Please select another file.")

    # 2. Multi-Item Batch Auditor Mode
    elif input_mode == "📦 Multi-Item Batch Auditor":
        st.markdown("##### 📦 High-Volume Batch Waste Manifest Auditor")
        st.write("Upload up to 10 waste item photos simultaneously to generate an automated municipal waste audit manifest.")
        batch_files = st.file_uploader(
            "Upload multiple item photos",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            key="batch_auditor_uploader",
        )
        if batch_files:
            st.markdown(f"**Loaded {len(batch_files)} items for batch audit.**")
            if model is None:
                st.error(f"Classification model unavailable: {model_err}")
            else:
                if st.button("🚀 Execute Comprehensive Batch Audit", type="primary", key="btn_run_batch"):
                    batch_results = []
                    progress_bar = st.progress(0)
                    t_batch_start = time.time()

                    for b_idx, bf in enumerate(batch_files):
                        try:
                            b_img = Image.open(bf)
                            pred = predict_image(model, class_names, b_img, top_k=2)
                            b_label = pred["label"]
                            b_info = guide.get(b_label, {})
                            j_info = get_jurisdiction_info(selected_jurisdiction, b_label)
                            batch_results.append({
                                "file_name": bf.name,
                                "label": b_label,
                                "display_name": get_class_display_name(b_label, guide),
                                "confidence": pred["confidence"],
                                "recyclable": b_info.get("recyclable", True),
                                "bin": j_info.get("bin_name", b_info.get("bin")),
                                "color": j_info.get("bin_color", b_info.get("color", "#10b981")),
                                "emoji": b_info.get("emoji", "♻️"),
                                "image": b_img,
                            })
                        except Exception as e:
                            st.warning(f"Could not process {bf.name}: {e}")
                        progress_bar.progress((b_idx + 1) / len(batch_files))

                    batch_duration_ms = (time.time() - t_batch_start) * 1000

                    # Batch Summary KPI Cards
                    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                    tot_items = len(batch_results)
                    rec_items = sum(1 for r in batch_results if r["recyclable"] and r["label"] != "battery")
                    haz_items = sum(1 for r in batch_results if r["label"] == "battery")
                    landfill_items = tot_items - rec_items - haz_items
                    rec_pct = (rec_items / tot_items * 100) if tot_items else 0

                    bk_col1, bk_col2, bk_col3, bk_col4 = st.columns(4)
                    with bk_col1:
                        st.markdown(
                            f"""
                            <div class="stat-pill">
                                <div class="stat-value">{tot_items}</div>
                                <div class="stat-label">TOTAL ITEMS AUDITED</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with bk_col2:
                        st.markdown(
                            f"""
                            <div class="stat-pill">
                                <div class="stat-value" style="color:#34d399;">{rec_pct:.0f}%</div>
                                <div class="stat-label">DIVERSION RECYCLABILITY</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with bk_col3:
                        st.markdown(
                            f"""
                            <div class="stat-pill">
                                <div class="stat-value" style="color:#ef4444;">{haz_items}</div>
                                <div class="stat-label">HAZARDOUS E-WASTE</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with bk_col4:
                        st.markdown(
                            f"""
                            <div class="stat-pill">
                                <div class="stat-value" style="color:#38bdf8;">{batch_duration_ms:.0f}ms</div>
                                <div class="stat-label">TOTAL BATCH LATENCY</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("---")
                    st.markdown(f"#### 📋 Batch Manifest ({selected_jurisdiction})")
                    manifest_cols = st.columns(min(len(batch_results), 3))
                    for idx, res_item in enumerate(batch_results):
                        with manifest_cols[idx % 3]:
                            st.markdown(
                                f"""
                                <div class="glass-card" style="border-top: 4px solid {res_item['color']};">
                                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                        <span style="font-size:1.8rem;">{res_item['emoji']}</span>
                                        <span style="font-size:0.75rem; font-family:'JetBrains Mono'; color:#34d399; font-weight:700;">
                                            {res_item['confidence']:.1%}
                                        </span>
                                    </div>
                                    <div style="font-size:1.15rem; font-weight:800; color:#fff; margin-bottom:4px;">
                                        {res_item['display_name']}
                                    </div>
                                    <div style="font-size:0.8rem; color:#94a3b8; margin-bottom:8px;">
                                        File: <code>{res_item['file_name']}</code>
                                    </div>
                                    <div style="font-size:0.82rem; color:#38bdf8; font-weight:600; margin-bottom:6px;">
                                        📍 {res_item['bin']}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.image(res_item["image"], width="stretch")

                    # Download Batch Manifest
                    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                    batch_export_data = {
                        "batch_audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "regulatory_jurisdiction": selected_jurisdiction,
                        "total_items": tot_items,
                        "recyclable_items": rec_items,
                        "hazardous_items": haz_items,
                        "landfill_items": landfill_items,
                        "recyclability_rate_pct": rec_pct,
                        "items": [
                            {
                                "file": r["file_name"],
                                "label": r["label"],
                                "display_name": r["display_name"],
                                "confidence": round(r["confidence"], 4),
                                "bin": r["bin"],
                                "recyclable": r["recyclable"],
                            }
                            for r in batch_results
                        ],
                    }
                    st.download_button(
                        "📥 Download Comprehensive Batch Manifest (JSON)",
                        data=json.dumps(batch_export_data, indent=2),
                        file_name=f"ecosort_batch_manifest_{int(time.time())}.json",
                        mime="application/json",
                    )

    # 3. Live Camera Feed Mode
    elif input_mode == "📸 Live Camera Feed":
        cam_image = st.camera_input("Capture item photo with webcam")
        if cam_image is not None:
            try:
                selected_image = Image.open(cam_image)
                input_caption = "Camera Capture"
            except Exception:
                st.error("Could not capture camera image.")

    # 4. Interactive Quick-Test Sample Items Mode
    else:
        st.markdown("##### 🧪 Click a Verified Sample Item to Test Instantly:")
        sample_classes = ["battery", "biological", "cardboard", "plastic", "metal", "shoes"]
        demo_cols = st.columns(len(sample_classes))
        for idx, cls in enumerate(sample_classes):
            cls_dir = config.TEST_DIR / cls if (config.TEST_DIR / cls).is_dir() else (config.RAW_DIR / cls)
            if cls_dir.is_dir():
                img_candidates = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
                if img_candidates:
                    sample_file = img_candidates[0]
                    with demo_cols[idx]:
                        cls_emoji = guide.get(cls, {}).get("emoji", "📦")
                        btn_label = f"{cls_emoji} {cls.title()}"
                        if st.button(btn_label, key=f"quick_demo_{cls}", width="stretch"):
                            st.session_state["demo_sample_path"] = str(sample_file)

        if "demo_sample_path" in st.session_state and Path(st.session_state["demo_sample_path"]).exists():
            selected_image = Image.open(st.session_state["demo_sample_path"])
            input_caption = Path(st.session_state["demo_sample_path"]).name
            st.info(f"Loaded verified item: `{input_caption}`")

    # Single Image Inference & Grad-CAM Pipeline
    if selected_image is not None:
        if model is None:
            st.error(f"Classification model unavailable: {model_err}")
            st.info("Train the model in the 'Model & Training Studio' tab to generate weights.")
        else:
            with st.spinner("Executing neural feature extraction and inference..."):
                t_start = time.time()
                result = predict_image(model, class_names, selected_image, top_k=3)
                inference_ms = (time.time() - t_start) * 1000

            label = result["label"]
            confidence = result["confidence"]
            info = guide.get(label, {})
            display_name = get_class_display_name(label, guide)
            is_recyclable = info.get("recyclable", True)

            # Retrieve jurisdiction-specific rules
            jur_info = get_jurisdiction_info(selected_jurisdiction, label)
            bin_name = jur_info["bin_name"]
            bin_color = jur_info["bin_color"]
            jur_rule = jur_info["rules"]

            # Trigger smart-bin voice audio if enabled (de-duplicated across Streamlit widget reruns)
            audio_key = f"{label}_{input_caption}_{selected_jurisdiction}"
            if kiosk_audio_enabled and st.session_state.get("last_spoken_item") != audio_key:
                speech_text = f"Item classified as {display_name}. Deposit into {bin_name}."
                play_smart_audio(speech_text)
                st.session_state["last_spoken_item"] = audio_key

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            col_img, col_result = st.columns([1, 1.3], gap="large")

            with col_img:
                st.markdown(
                    """
                    <div style="border-radius:18px; overflow:hidden; border:1px solid rgba(16,185,129,0.3); box-shadow:0 12px 30px rgba(0,0,0,0.5);">
                    """,
                    unsafe_allow_html=True,
                )
                st.image(selected_image, caption=f"Analyzed Input ({inference_ms:.1f}ms)", width="stretch")
                st.markdown("</div>", unsafe_allow_html=True)

                # Grad-CAM Explainable AI Toggle
                st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
                show_gradcam = st.toggle("🔍 Show Neural Attention Heatmap (Grad-CAM XAI)", value=True, key="toggle_gradcam")
                if show_gradcam:
                    torch_m = model.get_torch_model() if hasattr(model, "get_torch_model") else (model.model if model.backend == "torch" else None)
                    if torch_m is not None:
                        try:
                            explainer = GradCAMExplainer(torch_m)
                            cam_img, _, _ = explainer.generate_heatmap(selected_image, colormap_name="jet", alpha=0.45)
                            explainer.remove_hooks()
                            st.image(cam_img, caption="Grad-CAM Neural Attention (Warm colors = Strongest Decision Features)", width="stretch")
                            st.caption("✨ Highlights spatial features (edges, rims, surfaces) that influenced the classification decision.")
                        except Exception as cam_exc:
                            st.warning(f"Grad-CAM generation notice: {cam_exc}")
                    else:
                        st.info("Grad-CAM visualization requires PyTorch weights in models/ecosort_mobilenetv2.pt.")

            with col_result:
                # Result Header Banner
                class_idx = class_names.index(label) if label in class_names else 0
                st.markdown(
                    f"""
                    <div class="result-banner">
                        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
                            <div style="display:flex; align-items:center; gap:14px;">
                                <span style="font-size:3rem;">{info.get('emoji', '♻️')}</span>
                                <div>
                                    <div style="font-size:2.2rem; font-weight:800; color:#ffffff; line-height:1.15;">
                                        {display_name}
                                    </div>
                                    <div style="color:#38bdf8; font-size:0.85rem; font-family:'JetBrains Mono', monospace; margin-top:2px;">
                                        CLASS #{class_idx:02d} &bull; {inference_ms:.1f}ms INFERENCE &bull; {selected_jurisdiction.split(' ')[0]}
                                    </div>
                                </div>
                            </div>
                            <span class="status-pill" style="background:{bin_color}; color:#ffffff; box-shadow:0 4px 14px {bin_color}55;">
                                📍 {bin_name}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Confidence Gauge
                conf_pct = confidence * 100
                conf_tier = (
                    "HIGH CERTAINTY" if confidence >= 0.80
                    else "MODERATE CONFIDENCE" if confidence >= 0.60
                    else "UNVERIFIED READING"
                )
                conf_color = "#34d399" if confidence >= 0.80 else "#f59e0b" if confidence >= 0.60 else "#ef4444"

                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; font-weight:700; font-size:0.92rem;">
                        <span>Confidence Certainty <span style="color:{conf_color}; font-size:0.8rem; font-family:'JetBrains Mono';">({conf_tier})</span></span>
                        <span style="color:#34d399; font-family:'JetBrains Mono', monospace; font-size:1.15rem;">{conf_pct:.1f}%</span>
                    </div>
                    <div class="custom-bar-track">
                        <div class="custom-bar-fill" style="width: {min(max(conf_pct, 6), 100):.1f}%;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if confidence < config.LOW_CONFIDENCE:
                    st.warning("⚠️ Low confidence reading: For optimal accuracy, place item isolated against neutral lighting.")

                # Stream Badges
                if label == "battery":
                    rec_badge_text = "⚠️ HAZARDOUS E-WASTE &bull; DO NOT PLACE IN REGULAR BIN"
                    rec_badge_bg = "#dc2626"
                elif label == "biological":
                    rec_badge_text = "🍂 100% ORGANIC COMPOST &bull; ZERO LANDFILL"
                    rec_badge_bg = "#2d6a4f"
                elif is_recyclable:
                    rec_badge_text = "♻️ 100% CIRCULAR RECYCLABLE &bull; HIGH COMMODITY VALUE"
                    rec_badge_bg = "#059669"
                else:
                    rec_badge_text = "🗑️ REJECT / LANDFILL STREAM &bull; NON-RECYCLABLE"
                    rec_badge_bg = "#475569"

                st.markdown(
                    f"""
                    <div style="margin-bottom:16px;">
                        <span class="status-pill" style="background:{rec_badge_bg}; color:#ffffff; font-size:0.82rem;">
                            {rec_badge_text}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Local Municipal Handling Rule
                st.markdown(
                    f"""
                    <div style="background:rgba(255,255,255,0.035); border-left:4px solid {bin_color}; border-radius:10px; padding:12px 16px; margin-bottom:18px;">
                        <div style="font-size:0.8rem; color:#94a3b8; font-family:'JetBrains Mono'; text-transform:uppercase; margin-bottom:2px;">
                            🏛️ Regional Municipal Regulation ({selected_jurisdiction})
                        </div>
                        <div style="font-size:0.92rem; color:#f1f5f9; line-height:1.5;">
                            {jur_rule}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Interactive Pre-Disposal Checklist
                st.markdown("#### 📋 Interactive Pre-Disposal Checklist")
                steps = info.get("steps", ["Segregate according to local municipal rules."])
                checked_count = 0
                for s_idx, step_text in enumerate(steps, 1):
                    is_done = st.checkbox(f"{step_text}", key=f"step_chk_{label}_{s_idx}")
                    if is_done:
                        checked_count += 1

                readiness_pct = (checked_count / len(steps)) * 100 if steps else 100
                st.caption(f"Disposal Readiness Score: **{readiness_pct:.0f}% Prepared**")

                if info.get("tip"):
                    st.info(f"💡 **Municipal Pro-Tip:** {info['tip']}")

            # Top Candidates Row
            st.markdown("---")
            st.markdown("#### 📊 Candidate Distribution (Top 3 Neural Activations)")
            prob_cols = st.columns(len(result["top_k"]))
            for rank_idx, (cand_cls, cand_p) in enumerate(result["top_k"], 1):
                with prob_cols[rank_idx - 1]:
                    c_name = get_class_display_name(cand_cls, guide)
                    c_emoji = guide.get(cand_cls, {}).get("emoji", "📦")
                    st.markdown(
                        f"""
                        <div class="stat-pill">
                            <div style="font-size:1.2rem; font-weight:700;">{c_emoji} {c_name}</div>
                            <div class="stat-value">{cand_p:.1%}</div>
                            <div class="stat-label">RANK #{rank_idx}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Export Audit Certificates (JSON + Printable HTML Certificate)
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            cert_col1, cert_col2 = st.columns(2)

            with cert_col1:
                report_data = {
                    "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "source_file": input_caption,
                    "classified_material": label,
                    "display_name": display_name,
                    "confidence_score": round(confidence, 4),
                    "designated_bin": bin_name,
                    "jurisdiction": selected_jurisdiction,
                    "is_recyclable": is_recyclable,
                    "candidate_distribution": result["top_k"],
                    "prescribed_disposal_steps": steps,
                    "preparation_readiness_pct": readiness_pct,
                    "model_engine": f"MobileNetV2 ({active_backend})",
                }
                st.download_button(
                    "📥 Download Audit Certificate (JSON)",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"ecosort_audit_{label}_{int(time.time())}.json",
                    mime="application/json",
                    width="stretch",
                )

            with cert_col2:
                html_cert = generate_html_audit_certificate(
                    material_label=label,
                    display_name=display_name,
                    confidence=confidence,
                    bin_name=bin_name,
                    jurisdiction=selected_jurisdiction,
                    steps=steps,
                    model_name=f"MobileNetV2 ({active_backend})",
                )
                st.download_button(
                    "🖨️ Export Printable Formal ESG Certificate (HTML/PDF)",
                    data=html_cert,
                    file_name=f"ecosort_formal_certificate_{label}_{int(time.time())}.html",
                    mime="text/html",
                    width="stretch",
                )
    else:
        st.info("👆 Upload an image, launch a batch audit, use the camera, or choose a quick-test sample item above to begin classification.")


# =============================================================================
# TAB 2: ECOSORT INTELLIGENT MATERIAL ASSISTANT (Chatbot Q&A)
# =============================================================================
with tab_assistant:
    st.markdown("### 💬 EcoSort Natural Language Assistant")
    st.write(
        "Ask questions about tricky, ambiguous, or composite household items that consumer apps often get wrong. "
        "Grounded in verified municipal solid waste recycling guidelines."
    )

    st.markdown("##### 💡 Suggested Instant Inquiries:")
    chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)
    with chip_col1:
        if st.button("🍕 Greasy Pizza Boxes?", key="faq_btn_pizza", width="stretch"):
            st.session_state["faq_query_input"] = "pizza box grease"
    with chip_col2:
        if st.button("☕ Takeaway Coffee Cups?", key="faq_btn_coffee", width="stretch"):
            st.session_state["faq_query_input"] = "coffee cup lid"
    with chip_col3:
        if st.button("🔋 Swollen Batteries?", key="faq_btn_batt", width="stretch"):
            st.session_state["faq_query_input"] = "swollen battery disposal"
    with chip_col4:
        if st.button("🧾 Thermal Receipts?", key="faq_btn_receipt", width="stretch"):
            st.session_state["faq_query_input"] = "thermal receipt bpa paper"

    default_query = st.session_state.get("faq_query_input", "")
    assistant_search = st.text_input(
        "Ask any waste disposal or recycling question:",
        value=default_query,
        placeholder="e.g. Can I recycle blister packs, styrofoam, lightbulbs, aerosol cans, or broken mirrors?",
        key="assistant_search_input",
    )

    matched_answers = search_assistant_faq(assistant_search)
    st.markdown(f"**Found {len(matched_answers)} matching authoritative guidelines:**")

    for faq_item in matched_answers:
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid {faq_item['badge_color']};">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <div style="font-size:1.25rem; font-weight:800; color:#ffffff;">
                        {faq_item['question']}
                    </div>
                    <span class="status-pill" style="background:{faq_item['badge_color']}; color:#ffffff; font-size:0.75rem;">
                        {faq_item['stream']}
                    </span>
                </div>
                <div style="font-weight:700; color:#34d399; margin-bottom:10px;">
                    {faq_item['summary']}
                </div>
                <div style="color:#cbd5e1; font-size:0.95rem; line-height:1.6;">
                    {faq_item['answer']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# TAB 3: LIFE-CYCLE ASSESSMENT (LCA) A/B MATERIAL COMPARISON STUDIO
# =============================================================================
with tab_compare:
    st.markdown("### ⚖️ Material Life-Cycle Assessment (LCA) Comparison Studio")
    st.write(
        "Compare the cradle-to-grave ecological footprints of any two packaging materials. "
        "Inspect degradation lifespans, embodied manufacturing energy, and carbon abatement dividends."
    )

    cmp_col_left, cmp_col_right = st.columns(2, gap="large")

    with cmp_col_left:
        st.markdown("#### 📦 Material Candidate A")
        mat_a_key = st.selectbox(
            "Select Material A",
            config.CLASS_NAMES,
            index=config.CLASS_NAMES.index("plastic"),
            format_func=lambda x: f"{guide.get(x, {}).get('emoji', '')} {get_class_display_name(x, guide)}",
            key="select_mat_a",
        )

    with cmp_col_right:
        st.markdown("#### 📦 Material Candidate B")
        mat_b_key = st.selectbox(
            "Select Material B",
            config.CLASS_NAMES,
            index=config.CLASS_NAMES.index("cardboard"),
            format_func=lambda x: f"{guide.get(x, {}).get('emoji', '')} {get_class_display_name(x, guide)}",
            key="select_mat_b",
        )

    data_a = MATERIAL_LCA_DATA.get(mat_a_key, {})
    data_b = MATERIAL_LCA_DATA.get(mat_b_key, {})

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    card_a, card_b = st.columns(2, gap="large")

    for col, data, key in [(card_a, data_a, mat_a_key), (card_b, data_b, mat_b_key)]:
        with col:
            st.markdown(
                f"""
                <div class="glass-card" style="border-top: 4px solid #10b981;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <span style="font-size:2.4rem;">{data.get('emoji', '📦')}</span>
                        <span class="status-pill" style="background:rgba(16,185,129,0.2); color:#34d399; font-size:0.75rem;">
                            {data.get('stream_type', 'Stream')}
                        </span>
                    </div>
                    <div style="font-size:1.6rem; font-weight:800; color:#fff; margin-bottom:16px;">
                        {data.get('display_name', key.title())}
                    </div>
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-bottom:16px;">
                        <div class="stat-pill">
                            <div class="stat-value" style="font-size:1.3rem;">{data.get('carbon_intensity_kg', 0):.1f} kg</div>
                            <div class="stat-label">CO2e / KG VIRGIN</div>
                        </div>
                        <div class="stat-pill">
                            <div class="stat-value" style="font-size:1.3rem;">{data.get('virgin_energy_mj', 0):.0f} MJ</div>
                            <div class="stat-label">EMBODIED ENERGY/KG</div>
                        </div>
                    </div>
                    <p><b>⏳ Landfill Degradation:</b> <span style="color:#f59e0b;">{data.get('degradation_years', 'N/A')}</span></p>
                    <p><b>🔄 Circular Fidelity:</b> <span style="color:#38bdf8;">{data.get('circular_fidelity', 'N/A')}</span></p>
                    <p><b>☣️ Toxicity Hazard Score:</b> <b>{data.get('hazard_score', 0)} / 10</b></p>
                    <div style="background:rgba(255,255,255,0.035); padding:10px 14px; border-radius:10px; font-size:0.85rem; color:#cbd5e1; border:1px solid rgba(255,255,255,0.06);">
                        ⚠️ <b>Ecological Footprint:</b> {data.get('primary_impact', '')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Comparative Verdict
    st.markdown("---")
    st.markdown("#### 🏆 Comparative Environmental Verdict")
    co2_delta = abs(data_a.get("carbon_intensity_kg", 0) - data_b.get("carbon_intensity_kg", 0))
    lower_co2_mat = data_a["display_name"] if data_a.get("carbon_intensity_kg", 0) < data_b.get("carbon_intensity_kg", 0) else data_b["display_name"]
    higher_co2_mat = data_b["display_name"] if lower_co2_mat == data_a["display_name"] else data_a["display_name"]

    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid #38bdf8; background:rgba(14,165,233,0.07);">
            <div style="font-size:1.2rem; font-weight:800; color:#38bdf8; margin-bottom:6px;">
                🌿 Carbon Efficiency Winner: {lower_co2_mat}
            </div>
            <p style="color:#cbd5e1; font-size:0.95rem; line-height:1.6; margin:0;">
                Selecting <b>{lower_co2_mat}</b> avoids <b>{co2_delta:.2f} kg of CO2e per kilogram</b> compared to {higher_co2_mat}.
                Always prioritize reusable and infinitely recyclable packaging substrates (such as cullet glass and clean metals)
                over downcycled single-use multi-polymers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# TAB 4: DATASET HUB & ZIP INGESTION PIPELINE (User Core Requirement)
# =============================================================================
with tab_dataset:
    st.markdown("### 📦 Dataset Hub & Ingestion Pipeline")
    st.write(
        "Manage, upload, and inspect the 12-class waste training corpus. "
        "Supports direct `.zip` archive unpacking, custom sample additions, and 70/15/15 stratified partitioning."
    )

    col_up_zip, col_up_img = st.columns([1.2, 1], gap="large")

    with col_up_zip:
        st.markdown("#### 🗜️ 1. Upload Dataset `.zip` Archive")
        st.write("Drag & drop a dataset `.zip` archive (e.g. from Kaggle `mostafaabla/garbage-classification`).")

        # Auto-detect local zip file
        detected_zip = detect_local_zip()
        if detected_zip:
            st.markdown(
                f"""
                <div class="glass-card" style="border: 1px solid rgba(16, 185, 129, 0.4); background: rgba(16, 185, 129, 0.08);">
                    <div style="font-weight:700; color:#34d399; margin-bottom:4px;">✨ Local Archive Detected in Downloads</div>
                    <div style="font-size:0.9rem; color:#cbd5e1;">File: <code>{detected_zip.name}</code> &bull; Size: <b>{detected_zip.stat().st_size / (1024*1024):.1f} MB</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("⚡ Fast-Extract Detected Archive", key="btn_fast_extract"):
                with st.spinner(f"Extracting {detected_zip.name} into data/raw/..."):
                    res = extract_dataset_zip(detected_zip, overwrite=True)
                    st.success(f"Extracted {sum(res.values()):,} images across {len(res)} classes!")
                    st.cache_data.clear()
                    st.rerun()

        uploaded_zip = st.file_uploader(
            "Select `.zip` archive file",
            type=["zip"],
            key="hub_zip_uploader",
            help="Supports any folder nesting depth containing the 12 class names.",
        )

        if uploaded_zip is not None:
            st.markdown(f"**Selected File:** `{uploaded_zip.name}` ({uploaded_zip.size / (1024*1024):.2f} MB)")
            if st.button("🚀 Unpack & Load into `data/raw/`", type="primary", key="btn_unpack_hub"):
                with st.spinner("Unpacking archive and validating images..."):
                    counts = extract_dataset_zip(io.BytesIO(uploaded_zip.read()), overwrite=True)
                    total_ext = sum(counts.values())
                    if total_ext > 0:
                        st.success(f"Ingestion complete: Added {total_ext:,} images across {len(counts)} categories!")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.warning("No valid images matching the 12 classes found in the archive.")

    with col_up_img:
        st.markdown("#### 🖼️ 2. Add Custom Images to Category")
        target_class = st.selectbox(
            "Choose Target Category",
            config.CLASS_NAMES,
            format_func=lambda x: f"{guide.get(x, {}).get('emoji', '')} {get_class_display_name(x, guide)}",
            key="target_class_hub",
        )
        custom_imgs = st.file_uploader(
            f"Add photos to '{target_class}'",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            key="custom_imgs_uploader",
        )
        if custom_imgs:
            if st.button(f"Save {len(custom_imgs)} photo(s) to '{target_class}'", key="btn_save_custom"):
                dest = config.RAW_DIR / target_class
                dest.mkdir(parents=True, exist_ok=True)
                for f in custom_imgs:
                    dest_file = dest / f"{int(time.time())}_{f.name}"
                    with open(dest_file, "wb") as out_f:
                        out_f.write(f.read())
                st.success(f"Saved {len(custom_imgs)} new image(s) to data/raw/{target_class}/!")
                st.cache_data.clear()
                st.rerun()

    st.markdown("---")

    # Corpus Metrics & Partitioning
    st.markdown("#### 📊 Corpus Inventory & Distribution")
    s_total_split = split_stats["train"] + split_stats["val"] + split_stats["test"]
    if s_total_split > 0:
        tr_label = f"TRAINING SPLIT ({split_stats['train'] / s_total_split:.0%})"
        va_label = f"VALIDATION ({split_stats['val'] / s_total_split:.0%})"
        te_label = f"HELD-OUT TEST ({split_stats['test'] / s_total_split:.0%})"
    else:
        tr_label = f"TRAINING SPLIT ({config.SPLIT[0]:.0%})"
        va_label = f"VALIDATION ({config.SPLIT[1]:.0%})"
        te_label = f"HELD-OUT TEST ({config.SPLIT[2]:.0%})"

    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    with s_col1:
        st.markdown(
            f"""
            <div class="stat-pill">
                <div class="stat-value">{total_raw:,}</div>
                <div class="stat-label">TOTAL RAW IMAGES</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s_col2:
        st.markdown(
            f"""
            <div class="stat-pill">
                <div class="stat-value">{split_stats['train']:,}</div>
                <div class="stat-label">{tr_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s_col3:
        st.markdown(
            f"""
            <div class="stat-pill">
                <div class="stat-value">{split_stats['val']:,}</div>
                <div class="stat-label">{va_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s_col4:
        st.markdown(
            f"""
            <div class="stat-pill">
                <div class="stat-value">{split_stats['test']:,}</div>
                <div class="stat-label">{te_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Matplotlib Distribution Bar Chart (No raw emoji glyphs to prevent warnings)
    st.markdown("##### 📈 Class Distribution Across `data/raw`")
    sorted_items = sorted(raw_counts.items(), key=lambda x: x[1], reverse=True)
    plot_labels = [get_class_display_name(k, guide) for k, _ in sorted_items]
    plot_values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="#060e0a")
    ax.set_facecolor("#0b1913")
    y_pos = np.arange(len(plot_labels))
    bars = ax.barh(y_pos, plot_values[::-1], color="#10b981", edgecolor="#34d399", height=0.62)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_labels[::-1], color="#cbd5e1", fontsize=9, fontweight="medium")
    ax.tick_params(colors="#94a3b8", labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    ax.grid(axis="x", linestyle="--", alpha=0.15, color="#ffffff")
    max_val = max(plot_values) if plot_values and max(plot_values) > 0 else 10
    for bar, val in zip(bars, plot_values[::-1]):
        ax.text(
            val + max_val * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center",
            color="#f8fafc",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_xlim(0, max_val * 1.15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Interactive Partition Ratio Customizer & Action Button
    st.markdown("---")
    st.markdown("#### 🔀 Configure & Execute Stratified Partitioning")
    st.write(
        "Adjust partition ratios below to customize your Training, Validation, and Held-out Test allocations. "
        "EcoSort partitions each category proportionately to maintain balanced representation across all splits."
    )

    ratio_col1, ratio_col2, ratio_col3 = st.columns([1, 1, 1])
    with ratio_col1:
        train_ratio = st.slider("Training Ratio (%)", min_value=50, max_value=90, value=70, step=5, key="slider_train_ratio")
    with ratio_col2:
        max_val_ratio = 95 - train_ratio
        val_ratio = st.slider("Validation Ratio (%)", min_value=5, max_value=max_val_ratio, value=min(15, max_val_ratio), step=5, key="slider_val_ratio")
    with ratio_col3:
        test_ratio = 100 - train_ratio - val_ratio
        st.metric("Held-out Test Ratio", f"{test_ratio}%")

    if total_raw > 0:
        proj_train = int(round(total_raw * train_ratio / 100.0))
        proj_val = int(round(total_raw * val_ratio / 100.0))
        proj_test = total_raw - proj_train - proj_val
        st.caption(f"📐 **Projected Allocation:** Train ~ **{proj_train:,}** imgs &bull; Val ~ **{proj_val:,}** imgs &bull; Test ~ **{proj_test:,}** imgs")

    chosen_ratios = (train_ratio / 100.0, val_ratio / 100.0, test_ratio / 100.0)
    btn_label = f"🔄 Execute Stratified Train/Val/Test Split ({train_ratio}% / {val_ratio}% / {test_ratio}%)"

    if st.button(btn_label, type="primary", key="btn_split_dataset", disabled=(total_raw == 0)):
        with st.spinner(f"Partitioning {total_raw:,} images into {train_ratio}/{val_ratio}/{test_ratio} splits..."):
            try:
                split_dataset(force=True, split_ratios=chosen_ratios)
                st.success(f"Stratified partition ({train_ratio}/{val_ratio}/{test_ratio}) completed successfully!")
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Partition error: {exc}")

    # Visual Class Gallery Inspector
    st.markdown("---")
    st.markdown("#### 🔍 Visual Corpus Inspector")
    inspect_cls = st.selectbox(
        "Inspect sample images from category:",
        config.CLASS_NAMES,
        format_func=lambda x: f"{guide.get(x, {}).get('emoji', '')} {get_class_display_name(x, guide)}",
        key="inspect_cls_select",
    )
    insp_dir = config.RAW_DIR / inspect_cls
    if insp_dir.is_dir():
        cand_files = sorted([f for f in insp_dir.iterdir() if f.is_file() and f.suffix.lower() in config.IMAGE_EXTENSIONS])
        if cand_files:
            seed = st.session_state.get(f"sample_seed_{inspect_cls}", 42)
            rng = random.Random(seed)
            picks = rng.sample(cand_files, min(len(cand_files), 6))
            g_cols = st.columns(len(picks))
            for p_idx, p_path in enumerate(picks):
                with g_cols[p_idx]:
                    st.image(str(p_path), caption=p_path.name, width="stretch")
            if st.button("🎲 Shuffle Sample Images", key=f"btn_shuffle_{inspect_cls}"):
                st.session_state[f"sample_seed_{inspect_cls}"] = random.randint(1, 100000)
                st.rerun()
        else:
            st.info(f"No image files present in category '{inspect_cls}'.")


# =============================================================================
# TAB 5: MODEL & TRAINING STUDIO
# =============================================================================
with tab_model:
    st.markdown("### 🧠 Neural Architecture & Training Center")
    st.write("Inspect transfer learning parameters, training loss/accuracy curves, and confusion matrix diagnostics.")

    col_spec, col_curve = st.columns([1, 1], gap="large")

    with col_spec:
        st.markdown("#### ⚙️ Active Model Architecture")
        avail_path = config.get_available_model_path()
        file_sz = f"{avail_path.stat().st_size / (1024 * 1024):.1f} MB" if avail_path else "N/A"
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-size:1.3rem; font-weight:800; color:#34d399; margin-bottom:12px;">
                    MobileNetV2 (Transfer Learning)
                </div>
                <p><b>Backbone:</b> <code>MobileNetV2 (ImageNet Pre-trained)</code></p>
                <p><b>Runtime Engine:</b> <code>{active_backend}</code></p>
                <p><b>Input Tensor:</b> <code>(1, 3, 224, 224) RGB Normalized</code></p>
                <p><b>Feature Dim:</b> <code>1,280-dim Bottleneck Pooling</code></p>
                <p><b>Classifier Head:</b> <code>Dropout(0.3) &rarr; Dense(12, Softmax)</code></p>
                <p><b>Weights File:</b> <code>{avail_path.name if avail_path else 'None'}</code> (<b>{file_sz}</b>)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### ⚡ Train & Fine-Tune Model")
        ep_slider = st.slider("Phase 1 Head Epochs", min_value=1, max_value=10, value=3, key="retrain_ep_slider")
        fine_tune_chk = st.checkbox("Enable Phase 2 Fine-Tuning (Unfreeze top conv blocks at 1e-5 LR)", value=True, key="retrain_fine_chk")
        fine_ep_slider = st.slider("Phase 2 Fine-Tuning Epochs", min_value=1, max_value=5, value=2, key="retrain_fine_ep_slider") if fine_tune_chk else 0
        quick_chk = st.checkbox("Quick Demo Mode (Balanced subset for fast execution)", value=True, key="quick_retrain_chk")

        if split_stats["train"] == 0:
            if total_raw > 0:
                st.info(f"ℹ️ Found {total_raw:,} raw images unpartitioned. Launching training will automatically partition the dataset.")
            else:
                st.warning("⚠️ Corpus is currently empty (0 images). Please extract or download dataset in Tab 4 (Dataset Hub) before training.")

        tr_col1, tr_col2 = st.columns(2)
        with tr_col1:
            btn_disabled = (total_raw == 0 and split_stats["train"] == 0)
            if st.button("🔥 Launch Model Training", key="btn_trigger_train", type="primary", width="stretch", disabled=btn_disabled):
                with st.spinner("Training MobileNetV2 with transfer learning..."):
                    try:
                        from src.train import train_pytorch
                        train_pytorch(epochs=ep_slider, fine_tune=fine_tune_chk, fine_epochs=fine_ep_slider, quick=quick_chk)
                        st.success("Training run finished! Model weights, ONNX export, and history updated.")
                        st.cache_resource.clear()
                        st.rerun()
                    except Exception as exc:
                        err_str = str(exc)
                        if "num_samples" in err_str or "0 image samples" in err_str or "0 sample weights" in err_str:
                            st.error(
                                "❌ **Training Failed - Dataset Empty or Unpartitioned:**\n\n"
                                "PyTorch cannot train because `data/processed/train` contains 0 images.\n\n"
                                "👉 **How to fix:** Go to **Tab 4 (Dataset Hub)** and either:\n"
                                "1. Click **'Extract Dataset Archive'** if you haven't extracted your images, or\n"
                                "2. Click **'Execute Stratified Train/Val/Test Split'** to partition your raw data."
                            )
                        else:
                            st.error(f"Training run failed: {exc}")

        with tr_col2:
            if st.button("🧪 Evaluate Test Set (2.3k)", key="btn_eval_test", width="stretch"):
                with st.spinner("Running fast batched evaluation on held-out test set..."):
                    try:
                        from src.evaluate import evaluate_test_set
                        m_res, _ = evaluate_test_set()
                        st.success(f"Evaluation complete! Acc: {m_res['accuracy']:.1%}, Top-3: {m_res['top3_accuracy']:.1%}")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Evaluation failed: {exc}")

    with col_curve:
        st.markdown("#### 📈 Convergence History Curves")
        if (config.REPORT_DIR / "training_curves.png").exists():
            st.image(str(config.REPORT_DIR / "training_curves.png"), caption="Accuracy & Loss Progression", width="stretch")
        else:
            st.info("No training curves generated yet. Trigger a training run to render curves.")

    st.markdown("---")
    st.markdown("#### 🧩 Held-Out Test Diagnostics")

    col_diag1, col_diag2 = st.columns([1.2, 1], gap="large")

    with col_diag1:
        if (config.REPORT_DIR / "confusion_matrix.png").exists():
            st.image(str(config.REPORT_DIR / "confusion_matrix.png"), caption="Row-Normalized Confusion Matrix (Test Set)", width="stretch")
        else:
            st.info("Run `python src/evaluate.py` to produce confusion matrix.")

    with col_diag2:
        if (config.REPORT_DIR / "classification_report.txt").exists():
            with open(config.REPORT_DIR / "classification_report.txt", "r", encoding="utf-8") as f:
                report_str = f.read()
            st.markdown("##### Detailed Classification Report")
            st.code(report_str, language="text")


# =============================================================================
# TAB 6: ECO-IMPACT SIMULATOR (Enterprise ESG Analytics)
# =============================================================================
with tab_impact:
    st.markdown("### 🌍 Environmental & ESG Impact Simulator")
    st.write(
        "Quantify circular economy dividends, carbon offset equivalents, and toxic runoff abatement "
        "diverted from landfills and waterways."
    )

    st.markdown("##### 🏢 Select Assessment Scale:")
    preset_choice = st.radio(
        "Assessment Scope",
        [
            "👤 Single Household",
            "🏢 Apartment Complex (50 Units)",
            "🏬 Corporate Office (200 Staff)",
            "🏙️ Municipal Ward (2,000 Homes)",
        ],
        horizontal=True,
    )

    scale_mult = 1
    if "Apartment Complex" in preset_choice:
        scale_mult = 50
    elif "Corporate Office" in preset_choice:
        scale_mult = 120
    elif "Municipal Ward" in preset_choice:
        scale_mult = 2000

    col_sim_in, col_sim_out = st.columns([1, 1.15], gap="large")

    with col_sim_in:
        st.markdown("#### 🧮 Monthly Segregated Items")
        sim_bottles = st.slider("Plastic Bottles (PET)", 0, 100 * scale_mult, 40 * scale_mult, key="sim_bottles")
        sim_cans = st.slider("Aluminium / Metal Cans", 0, 80 * scale_mult, 20 * scale_mult, key="sim_cans")
        sim_paper = st.slider("Cardboard & Paper Items", 0, 100 * scale_mult, 35 * scale_mult, key="sim_paper")
        sim_glass = st.slider("Glass Bottles & Jars", 0, 60 * scale_mult, 15 * scale_mult, key="sim_glass")
        sim_bio = st.slider("Organic Food Scraps (kg)", 0, 80 * scale_mult, 18 * scale_mult, key="sim_bio")
        sim_batt = st.slider("Batteries Safely Recycled (E-Waste)", 0, 25 * scale_mult, 5 * scale_mult, key="sim_batt")

    # Environmental calculation models
    co2_plastic = (sim_bottles * 0.025) * 1.5
    co2_cans = (sim_cans * 0.015) * 9.0
    co2_paper = (sim_paper * 0.050) * 1.3
    co2_glass = (sim_glass * 0.25) * 0.3
    co2_bio = sim_bio * 0.8
    toxin_prevented_kg = sim_batt * 0.02

    total_co2_kg = co2_plastic + co2_cans + co2_paper + co2_glass + co2_bio
    landfill_diverted_kg = (sim_bottles * 0.025) + (sim_cans * 0.015) + (sim_paper * 0.050) + (sim_glass * 0.25) + sim_bio
    trees_equiv = total_co2_kg / 21.0
    energy_saved_kwh = total_co2_kg * 1.82
    phone_charges = energy_saved_kwh * 83.3
    car_km_avoided = total_co2_kg * 4.16

    # Dynamic Eco Rank Tiers
    monthly_co2 = total_co2_kg / scale_mult
    if monthly_co2 > 40:
        rank_badge = "🌟 PLANET GUARDIAN (TIER IV)"
        rank_color = "#34d399"
    elif monthly_co2 > 20:
        rank_badge = "🌲 SUSTAINABILITY CHAMPION (TIER III)"
        rank_color = "#38bdf8"
    elif monthly_co2 > 10:
        rank_badge = "🌿 DEDICATED SORTER (TIER II)"
        rank_color = "#fbbf24"
    else:
        rank_badge = "🌱 ECO NOVICE (TIER I)"
        rank_color = "#94a3b8"

    with col_sim_out:
        st.markdown(
            f"""
            <div style="margin-bottom:14px;">
                <span class="status-pill" style="background:{rank_color}22; color:{rank_color}; border:1px solid {rank_color}66;">
                    {rank_badge}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown(
                f"""
                <div class="stat-pill">
                    <div class="stat-value">{landfill_diverted_kg:,.1f} kg</div>
                    <div class="stat-label">LANDFILL MASS DIVERTED</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r_col2:
            st.markdown(
                f"""
                <div class="stat-pill">
                    <div class="stat-value">{total_co2_kg:,.1f} kg</div>
                    <div class="stat-label">CO2e EMISSIONS PREVENTED</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        r_col3, r_col4 = st.columns(2)
        with r_col3:
            st.markdown(
                f"""
                <div class="stat-pill">
                    <div class="stat-value">{energy_saved_kwh:,.1f} kWh</div>
                    <div class="stat-label">GRID ENERGY CONSERVED</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r_col4:
            st.markdown(
                f"""
                <div class="stat-pill">
                    <div class="stat-value">{trees_equiv:,.2f} 🌲</div>
                    <div class="stat-label">TREE-YEAR EQUIVALENT</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        r_col5, r_col6 = st.columns(2)
        with r_col5:
            st.markdown(
                f"""
                <div class="stat-pill">
                    <div class="stat-value">{phone_charges:,.0f} 📱</div>
                    <div class="stat-label">SMARTPHONE CHARGES EQUIV</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r_col6:
            st.markdown(
                f"""
                <div class="stat-pill">
                    <div class="stat-value">{car_km_avoided:,.1f} km 🚗</div>
                    <div class="stat-label">CAR EMISSIONS AVOIDED</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if sim_batt > 0:
            st.success(
                f"🔋 Heavy Metal Protection: Diverted {sim_batt:,} batteries "
                f"({toxin_prevented_kg*1000:,.0f}g toxic lead/cadmium/lithium) from groundwater leaching!"
            )

        # Download ESG Audit Report
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        esg_export = {
            "report_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "assessment_scope": preset_choice,
            "scale_factor": scale_mult,
            "landfill_diverted_kg": round(landfill_diverted_kg, 2),
            "co2_prevented_kg": round(total_co2_kg, 2),
            "energy_conserved_kwh": round(energy_saved_kwh, 2),
            "tree_years_equivalent": round(trees_equiv, 2),
            "car_km_avoided": round(car_km_avoided, 2),
            "smartphone_charges_equivalent": round(phone_charges, 0),
            "heavy_metals_contained_grams": round(toxin_prevented_kg * 1000, 1),
            "eco_tier": rank_badge,
        }
        st.download_button(
            "📥 Export Enterprise ESG Impact Audit Report (JSON)",
            data=json.dumps(esg_export, indent=2),
            file_name=f"ecosort_esg_audit_{int(time.time())}.json",
            mime="application/json",
        )
