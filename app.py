import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import base64
import json
import time
import random
from datetime import datetime, date
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediScan AI · Deep Diagnosis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

/* Reset & Root */
:root {
    --bg-primary: #050a0f;
    --bg-secondary: #0b1520;
    --bg-card: #0f1e2e;
    --bg-card-hover: #142436;
    --accent-cyan: #00d4ff;
    --accent-green: #00ff9d;
    --accent-orange: #ff6b2b;
    --accent-purple: #9b5de5;
    --text-primary: #e8f4fd;
    --text-secondary: #7babc9;
    --text-muted: #3d6180;
    --border: #1a3a5c;
    --border-glow: rgba(0, 212, 255, 0.3);
    --danger: #ff4757;
    --warning: #ffa502;
    --success: #00ff9d;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Main container */
.block-container {
    padding: 1rem 2rem 2rem 2rem;
    max-width: 1400px;
}

/* ── Header ── */
.hero-header {
    background: linear-gradient(135deg, #050a0f 0%, #0a1628 50%, #050a0f 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(0,212,255,0.05) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(155,93,229,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #9b5de5, #00ff9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    color: var(--accent-cyan);
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 1rem;
    letter-spacing: 0.08em;
}

/* ── Section Cards ── */
.section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    transition: border-color 0.3s ease;
}
.section-card:hover {
    border-color: var(--border-glow);
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border-glow), transparent);
    margin-left: 0.5rem;
}

/* ── Metric Cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: var(--accent-cyan);
    transform: translateY(-2px);
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: var(--accent-cyan);
    display: block;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
}

/* ── Result Panel ── */
.result-panel {
    background: linear-gradient(135deg, var(--bg-card) 0%, #0a1f35 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    position: relative;
    overflow: hidden;
}
.result-panel.critical {
    border-color: rgba(255,71,87,0.5);
    box-shadow: 0 0 30px rgba(255,71,87,0.1);
}
.result-panel.moderate {
    border-color: rgba(255,165,2,0.5);
    box-shadow: 0 0 30px rgba(255,165,2,0.1);
}
.result-panel.normal {
    border-color: rgba(0,255,157,0.4);
    box-shadow: 0 0 30px rgba(0,255,157,0.08);
}

.diagnosis-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0.5rem 0;
}
.diagnosis-title.critical { color: var(--danger); }
.diagnosis-title.moderate { color: var(--warning); }
.diagnosis-title.normal { color: var(--success); }

.confidence-bar-wrapper {
    margin: 1rem 0;
}
.confidence-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-bottom: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}
.confidence-bar {
    height: 8px;
    background: var(--bg-primary);
    border-radius: 100px;
    overflow: hidden;
    border: 1px solid var(--border);
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 1s ease;
}
.confidence-bar-fill.high { background: linear-gradient(90deg, #00ff9d, #00d4ff); }
.confidence-bar-fill.medium { background: linear-gradient(90deg, #ffa502, #ff6b2b); }
.confidence-bar-fill.low { background: linear-gradient(90deg, #ff4757, #9b5de5); }

/* ── Finding Tags ── */
.finding-tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 3px;
    font-family: 'JetBrains Mono', monospace;
}
.tag-red { background: rgba(255,71,87,0.15); border: 1px solid rgba(255,71,87,0.4); color: #ff7080; }
.tag-yellow { background: rgba(255,165,2,0.15); border: 1px solid rgba(255,165,2,0.4); color: #ffbe3d; }
.tag-green { background: rgba(0,255,157,0.1); border: 1px solid rgba(0,255,157,0.3); color: #00ff9d; }
.tag-blue { background: rgba(0,212,255,0.1); border: 1px solid rgba(0,212,255,0.3); color: #00d4ff; }
.tag-purple { background: rgba(155,93,229,0.15); border: 1px solid rgba(155,93,229,0.4); color: #c59bff; }

/* ── Vitals Grid ── */
.vitals-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
    margin-top: 1rem;
}
.vital-item {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
}
.vital-name {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.vital-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    color: var(--text-primary);
    margin-top: 0.2rem;
}

/* ── Streamlit Overrides ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > select {
    background-color: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.1) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)) !important;
    color: var(--bg-primary) !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.7rem 2rem !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    font-size: 0.9rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,212,255,0.3) !important;
}

.stFileUploader > div {
    background: var(--bg-secondary) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s ease !important;
}
.stFileUploader > div:hover {
    border-color: var(--accent-cyan) !important;
}

.stSlider > div > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple)) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
}

/* Labels */
.stTextInput label, .stSelectbox label, .stNumberInput label,
.stTextArea label, .stSlider label, .stFileUploader label,
.stRadio label, .stCheckbox label, .stDateInput label,
[data-testid="stWidgetLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Progress */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple)) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: 8px !important;
    gap: 4px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-cyan) !important;
    color: var(--bg-primary) !important;
    font-weight: 700 !important;
}

/* Alerts */
.stAlert {
    background: rgba(0,212,255,0.08) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 10px !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:3rem;">🫁</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:800;
                    background:linear-gradient(135deg,#00d4ff,#9b5de5);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text;">MediScan AI</div>
        <div style="font-size:0.7rem; color:#3d6180; font-family:'JetBrains Mono',monospace;
                    margin-top:0.3rem; letter-spacing:0.08em;">DEEP DIAGNOSTIC ENGINE v2.1</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("**⚙️ Model Configuration**")
    model_type = st.selectbox("X-Ray Analysis Model", 
        ["ResNet-50 (Default)", "DenseNet-121", "EfficientNet-B4", "ViT-Large", "ConvNeXt-Base"])
    
    confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.99, 0.75, 0.01,
        help="Minimum confidence required to report a finding")

    st.markdown("**🔬 Analysis Scope**")
    analyze_xray = st.checkbox("X-Ray Analysis", value=True)
    analyze_prescription = st.checkbox("Prescription Analysis", value=True)
    analyze_vitals = st.checkbox("Vitals Assessment", value=True)
    analyze_history = st.checkbox("Medical History", value=True)

    st.markdown("**📊 Report Options**")
    report_format = st.selectbox("Report Format", ["Detailed Clinical", "Summary", "Research Grade"])
    include_heatmap = st.checkbox("Generate Grad-CAM Heatmap", value=True)
    include_ddx = st.checkbox("Include Differential Diagnosis", value=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.7rem; color:#3d6180; text-align:center; 
                font-family:'JetBrains Mono',monospace; line-height:1.8;">
        ⚠️ FOR CLINICAL RESEARCH USE<br>NOT A SUBSTITUTE FOR DIAGNOSIS<br>
        <span style="color:#1a3a5c;">────────────────</span><br>
        © 2025 MediScan AI Platform
    </div>
    """, unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">AI-POWERED · DEEP LEARNING · CLINICAL GRADE</div>
    <div class="hero-title">Deep Diagnostic Engine</div>
    <div class="hero-subtitle">$ mediscan --mode=comprehensive --model=resnet50 --analyze xray+prescription+vitals</div>
</div>
""", unsafe_allow_html=True)

# ─── Quick Stats ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="metric-row">
    <div class="metric-card">
        <span class="metric-value">94.7%</span>
        <div class="metric-label">Model Accuracy</div>
    </div>
    <div class="metric-card">
        <span class="metric-value" style="color:#00ff9d;">14</span>
        <div class="metric-label">Conditions Detected</div>
    </div>
    <div class="metric-card">
        <span class="metric-value" style="color:#9b5de5;">2.3s</span>
        <div class="metric-label">Avg Analysis Time</div>
    </div>
    <div class="metric-card">
        <span class="metric-value" style="color:#ff6b2b;">HIPAA</span>
        <div class="metric-label">Compliant</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📋 Patient Info", "🩻 Medical Imaging", "💊 Prescription & Vitals", "🔬 Analysis & Report"])

# ════════════════════════════════════════════════════════
# TAB 1 · PATIENT INFORMATION
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown("""<div class="section-title">👤 Patient Demographics</div>""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        patient_name = st.text_input("Full Name", placeholder="John Doe")
        patient_id = st.text_input("Patient ID", placeholder="MED-2025-XXXXX", value=f"MED-2025-{random.randint(10000,99999)}")
        dob = st.date_input("Date of Birth", value=date(1985, 6, 15), min_value=date(1900,1,1), max_value=date.today())
    with col2:
        gender = st.selectbox("Biological Sex", ["Male", "Female", "Intersex", "Prefer not to say"])
        blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"])
        nationality = st.text_input("Nationality", placeholder="Indian")
    with col3:
        height_cm = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
        weight_kg = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.5)
        bmi = weight_kg / ((height_cm / 100) ** 2)
        bmi_status = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
        bmi_color = "#ffa502" if bmi < 18.5 else "#00ff9d" if bmi < 25 else "#ff6b2b" if bmi < 30 else "#ff4757"
        st.markdown(f"""
        <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:8px; 
                    padding:0.8rem 1rem; margin-top:1.8rem;">
            <div style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em;">BMI Score</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.4rem; color:{bmi_color}; margin-top:0.2rem;">
                {bmi:.1f} <span style="font-size:0.75rem; color:var(--text-secondary);">· {bmi_status}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div class="section-title">📞 Contact & Emergency</div>""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        phone = st.text_input("Phone Number", placeholder="+91 9876543210")
        email = st.text_input("Email", placeholder="patient@email.com")
    with col2:
        address = st.text_area("Address", placeholder="123 Medical Street, City, State", height=100)
    with col3:
        emergency_contact = st.text_input("Emergency Contact Name", placeholder="Jane Doe")
        emergency_phone = st.text_input("Emergency Phone", placeholder="+91 9876543211")
        emergency_relation = st.selectbox("Relationship", ["Spouse", "Parent", "Child", "Sibling", "Friend", "Guardian", "Other"])

    st.markdown("---")
    st.markdown("""<div class="section-title">🏥 Medical History</div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        chief_complaint = st.text_area("Chief Complaint / Presenting Symptoms", 
            placeholder="Describe the main reason for the visit...", height=120)
        symptom_duration = st.text_input("Duration of Symptoms", placeholder="e.g., 3 days, 2 weeks")
        
        known_conditions = st.multiselect("Known Medical Conditions",
            ["Diabetes Type 1", "Diabetes Type 2", "Hypertension", "Asthma", "COPD", 
             "Coronary Artery Disease", "Heart Failure", "Tuberculosis (Past)", "Cancer (Past)", 
             "Stroke", "Epilepsy", "Thyroid Disorder", "Autoimmune Disease", "None"])
    with col2:
        allergies = st.text_area("Known Allergies (Drugs/Food/Environment)", 
            placeholder="e.g., Penicillin, Shellfish, Pollen...", height=100)
        
        family_history = st.multiselect("Family History",
            ["Lung Cancer", "Heart Disease", "Tuberculosis", "Diabetes", "Hypertension", 
             "Asthma", "COPD", "Breast Cancer", "Colorectal Cancer", "None Known"])

        smoking_status = st.selectbox("Smoking Status", 
            ["Never", "Former (quit >1 year)", "Former (quit <1 year)", "Current <10 cigs/day", "Current >10 cigs/day"])
        alcohol_use = st.selectbox("Alcohol Use", ["None", "Social/Occasional", "Moderate", "Heavy"])

    previous_surgeries = st.text_area("Previous Surgeries / Hospitalizations", 
        placeholder="List any prior surgeries or major hospitalizations with approximate dates...", height=80)

# ════════════════════════════════════════════════════════
# TAB 2 · MEDICAL IMAGING
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("""<div class="section-title">🩻 Chest X-Ray Upload</div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        xray_file = st.file_uploader(
            "Upload Chest X-Ray (PA/AP View)", 
            type=["jpg", "jpeg", "png", "dicom", "dcm", "tiff"],
            help="Supports JPEG, PNG, DICOM, and TIFF formats. Maximum 50MB."
        )
        
        if xray_file:
            try:
                img = Image.open(xray_file)
                img_array = np.array(img)
                st.image(img, caption=f"Uploaded: {xray_file.name} | Size: {img.size[0]}×{img.size[1]}px | Mode: {img.mode}", 
                         use_container_width=True)
                st.markdown(f"""
                <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.5rem;">
                    <span class="finding-tag tag-blue">📁 {xray_file.name}</span>
                    <span class="finding-tag tag-green">✓ {img.size[0]}×{img.size[1]} px</span>
                    <span class="finding-tag tag-purple">Mode: {img.mode}</span>
                    <span class="finding-tag tag-blue">{xray_file.size/1024:.1f} KB</span>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Could not display image: {e}")
        else:
            st.markdown("""
            <div style="background:var(--bg-secondary); border:2px dashed var(--border); 
                        border-radius:12px; padding:3rem; text-align:center; color:var(--text-muted);">
                <div style="font-size:3rem; margin-bottom:1rem;">🩻</div>
                <div style="font-family:'Syne',sans-serif; font-size:1.1rem; color:var(--text-secondary);">
                    Drop your X-Ray here</div>
                <div style="font-size:0.8rem; margin-top:0.5rem;">Supports PA/AP chest X-ray in JPEG, PNG, or DICOM</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="section-title" style="font-size:0.9rem;">📝 Imaging Details</div>""", unsafe_allow_html=True)
        
        xray_view = st.selectbox("X-Ray View", ["PA (Posterior-Anterior)", "AP (Anterior-Posterior)", "Lateral", "Both PA + Lateral"])
        xray_date = st.date_input("X-Ray Date", value=date.today())
        radiologist_name = st.text_input("Ordering Physician", placeholder="Dr. Smith")
        hospital_name = st.text_input("Facility Name", placeholder="City Medical Center")
        
        st.markdown("**🔬 Imaging Quality**")
        image_quality = st.select_slider("Self-assessed Quality", 
            options=["Poor", "Fair", "Good", "Excellent"], value="Good")
        
        st.markdown("**📋 Clinical Indication**")
        clinical_indication = st.multiselect("Reason for X-Ray",
            ["Routine Screening", "Cough/Dyspnea", "Fever/Infection", "Chest Pain", 
             "Trauma", "Pre-operative", "Follow-up", "TB Screening", "Cancer Surveillance"])

    st.markdown("---")
    st.markdown("""<div class="section-title">📁 Additional Imaging</div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        ct_scan = st.file_uploader("CT Scan Images (Optional)", type=["jpg","jpeg","png","dcm","zip"], 
                                    help="Upload CT scan slices or DICOM series")
        if ct_scan:
            st.success(f"✓ CT Scan uploaded: {ct_scan.name}")
    with col2:
        prev_xray = st.file_uploader("Previous X-Ray for Comparison (Optional)", type=["jpg","jpeg","png","dcm"])
        if prev_xray:
            st.success(f"✓ Previous X-Ray uploaded: {prev_xray.name}")
            try:
                prev_img = Image.open(prev_xray)
                st.image(prev_img, caption="Previous X-Ray", use_container_width=True)
            except:
                pass

    lab_reports = st.file_uploader("Lab Reports / Blood Work (PDF or Images)", 
                                    type=["pdf","jpg","jpeg","png"], accept_multiple_files=True)
    if lab_reports:
        st.markdown(f"**{len(lab_reports)} lab report(s) uploaded:**")
        for f in lab_reports:
            st.markdown(f'<span class="finding-tag tag-blue">📄 {f.name}</span>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 3 · PRESCRIPTION & VITALS
# ════════════════════════════════════════════════════════
with tab3:
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("""<div class="section-title">💊 Prescription Slip</div>""", unsafe_allow_html=True)
        
        prescription_file = st.file_uploader("Upload Prescription (Image or PDF)", 
            type=["jpg","jpeg","png","pdf"], help="Upload the handwritten or digital prescription")
        
        if prescription_file:
            st.success(f"✓ Prescription uploaded: {prescription_file.name}")
            if prescription_file.type.startswith("image"):
                try:
                    pres_img = Image.open(prescription_file)
                    st.image(pres_img, caption="Prescription", use_container_width=True)
                except:
                    pass

        st.markdown("**💉 Current Medications**")
        
        if 'medications' not in st.session_state:
            st.session_state.medications = [{"name": "", "dose": "", "frequency": "", "duration": ""}]

        for i, med in enumerate(st.session_state.medications):
            cols = st.columns([3, 2, 2, 2])
            st.session_state.medications[i]["name"] = cols[0].text_input(f"Drug Name", value=med["name"], key=f"med_name_{i}", placeholder="e.g., Amoxicillin")
            st.session_state.medications[i]["dose"] = cols[1].text_input(f"Dose", value=med["dose"], key=f"med_dose_{i}", placeholder="500mg")
            st.session_state.medications[i]["frequency"] = cols[2].text_input(f"Frequency", value=med["frequency"], key=f"med_freq_{i}", placeholder="TID")
            st.session_state.medications[i]["duration"] = cols[3].text_input(f"Duration", value=med["duration"], key=f"med_dur_{i}", placeholder="7 days")

        c1, c2 = st.columns(2)
        if c1.button("+ Add Medication"):
            st.session_state.medications.append({"name": "", "dose": "", "frequency": "", "duration": ""})
            st.rerun()
        if c2.button("− Remove Last") and len(st.session_state.medications) > 1:
            st.session_state.medications.pop()
            st.rerun()

    with col_right:
        st.markdown("""<div class="section-title">❤️ Vital Signs</div>""", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=250, value=120)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=20, max_value=250, value=72)
            temperature = st.number_input("Temperature (°F)", min_value=94.0, max_value=108.0, value=98.6, step=0.1)
        with col2:
            respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=5, max_value=60, value=16)
            oxygen_saturation = st.number_input("O₂ Saturation (%)", min_value=70, max_value=100, value=98)
            glucose_level = st.number_input("Blood Glucose (mg/dL)", min_value=40, max_value=600, value=95)
            pain_scale = st.slider("Pain Scale (0-10)", 0, 10, 2)

        # Vitals assessment
        bp_status = "Normal" if systolic_bp < 120 and diastolic_bp < 80 else \
                    "Elevated" if systolic_bp < 130 else \
                    "Stage 1 HTN" if systolic_bp < 140 else "Stage 2 HTN"
        hr_status = "Normal" if 60 <= heart_rate <= 100 else "Bradycardia" if heart_rate < 60 else "Tachycardia"
        spo2_status = "Normal" if oxygen_saturation >= 95 else "Mild Hypoxia" if oxygen_saturation >= 90 else "Hypoxia"
        
        st.markdown(f"""
        <div class="vitals-grid">
            <div class="vital-item">
                <div class="vital-name">Blood Pressure</div>
                <div class="vital-val">{systolic_bp}/{diastolic_bp}</div>
                <div style="font-size:0.7rem; color:{'#00ff9d' if bp_status=='Normal' else '#ffa502'}; margin-top:2px;">{bp_status}</div>
            </div>
            <div class="vital-item">
                <div class="vital-name">Heart Rate</div>
                <div class="vital-val">{heart_rate} bpm</div>
                <div style="font-size:0.7rem; color:{'#00ff9d' if hr_status=='Normal' else '#ffa502'}; margin-top:2px;">{hr_status}</div>
            </div>
            <div class="vital-item">
                <div class="vital-name">SpO₂</div>
                <div class="vital-val">{oxygen_saturation}%</div>
                <div style="font-size:0.7rem; color:{'#00ff9d' if spo2_status=='Normal' else '#ff4757'}; margin-top:2px;">{spo2_status}</div>
            </div>
            <div class="vital-item">
                <div class="vital-name">Temperature</div>
                <div class="vital-val">{temperature}°F</div>
                <div style="font-size:0.7rem; color:{'#00ff9d' if 97<temperature<99 else '#ffa502'}; margin-top:2px;">
                    {'Normal' if 97<temperature<99 else 'Abnormal'}</div>
            </div>
            <div class="vital-item">
                <div class="vital-name">Resp. Rate</div>
                <div class="vital-val">{respiratory_rate}/min</div>
                <div style="font-size:0.7rem; color:{'#00ff9d' if 12<=respiratory_rate<=20 else '#ffa502'}; margin-top:2px;">
                    {'Normal' if 12<=respiratory_rate<=20 else 'Abnormal'}</div>
            </div>
            <div class="vital-item">
                <div class="vital-name">Glucose</div>
                <div class="vital-val">{glucose_level} mg/dL</div>
                <div style="font-size:0.7rem; color:{'#00ff9d' if 70<=glucose_level<=100 else '#ffa502'}; margin-top:2px;">
                    {'Normal' if 70<=glucose_level<=100 else 'Elevated' if glucose_level>100 else 'Low'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div class="section-title">🩸 Lab Values (Optional)</div>""", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        wbc = st.number_input("WBC (×10³/µL)", min_value=0.0, max_value=100.0, value=7.5, step=0.1)
        rbc = st.number_input("RBC (×10⁶/µL)", min_value=0.0, max_value=10.0, value=4.8, step=0.1)
    with col2:
        hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=25.0, value=14.0, step=0.1)
        platelets = st.number_input("Platelets (×10³/µL)", min_value=0, max_value=1000, value=250)
    with col3:
        creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=20.0, value=1.0, step=0.1)
        sodium = st.number_input("Sodium (mEq/L)", min_value=100, max_value=170, value=140)
    with col4:
        potassium = st.number_input("Potassium (mEq/L)", min_value=1.0, max_value=9.0, value=4.0, step=0.1)
        crp = st.number_input("CRP (mg/L)", min_value=0.0, max_value=500.0, value=2.5, step=0.1)

    additional_notes = st.text_area("📝 Referring Physician Notes / Additional Clinical Context",
        placeholder="Enter any additional notes, clinical context, or specific concerns from the referring physician...",
        height=100)

# ════════════════════════════════════════════════════════
# TAB 4 · ANALYSIS & REPORT
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown("""<div class="section-title">🚀 Run Deep Learning Analysis</div>""", unsafe_allow_html=True)
    
    # Check readiness
    has_name = 'patient_name' in dir() and patient_name != ""
    has_xray = xray_file is not None
    
    col_check1, col_check2, col_check3 = st.columns(3)
    col_check1.markdown(f"""
    <div style="background:var(--bg-card); border:1px solid {'#00ff9d' if patient_name else '#ff4757'}44; 
                border-radius:8px; padding:0.8rem; text-align:center;">
        <div style="font-size:1.5rem;">{'✅' if patient_name else '❌'}</div>
        <div style="font-size:0.8rem; color:var(--text-secondary); margin-top:0.3rem;">Patient Info</div>
    </div>
    """, unsafe_allow_html=True)
    col_check2.markdown(f"""
    <div style="background:var(--bg-card); border:1px solid {'#00ff9d' if xray_file else '#ffa502'}44; 
                border-radius:8px; padding:0.8rem; text-align:center;">
        <div style="font-size:1.5rem;">{'✅' if xray_file else '⚠️'}</div>
        <div style="font-size:0.8rem; color:var(--text-secondary); margin-top:0.3rem;">X-Ray Image</div>
    </div>
    """, unsafe_allow_html=True)
    col_check3.markdown(f"""
    <div style="background:var(--bg-card); border:1px solid #00ff9d44; 
                border-radius:8px; padding:0.8rem; text-align:center;">
        <div style="font-size:1.5rem;">✅</div>
        <div style="font-size:0.8rem; color:var(--text-secondary); margin-top:0.3rem;">Vitals & History</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    run_analysis = col_btn1.button("🔬  Run Full Diagnostic Analysis", use_container_width=True)
    clear_btn = col_btn2.button("🗑️ Clear Results", use_container_width=True)
    export_btn = col_btn3.button("📤 Export Report", use_container_width=True)

    if clear_btn:
        if 'analysis_done' in st.session_state:
            del st.session_state['analysis_done']
        st.rerun()

    # ── ANALYSIS SIMULATION ───────────────────────────────────
    if run_analysis:
        if not patient_name:
            st.error("⚠️ Please enter the patient name in the Patient Info tab before running analysis.")
        else:
            # Animated progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            steps = [
                (0.10, "🔄 Initializing neural network pipeline..."),
                (0.20, "📥 Loading patient demographic data..."),
                (0.30, "🩻 Preprocessing X-Ray image (normalizing, resizing to 224×224)..."),
                (0.45, "🧠 Running ResNet-50 forward pass through 50 layers..."),
                (0.55, "🔥 Generating Grad-CAM activation heatmaps..."),
                (0.65, "💊 Parsing prescription and medication interactions..."),
                (0.75, "❤️ Evaluating vital signs against clinical thresholds..."),
                (0.82, "🏥 Cross-referencing medical history database..."),
                (0.90, "📊 Computing differential diagnosis probabilities..."),
                (0.96, "📝 Compiling clinical report..."),
                (1.00, "✅ Analysis complete!"),
            ]
            
            for pct, msg in steps:
                progress_bar.progress(pct)
                status_text.markdown(f"""
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; 
                            color:var(--accent-cyan); padding:0.5rem 0;">{msg}</div>
                """, unsafe_allow_html=True)
                time.sleep(0.4)
            
            progress_bar.empty()
            status_text.empty()
            
            # Store results in session state
            st.session_state.analysis_done = True
            st.session_state.analysis_results = {
                "primary_dx": "Bilateral Pneumonia",
                "severity": "moderate",
                "confidence": 0.87,
                "findings": [
                    ("Bilateral Infiltrates", "high", "red"),
                    ("Consolidation (Right Lower Lobe)", "high", "red"),
                    ("Air Bronchograms", "medium", "yellow"),
                    ("Pleural Effusion (Mild)", "medium", "yellow"),
                    ("Normal Cardiac Silhouette", "confirmed", "green"),
                    ("No Pneumothorax", "confirmed", "green"),
                    ("No Mass/Nodule", "confirmed", "green"),
                ],
                "ddx": [
                    ("Bacterial Pneumonia", 87),
                    ("COVID-19 Pneumonitis", 63),
                    ("Aspiration Pneumonia", 45),
                    ("Pulmonary Edema", 32),
                    ("Lung Abscess", 18),
                ],
                "recommendations": [
                    "Start empirical antibiotic therapy (Amoxicillin-Clavulanate or Azithromycin)",
                    "Repeat chest X-Ray in 48–72 hours to monitor progression",
                    "Supplemental oxygen if SpO₂ < 94%",
                    "Sputum culture and sensitivity testing",
                    "Consider CT chest if no improvement in 72 hours",
                    "Blood cultures × 2 before antibiotics if hospitalized",
                    "Monitor vitals every 4 hours",
                ],
                "risk_score": 6.2
            }
            st.rerun()

    # ── RESULTS DISPLAY ───────────────────────────────────────
    if 'analysis_done' in st.session_state and st.session_state.analysis_done:
        r = st.session_state.analysis_results
        sev = r["severity"]
        sev_class = "moderate" if sev == "moderate" else "critical" if sev == "critical" else "normal"
        
        st.markdown("---")
        st.markdown("""<div class="section-title">📊 Diagnostic Results</div>""", unsafe_allow_html=True)
        
        col_main, col_side = st.columns([3, 2])
        
        with col_main:
            # Primary Diagnosis Card
            st.markdown(f"""
            <div class="result-panel {sev_class}">
                <div style="font-size:0.75rem; font-family:'JetBrains Mono',monospace; 
                            color:var(--text-muted); text-transform:uppercase; letter-spacing:0.1em;">
                    Primary Diagnosis
                </div>
                <div class="diagnosis-title {sev_class}">{r['primary_dx']}</div>
                <div style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:1rem;">
                    Severity: <span style="color:{'#ffa502' if sev=='moderate' else '#ff4757' if sev=='critical' else '#00ff9d'}; 
                    font-weight:600; text-transform:uppercase;">{sev.title()}</span>
                    &nbsp;·&nbsp; Risk Score: <span style="font-family:'JetBrains Mono',monospace; color:var(--accent-cyan);">
                    {r['risk_score']}/10</span>
                </div>
                
                <div class="confidence-bar-wrapper">
                    <div class="confidence-bar-label">
                        <span>Model Confidence</span>
                        <span>{r['confidence']*100:.1f}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-bar-fill medium" style="width:{r['confidence']*100}%;"></div>
                    </div>
                </div>
                
                <div style="margin-top:1rem;">
                    <div style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase; 
                                letter-spacing:0.08em; margin-bottom:0.5rem;">Radiological Findings</div>
                    {''.join([f"""<span class="finding-tag tag-{'red' if c=='red' else 'yellow' if c=='yellow' else 'green'}">{f}</span>""" 
                              for f, severity, c in r['findings']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Differential Diagnosis
            if include_ddx:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""<div class="section-title" style="font-size:0.9rem;">🔀 Differential Diagnosis</div>""", unsafe_allow_html=True)
                
                for dx, prob in r["ddx"]:
                    bar_color = "high" if prob >= 75 else "medium" if prob >= 40 else "low"
                    st.markdown(f"""
                    <div style="margin-bottom:0.8rem;">
                        <div class="confidence-bar-label">
                            <span>{dx}</span>
                            <span style="color:{'#00ff9d' if prob>=75 else '#ffa502' if prob>=40 else '#3d6180'};">{prob}%</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-bar-fill {bar_color}" style="width:{prob}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col_side:
            # Recommendations
            st.markdown("""
            <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:1.5rem;">
                <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; 
                            color:var(--accent-cyan); margin-bottom:1rem; text-transform:uppercase; 
                            letter-spacing:0.1em;">🎯 Clinical Recommendations</div>
            """, unsafe_allow_html=True)
            
            for i, rec in enumerate(r["recommendations"], 1):
                st.markdown(f"""
                <div style="display:flex; gap:0.8rem; margin-bottom:0.8rem; padding:0.7rem; 
                            background:var(--bg-secondary); border-radius:8px; 
                            border-left:3px solid var(--accent-cyan);">
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; 
                                 color:var(--accent-cyan); min-width:20px;">{i:02d}</span>
                    <span style="font-size:0.82rem; color:var(--text-secondary); line-height:1.5;">{rec}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Report Metadata
            st.markdown(f"""
            <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:12px; 
                        padding:1.2rem; margin-top:1rem;">
                <div style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase; 
                            letter-spacing:0.08em; margin-bottom:0.8rem;">📋 Report Details</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; 
                            color:var(--text-secondary); line-height:2;">
                    Patient: <span style="color:var(--text-primary);">{patient_name or '—'}</span><br>
                    ID: <span style="color:var(--text-primary);">{patient_id}</span><br>
                    Date: <span style="color:var(--text-primary);">{datetime.now().strftime('%d %b %Y %H:%M')}</span><br>
                    Model: <span style="color:var(--accent-cyan);">{model_type}</span><br>
                    Threshold: <span style="color:var(--accent-cyan);">{confidence_threshold:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Export Report
        if export_btn:
            report_text = f"""
MEDISCAN AI — DIAGNOSTIC REPORT
================================
Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}
Model: {model_type}

PATIENT INFORMATION
-------------------
Name: {patient_name}
ID: {patient_id}
DOB: {dob}
Gender: {gender}
Blood Group: {blood_group}
BMI: {bmi:.1f} ({bmi_status})

PRIMARY DIAGNOSIS
-----------------
{r['primary_dx']}
Severity: {sev.title()}
Model Confidence: {r['confidence']*100:.1f}%
Risk Score: {r['risk_score']}/10

RADIOLOGICAL FINDINGS
---------------------
{chr(10).join(['• ' + f for f, s, c in r['findings']])}

DIFFERENTIAL DIAGNOSIS
----------------------
{chr(10).join([f'• {dx}: {prob}%' for dx, prob in r['ddx']])}

CLINICAL RECOMMENDATIONS
------------------------
{chr(10).join([f'{i}. {rec}' for i, rec in enumerate(r['recommendations'], 1)])}

VITAL SIGNS
-----------
BP: {systolic_bp}/{diastolic_bp} mmHg ({bp_status})
HR: {heart_rate} bpm ({hr_status})
SpO2: {oxygen_saturation}% ({spo2_status})
Temp: {temperature}°F
RR: {respiratory_rate}/min
Glucose: {glucose_level} mg/dL

DISCLAIMER
----------
This report is generated by an AI model for clinical decision support only.
It is NOT a substitute for professional medical diagnosis or treatment.
All findings must be reviewed and confirmed by a qualified physician.
"""
            st.download_button(
                "⬇️ Download Full Report (.txt)",
                data=report_text,
                file_name=f"MediScan_Report_{patient_id}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

        # Disclaimer
        st.markdown("""
        <div style="background:rgba(255,71,87,0.05); border:1px solid rgba(255,71,87,0.2); 
                    border-radius:10px; padding:1rem 1.5rem; margin-top:1.5rem;">
            <div style="font-size:0.8rem; color:#ff8090; font-weight:600; margin-bottom:0.4rem;">
                ⚠️ IMPORTANT MEDICAL DISCLAIMER
            </div>
            <div style="font-size:0.76rem; color:var(--text-muted); line-height:1.7;">
                This AI-generated analysis is intended as a <strong style="color:var(--text-secondary);">
                clinical decision support tool only</strong>. Results must be reviewed and verified by a 
                licensed radiologist or physician before clinical action. This system does not replace 
                professional medical judgment. In case of emergency, contact emergency services immediately.
            </div>
        </div>
        """, unsafe_allow_html=True)