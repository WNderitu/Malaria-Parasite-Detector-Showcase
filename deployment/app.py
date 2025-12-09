import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
import io
import altair as alt
import onnxruntime as ort
from datetime import datetime
import time
import zipfile
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Malaria Parasite (P.vivax) Detector v2",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Malaria Parasite Detection System v2.0 - Powered by YOLOv8n"
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .legend-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# --- File Paths ---
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'best.onnx')
classes_path = os.path.join(base_path, 'classes.txt')

# --- Validate Files ---
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-header">🔬 Malaria Parasite (P.vivax) Detector v2</div>', unsafe_allow_html=True)
with col2:
    if os.path.exists(model_path):
        st.success("✅ Model Ready")
    else:
        st.error("❌ Model Missing")

# --- Load Model & Classes ---
@st.cache_resource
def load_onnx_model(model_path):
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        return session
    except Exception as e:
        st.error(f"Error loading ONNX model: {e}")
        return None

@st.cache_data
def load_class_names(classes_path):
    try:
        with open(classes_path, "r") as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        return []

session = load_onnx_model(model_path)
class_names = load_class_names(classes_path)

# --- Color Legend in Sidebar ---
def display_color_legend(color_scheme="Default"):
    DEFAULT_COLOR_MAP = {
        'Red Blood Cell': (255, 0, 0),
        'Leukocyte': (255, 255, 0),
        'Schizont': (0, 255, 255),
        'Ring': (0, 255, 0),
        'Gametocyte': (255, 0, 255),
        'Trophozoite': (255, 165, 0),
    }
    HIGH_CONTRAST_MAP = {
        'Red Blood Cell': (255, 0, 0),
        'Leukocyte': (255, 255, 0),
        'Schizont': (0, 255, 255),
        'Ring': (0, 255, 0),
        'Gametocyte': (255, 0, 255),
        'Trophozoite': (255, 128, 0),
    }
    PASTEL_MAP = {
        'Red Blood Cell': (255, 182, 193),
        'Leukocyte': (255, 250, 205),
        'Schizont': (175, 238, 238),
        'Ring': (144, 238, 144),
        'Gametocyte': (221, 160, 221),
        'Trophozoite': (255, 218, 185),
    }
    
    if color_scheme == "High Contrast":
        COLOR_MAP = HIGH_CONTRAST_MAP
    elif color_scheme == "Pastel":
        COLOR_MAP = PASTEL_MAP
    else:
        COLOR_MAP = DEFAULT_COLOR_MAP
    
    st.sidebar.markdown("### 🎨 Color Legend")
    for class_name, color in COLOR_MAP.items():
        # Convert BGR to RGB for display
        r, g, b = color
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        st.sidebar.markdown(
            f'<div style="display: flex; align-items: center; margin: 5px 0;">'
            f'<div style="width: 20px; height: 20px; background-color: {hex_color}; '
            f'border: 2px solid #333; margin-right: 10px; border-radius: 3px;"></div>'
            f'<span>{class_name}</span></div>',
            unsafe_allow_html=True
        )

# --- Information Expandable Sections ---
with st.expander("⚠️ Model Performance Information", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Detection Accuracy")
        st.markdown("""
        | Class | Precision | Recall |
        |-------|-----------|--------|
        | **RBC** | 90.0% | 95.2% |
        | **Trophozoite** | 57.0% | 64.7% |
        | **Ring** | 46.0% | 13.6% |
        | **Schizont** | 18.1% | 9.1% |
        | **Gametocyte** | 0.0% | 0.0% |
        """)
    with col2:
        st.markdown("### ⚠️ Important Notes")
        st.warning("""
        - **Best Performance**: RBC & Trophozoite detection
        - **Moderate**: Ring stage detection
        - **Poor**: Schizont & Gametocyte detection
        - **Recommendation**: Manual verification required for rare parasite stages
        - **Use Case**: Screening tool, NOT diagnostic confirmation
        """)

with st.expander("📖 WHO Parasitemia Classification Guide", expanded=False):
    st.markdown("""
    ### Parasitemia Calculation
    **Parasitemia (%) = (Total Parasites Detected / Total Cells Detected) × 100**
    
    ---
    
    ### Classification Thresholds
    
    | Category | Parasitemia Range | Clinical Significance | Action Required |
    |----------|-------------------|----------------------|-----------------|
    | **Negative** | 0% | No parasites detected | No immediate action |
    | **Minimal** | < 0.1% | Very low parasitemia | Monitor, confirm with microscopy |
    | **Mild** | 0.1% - 2% | Low-level infection | Outpatient treatment |
    | **Moderate** | 2% - 5% | Moderate infection | Close monitoring, treatment |
    | **Severe** | > 5% | High parasite burden | **Immediate medical attention required** |
    
    ---
    
    ### Clinical Notes
    
    #### Severe Malaria Indicators (WHO):
    - Parasitemia > 5% in non-immune patients
    - Parasitemia > 10% regardless of immune status
    - Presence of schizonts in peripheral blood
    - Multiple parasite life stages present
    
    #### P. vivax Specific Considerations:
    - P. vivax typically has lower parasitemia than P. falciparum
    - Gametocytes indicate transmission potential
    - Relapses common due to hypnozoites (dormant liver stage)
    - Lower parasitemia can still cause severe symptoms
    
    ---
    
    ### Parasite Life Stages
    
    - **Ring Stage**: Early infection, most common form seen
    - **Trophozoite**: Active feeding stage, larger than rings
    - **Schizont**: Mature stage before red cell rupture, indicates active multiplication
    - **Gametocyte**: Sexual stage, indicates transmission potential to mosquitoes
    
    ---
    
    **⚠️ Important Disclaimer:**
    
    This tool is designed for **research and screening purposes only**. Always consult healthcare professionals for medical decisions.
    """)

with st.expander("📊 Model Evaluation Metrics (Test Set)", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall mAP50", "0.400", help="Mean Average Precision at IoU=0.50")
        st.metric("Mean Precision", "0.422", help="Average precision across all classes")
    with col2:
        st.metric("Overall mAP50-95", "0.301", help="Mean Average Precision at IoU=0.50:0.95")
        st.metric("Mean Recall", "0.365", help="Average recall across all classes")
    with col3:
        st.metric("F1-Score", "0.391", help="Harmonic mean of precision and recall")
        st.metric("Test Images", "120", help="Number of images in test set")

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Detection Settings")

# Detection settings - Constrained ranges
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    0.20, 0.40, 0.25, 0.01,
    help="Constrained to 0.20-0.40. Lower values detect more rare parasites but may increase false positives"
)
nms_threshold = st.sidebar.slider(
    "NMS IoU Threshold", 
    0.60, 0.70, 0.65, 0.01,
    help="Constrained to 0.60-0.70. Higher values allow more overlapping detections"
)

st.sidebar.markdown("---")
st.sidebar.header("🎨 Visualization Settings")

show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
show_labels = st.sidebar.checkbox("Show Class Labels", value=True)
show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)
show_only_parasites = st.sidebar.checkbox("Show Only Parasite Detections", value=False)
color_scheme = st.sidebar.selectbox("Color Scheme", ["Default", "High Contrast", "Pastel"], index=0)

# Display color legend
display_color_legend(color_scheme)

# Add class-specific confidence thresholds
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Advanced Settings")
use_class_thresholds = st.sidebar.checkbox("Enable Per-Class Thresholds", value=False)

class_thresholds = {}
if use_class_thresholds:
    st.sidebar.info("Per-class thresholds constrained to 0.20-0.40 range")
    class_thresholds = {
        'schizont': st.sidebar.slider("Schizont Threshold", 0.20, 0.40, 0.20, 0.01),
        'gametocyte': st.sidebar.slider("Gametocyte Threshold", 0.20, 0.40, 0.20, 0.01),
        'ring': st.sidebar.slider("Ring Threshold", 0.20, 0.40, 0.22, 0.01),
        'trophozoite': st.sidebar.slider("Trophozoite Threshold", 0.20, 0.40, 0.25, 0.01),
        'red blood cell': st.sidebar.slider("RBC Threshold", 0.20, 0.40, 0.35, 0.01),
        'leukocyte': st.sidebar.slider("Leukocyte Threshold", 0.20, 0.40, 0.30, 0.01),
    }

# --- Enhanced Image Processing Function ---
def process_image(session, image, conf_threshold, nms_threshold, class_names,
                  show_boxes=True, show_labels=True, show_confidence=True,
                  show_only_parasites=False, color_scheme="Default", class_thresholds=None):
    INPUT_WIDTH, INPUT_HEIGHT = 1280, 1280
    img_cv = np.array(image.convert("RGB"))
    original_height, original_width = img_cv.shape[:2]
    
    # Check image dimensions
    dim_warning = None
    if original_width < 640 or original_height < 640:
        dim_warning = "⚠️ Image resolution is low. For best results, use images ≥ 1280x1280 pixels."
    
    blob = cv2.dnn.blobFromImage(img_cv, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)

    # ONNXRuntime inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    preds = session.run([output_name], {input_name: blob})[0]
    detections = preds[0].T

    boxes, confidences, class_ids = [], [], []
    parasite_IDs = {2, 3, 4, 5}  # schizont, ring, gametocyte, trophozoite

    # Enhanced Color maps with better contrast
    DEFAULT_COLOR_MAP = {
        'red blood cell': (0, 0, 255),      # Red
        'leukocyte': (255, 255, 0),         # Bright Yellow
        'schizont': (0, 255, 255),          # Cyan
        'ring': (0, 255, 0),                # Green
        'gametocyte': (255, 0, 255),        # Magenta
        'trophozoite': (255, 165, 0),       # Orange
        'default': (128, 128, 128)
    }
    HIGH_CONTRAST_MAP = {
        'red blood cell': (255, 0, 0),
        'leukocyte': (255, 255, 0),
        'schizont': (0, 255, 255),
        'ring': (0, 255, 0),
        'gametocyte': (255, 0, 255),
        'trophozoite': (255, 128, 0),
        'default': (255, 255, 255)
    }
    PASTEL_MAP = {
        'red blood cell': (255, 182, 193),
        'leukocyte': (255, 250, 205),
        'schizont': (175, 238, 238),
        'ring': (144, 238, 144),
        'gametocyte': (221, 160, 221),
        'trophozoite': (255, 218, 185),
        'default': (211, 211, 211)
    }

    if color_scheme == "High Contrast":
        COLOR_MAP = HIGH_CONTRAST_MAP
    elif color_scheme == "Pastel":
        COLOR_MAP = PASTEL_MAP
    else:
        COLOR_MAP = DEFAULT_COLOR_MAP

    class_counts = {name: 0 for name in class_names}
    class_confidences = {name: [] for name in class_names}

    for row in detections:
        confidence = row[4]
        classes_scores = row[5:]
        class_id = np.argmax(classes_scores)
        
        if class_id >= len(class_names):
            continue
            
        detected_class_name = class_names[class_id]
        
        # Apply class-specific threshold if enabled
        if class_thresholds and detected_class_name in class_thresholds:
            threshold = class_thresholds[detected_class_name]
        else:
            threshold = conf_threshold
        
        if confidence > threshold and classes_scores[class_id] > 0.0:
            x_scale = original_width / INPUT_WIDTH
            y_scale = original_height / INPUT_HEIGHT
            center_x, center_y, width, height = row[0]*x_scale, row[1]*y_scale, row[2]*x_scale, row[3]*y_scale
            x, y, w, h = int(center_x - width/2), int(center_y - height/2), int(width), int(height)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    if not boxes:
        return img_cv, class_counts, class_confidences, dim_warning

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
    else:
        return img_cv, class_counts, class_confidences, dim_warning

    if show_only_parasites:
        indices = [i for i in indices if class_ids[i] in parasite_IDs]

    for i in indices:
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        detected_class_name = class_names[class_id]
        conf = confidences[i]
        
        class_counts[detected_class_name] += 1
        class_confidences[detected_class_name].append(conf)

        if show_boxes:
            color = COLOR_MAP.get(detected_class_name, COLOR_MAP['default'])
            thickness = 3 if class_id in parasite_IDs else 2
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), color, thickness)

        if show_labels:
            if show_confidence:
                label = f"{detected_class_name}: {conf:.2f}"
            else:
                label = detected_class_name
            
            # Better text visibility with background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_cv, (x, y-text_h-8), (x+text_w, y), color, -1)
            cv2.putText(img_cv, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    return img_cv, class_counts, class_confidences, dim_warning

# --- Diagnostic Interpretation ---
def get_diagnostic_interpretation(parasitemia, total_parasites, class_counts):
    """Provide clinical interpretation based on WHO guidelines"""
    interpretation = []
    severity = "Normal"
    
    if parasitemia == 0:
        interpretation.append("✅ No parasites detected")
        severity = "Negative"
    elif parasitemia < 0.1:
        interpretation.append("⚠️ Very Low Parasitemia (<0.1%)")
        severity = "Minimal"
    elif parasitemia < 2:
        interpretation.append("⚠️ Low Parasitemia (0.1-2%)")
        severity = "Mild"
    elif parasitemia < 5:
        interpretation.append("⚠️ Moderate Parasitemia (2-5%)")
        severity = "Moderate"
    else:
        interpretation.append("🚨 High Parasitemia (>5%) - Requires immediate attention")
        severity = "Severe"
    
    # Stage-specific notes
    parasite_stages = ['schizont', 'ring', 'gametocyte', 'trophozoite']
    stage_counts = {stage: class_counts.get(stage, 0) for stage in parasite_stages}
    
    if stage_counts['gametocyte'] > 0:
        interpretation.append(f"🔬 Gametocytes detected ({stage_counts['gametocyte']}) - transmission stage present")
    if stage_counts['schizont'] > 0:
        interpretation.append(f"🔬 Schizonts detected ({stage_counts['schizont']}) - mature stage present")
    if stage_counts['ring'] > 0:
        interpretation.append(f"🔬 Ring forms detected ({stage_counts['ring']}) - early stage infection")
    if stage_counts['trophozoite'] > 0:
        interpretation.append(f"🔬 Trophozoites detected ({stage_counts['trophozoite']}) - active feeding stage")
    
    return severity, interpretation

def create_confidence_histogram(class_confidences):
    """Create histogram of detection confidences"""
    conf_data = []
    for class_name, confs in class_confidences.items():
        for conf in confs:
            conf_data.append({'Class': class_name, 'Confidence': conf})
    
    if not conf_data:
        return None
    
    df = pd.DataFrame(conf_data)
    chart = alt.Chart(df).mark_bar(opacity=0.7).encode(
        x=alt.X('Confidence:Q', bin=alt.Bin(maxbins=20), title='Confidence Score'),
        y=alt.Y('count()', title='Number of Detections'),
        color=alt.Color('Class:N', legend=alt.Legend(title="Class")),
        tooltip=['Class:N', 'count()']
    ).properties(
        width='container',
        height=300,
        title='Detection Confidence Distribution'
    )
    return chart

# --- Main Interface ---
st.header("🩸 Upload Blood Smear Images")

# Example images section
with st.expander("📸 Need test images? Download example blood smears", expanded=False):
    st.info("""
    For testing purposes, you can use:
    - High-resolution microscopy images (recommended: 1280x1280 or higher)
    - JPEG, PNG, or BMP formats
    - Well-stained Giemsa or Field stain preparations
    - Images with clear cell boundaries
    """)

uploaded_files = st.file_uploader(
    "Choose one or more image files", 
    type=['jpg','jpeg','png','bmp'], 
    accept_multiple_files=True,
    help="Upload microscopy images of blood smears (1280x1280 recommended)"
)

# Chart settings
st.sidebar.markdown("---")
st.sidebar.header("📊 Chart Settings")
chart_mode = st.sidebar.radio(
    'Chart Mode',
    ['Counts', 'Percentages'],
    index=0,
    key='chart_mode_radio'
)
show_confidence_dist = st.sidebar.checkbox("Show Confidence Distribution", value=True)

# Reset button
if st.sidebar.button("🔄 Reset All", help="Clear all results and start fresh"):
    st.rerun()

if uploaded_files and session and class_names:
    st.subheader(f"📊 Processing {len(uploaded_files)} Image(s)...")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        run_button = st.button("▶️ Run Detection", type="primary", use_container_width=True)
    with col2:
        st.info(f"⚙️ Conf: {confidence_threshold:.2f} | NMS: {nms_threshold:.2f}")
    
    if run_button:
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_images = len(uploaded_files)
        results_summary = []
        all_annotated_images = []
        total_processing_time = 0

        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing image {i+1}/{total_images}: {file.name}...")
            
            # Start timer
            start_time = time.time()
            
            image = Image.open(file)
            detected_img_cv, class_counts, class_confidences, dim_warning = process_image(
                session, image, confidence_threshold, nms_threshold, class_names,
                show_boxes, show_labels, show_confidence, show_only_parasites, 
                color_scheme, class_thresholds if use_class_thresholds else None
            )
            detected_img_rgb = cv2.cvtColor(detected_img_cv, cv2.COLOR_BGR2RGB)
            
            # End timer
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            
            # Store annotated image for download
            all_annotated_images.append((file.name, detected_img_rgb))
            
            col_img, col_data = st.columns([2,1])
            
            with col_img:
                st.image(detected_img_rgb, caption=f"Processed: {file.name}", use_container_width=True)
                st.caption(f"⏱️ Processing time: {processing_time:.2f}s")
                if dim_warning:
                    st.warning(dim_warning)
            
            with col_data:
                st.markdown(f"#### 🧪 Results for **{file.name}**")
                
                # Calculate metrics
                parasite_stages = ['schizont','ring','gametocyte','trophozoite']
                total_parasite_count = sum(class_counts.get(stage,0) for stage in parasite_stages)
                total_detections = sum(class_counts.values())
                parasitemia = (total_parasite_count/total_detections)*100 if total_detections>0 else 0.0
                
                # Diagnostic interpretation
                severity, interpretation = get_diagnostic_interpretation(parasitemia, total_parasite_count, class_counts)
                
                # Key metrics with color coding
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Parasites", total_parasite_count)
                with col2:
                    if severity == "Severe":
                        st.metric("Parasitemia", f"{parasitemia:.2f}%", delta="High", delta_color="inverse")
                    elif severity == "Negative":
                        st.metric("Parasitemia", f"{parasitemia:.2f}%", delta="Negative", delta_color="off")
                    else:
                        st.metric("Parasitemia", f"{parasitemia:.2f}%")
                
                # Interpretation box
                st.info("\n\n".join(interpretation))
                
                with st.expander("📋 Detailed Counts & Confidence", expanded=False):
                    st.markdown("##### Class-wise Detection")
                    for class_name in class_names:
                        count = class_counts.get(class_name, 0)
                        if count > 0 and class_confidences[class_name]:
                            avg_conf = np.mean(class_confidences[class_name])
                            min_conf = np.min(class_confidences[class_name])
                            max_conf = np.max(class_confidences[class_name])
                            st.write(f"**{class_name.title()}**: {count} detections")
                            st.caption(f"Confidence - Avg: {avg_conf:.2f}, Min: {min_conf:.2f}, Max: {max_conf:.2f}")
                        else:
                            st.write(f"**{class_name.title()}**: 0 detections")
                    
                    st.metric("Total Detections", total_detections)

                # Visualization
                counts_df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
                if not counts_df.empty and counts_df["Count"].sum() > 0:
                    total = counts_df["Count"].sum()
                    counts_df["Percentage"] = (counts_df["Count"] / total) * 100
                    
                    if chart_mode == 'Counts':
                        x_field, x_title = "Count", "Number of Detections"
                    else:
                        x_field, x_title = "Percentage", "Detections (%)"

                    chart = (
                        alt.Chart(counts_df)
                        .mark_bar()
                        .encode(
                            x=alt.X(f"{x_field}:Q", title=x_title),
                            y=alt.Y("Class:N", sort='-x', title="Class"),
                            color=alt.Color("Class:N", legend=None),
                            tooltip=[
                                alt.Tooltip('Class:N'),
                                alt.Tooltip("Count:Q"),
                                alt.Tooltip("Percentage:Q", format=".2f")
                            ]
                        )
                        .properties(width="container", height=300)
                    )
                    st.altair_chart(chart, use_container_width=True)
                    
                    # Confidence distribution
                    if show_confidence_dist:
                        conf_chart = create_confidence_histogram(class_confidences)
                        if conf_chart:
                            st.altair_chart(conf_chart, use_container_width=True)

                # Store for CSV
                results_summary.append({
                    "Image": file.name,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Processing_Time_s": f"{processing_time:.2f}",
                    "Total Parasites": total_parasite_count,
                    "Total Detections": total_detections,
                    "Parasitemia (%)": f"{parasitemia:.2f}",
                    "Severity": severity,
                    **{f"{cls}": class_counts.get(cls, 0) for cls in class_names}
                })

            st.divider()
            progress_bar.progress((i+1)/total_images)

        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Detection complete! Total processing time: {total_processing_time:.2f}s")

        # Summary Statistics
        if len(results_summary) > 1:
            st.subheader("📊 Batch Summary Statistics")
            df_summary = pd.DataFrame(results_summary)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_parasitemia = df_summary['Parasitemia (%)'].str.rstrip('%').astype(float).mean()
                st.metric("Avg Parasitemia", f"{avg_parasitemia:.2f}%")
            with col2:
                total_parasites_all = df_summary['Total Parasites'].sum()
                st.metric("Total Parasites", total_parasites_all)
            with col3:
                positive_count = (df_summary['Total Parasites'] > 0).sum()
                st.metric("Positive Samples", f"{positive_count}/{len(results_summary)}")
            with col4:
                avg_time = df_summary['Processing_Time_s'].astype(float).mean()
                st.metric("Avg Time/Image", f"{avg_time:.2f}s")

        # Download Options
        st.subheader("📥 Download Results")
        col1, col2, col3 = st.columns(3)
        
        # CSV Export
        with col1:
            if results_summary:
                df_results = pd.DataFrame(results_summary)
                csv_buffer = io.StringIO()
                df_results.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📄 Download CSV Report",
                    data=csv_buffer.getvalue(),
                    file_name=f"malaria_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Export detailed results with timestamps",
                    use_container_width=True
                )
        
        # Download Annotated Images
        with col2:
            if all_annotated_images:
                # Create ZIP file with annotated images
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for img_name, img_array in all_annotated_images:
                        # Convert numpy array to PIL Image
                        pil_img = Image.fromarray(img_array)
                        img_buffer = BytesIO()
                        pil_img.save(img_buffer, format='PNG')
                        zip_file.writestr(f"annotated_{img_name}", img_buffer.getvalue())
                
                st.download_button(
                    label="🖼️ Download Annotated Images",
                    data=zip_buffer.getvalue(),
                    file_name=f"annotated_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    help="Download all processed images with bounding boxes",
                    use_container_width=True
                )
        
        # PDF Report (placeholder - would need reportlab library)
        with col3:
            st.button(
                "📑 Generate PDF Report",
                help="PDF report generation (requires additional setup)",
                disabled=True,
                use_container_width=True
            )
            st.caption("Coming soon!")

elif not session:
    st.error("❌ ONNX model could not be loaded. Please check the path and file integrity.")
else:
    st.info("👆 Upload one or more blood smear images to begin analysis")
    
    # Show some guidance when no images are uploaded
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📋 Preparation")
        st.write("""
        - Ensure blood smears are well-stained
        - Use high-resolution images
        - Good lighting and focus
        """)
    with col2:
        st.markdown("### ⚙️ Settings")
        st.write("""
        - Adjust confidence threshold
        - Choose color scheme
        - Enable per-class thresholds
        """)
    with col3:
        st.markdown("### 📊 Results")
        st.write("""
        - View parasitemia calculations
        - Download CSV reports
        - Export annotated images
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("""
**Model:** YOLOv8n Fine-tuned on P.vivax dataset

**Input Resolution:** 1280x1280

**Classes:** RBC, Leukocyte, Schizont, Ring, Gametocyte, Trophozoite

**Performance:**
- mAP50: 0.400
- mAP50-95: 0.301
- F1-Score: 0.391

**Version:** 2.0

**Note:** For research purposes only. Not for clinical diagnosis.
""")
