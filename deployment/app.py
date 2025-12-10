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

# --- Diagnostic and Chart Helper Functions (Fixes NameError) ---

def get_diagnostic_interpretation(parasitemia, total_parasites, class_counts):
    """Provides a severity level and interpretation based on WHO guidelines."""
    severity = "Negative"
    interpretation = []
    
    if total_parasites == 0:
        severity = "Negative"
        interpretation.append("Diagnosis: Negative.")
        interpretation.append("No parasites detected in this sample.")
    elif parasitemia < 0.1:
        severity = "Minimal"
        interpretation.append(f"Diagnosis: Minimal Parasitemia ({parasitemia:.2f}%)")
        interpretation.append("Monitor closely. Low-level infection detected, requires microscopy confirmation.")
    elif parasitemia < 2.0:
        severity = "Mild"
        interpretation.append(f"Diagnosis: Mild Parasitemia ({parasitemia:.2f}%)")
        interpretation.append("Low-level infection. Outpatient treatment may be appropriate. Follow up with physician.")
    elif parasitemia < 5.0:
        severity = "Moderate"
        interpretation.append(f"Diagnosis: Moderate Parasitemia ({parasitemia:.2f}%)")
        interpretation.append("Significant infection level. Close clinical monitoring and treatment required.")
    else: # >= 5.0
        severity = "Severe"
        interpretation.append(f"Diagnosis: Severe Parasitemia ({parasitemia:.2f}%)")
        interpretation.append("🔴 HIGH RISK: High parasite burden detected. IMMEDIATE medical attention required.")

    # Additional Checks for P. vivax severity factors
    if class_counts.get('schizont', 0) > 0:
        interpretation.append("⚠️ Schizonts Detected: Presence of schizonts suggests active, late-stage multiplication and is an indicator of potentially severe disease.")
    if class_counts.get('gametocyte', 0) > 0:
        interpretation.append("🦠 Gametocytes Detected: Patient may be capable of transmitting the disease to mosquitoes.")
        
    return severity, interpretation

def create_confidence_histogram(class_confidences):
    """Creates an Altair histogram for confidence score distribution."""
    data = []
    for class_name, confs in class_confidences.items():
        if confs:
            for conf in confs:
                data.append({'Class': class_name, 'Confidence': conf})

    if not data:
        return None

    df_conf = pd.DataFrame(data)

    chart = alt.Chart(df_conf).mark_bar().encode(
        x=alt.X("Confidence:Q", bin=alt.Bin(maxbins=20), title="Confidence Score (Binned)"),
        y=alt.Y("count()", title="Number of Detections"),
        color=alt.Color("Class:N", legend=alt.Legend(title="Class")),
        tooltip=["Class:N", alt.Tooltip("Confidence:Q", bin=True), "count():Q"]
    ).properties(
        title="Confidence Score Distribution"
    )
    return chart

# --- Color Legend in Sidebar ---
def display_color_legend(color_scheme="Default"):
    # (Simplified color map creation for legend)
    COLOR_MAPS = {
        "Default": {'Red Blood Cell': (0, 0, 255), 'Leukocyte': (255, 255, 0), 'Schizont': (0, 255, 255), 'Ring': (0, 255, 0), 'Gametocyte': (255, 0, 255), 'Trophozoite': (255, 165, 0)},
        "High Contrast": {'Red Blood Cell': (255, 0, 0), 'Leukocyte': (255, 255, 0), 'Schizont': (0, 255, 255), 'Ring': (0, 255, 0), 'Gametocyte': (255, 0, 255), 'Trophozoite': (255, 128, 0)},
        "Pastel": {'Red Blood Cell': (255, 182, 193), 'Leukocyte': (255, 250, 205), 'Schizont': (175, 238, 238), 'Ring': (144, 238, 144), 'Gametocyte': (221, 160, 221), 'Trophozoite': (255, 218, 185)},
    }
    COLOR_MAP = COLOR_MAPS.get(color_scheme, COLOR_MAPS["Default"])
    
    st.sidebar.markdown("### 🎨 Color Legend")
    for class_name, color in COLOR_MAP.items():
        # Convert BGR (used in CV2 mapping below) to RGB for display
        b, g, r = color # The actual color map in process_image uses BGR (CV2 default)
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
    "Global Confidence Threshold", 
    0.20, 0.40, 0.25, 0.01,
    help="Constrained to 0.20-0.40. Lower values detect more rare parasites but may increase false positives"
)
st.sidebar.info("NMS IoU is fixed: 0.75 for Cells, 0.50 for Parasites (Multi-Stage NMS)")

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

# --- Enhanced Image Processing Function with Multi-Stage NMS ---
def process_image(session, image, conf_threshold, class_names,
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

    # --- NMS Parameters for Multi-Stage approach ---
    NMS_TH_PARASITE = 0.50  # Stricter NMS for small, critical parasite detections
    NMS_TH_RBC_LEUKO = 0.75 # Lenient NMS for large, high-overlap cells (RBCs/Leukocytes)
    
    # Define classes by their group
    parasite_classes = {'schizont', 'ring', 'gametocyte', 'trophozoite'}
    cell_classes = {'red blood cell', 'leukocyte'}
    parasite_IDs = {class_names.index(c) for c in parasite_classes if c in class_names}
    
    # Initialize lists to hold raw, pre-NMS detections, separated by group
    all_boxes, all_confidences, all_class_ids = [], [], []
    
    # Enhanced Color maps (CV2 uses BGR format internally)
    DEFAULT_COLOR_MAP = {
        'red blood cell': (0, 0, 255),      # Blue (Appears Red in BGR conversion)
        'leukocyte': (255, 255, 0),         # Cyan (Appears Yellow)
        'schizont': (0, 255, 255),          # Yellow (Appears Cyan)
        'ring': (0, 255, 0),                # Green (Appears Green)
        'gametocyte': (255, 0, 255),        # Magenta (Appears Magenta)
        'trophozoite': (255, 165, 0),       # Orange (Appears Orange)
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

    # --- 1. Filter Raw Detections by Confidence ---
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
            
            all_boxes.append([x, y, w, h])
            all_confidences.append(float(confidence))
            all_class_ids.append(class_id)

    if not all_boxes:
        return img_cv, {name: 0 for name in class_names}, {name: [] for name in class_names}, dim_warning

    # --- 2. Separate Detections by NMS Group (Multi-Stage NMS Prep) ---
    # We use a list of tuples to keep the original index mapping
    parasite_detections, cell_detections = [], []
    
    for i in range(len(all_boxes)):
        box = all_boxes[i]
        conf = all_confidences[i]
        class_id = all_class_ids[i]
        detection = (box, conf, class_id)
        detected_class_name = class_names[class_id]
        
        if detected_class_name in parasite_classes:
            parasite_detections.append(detection)
        elif detected_class_name in cell_classes:
            cell_detections.append(detection)

    # --- 3. Apply Group-Specific NMS ---
    def apply_nms(detections, iou_threshold):
        if not detections:
            return []
        boxes_list = [d[0] for d in detections]
        conf_list = [d[1] for d in detections]
        
        # NOTE: conf_threshold is passed to NMSBoxes as the confidence threshold 
        # (even though we already filtered by it, it's required by the function signature)
        indices = cv2.dnn.NMSBoxes(boxes_list, conf_list, conf_threshold, iou_threshold)
        
        # Flatten and return the filtered detection tuples
        filtered_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                filtered_detections.append(detections[i])
        return filtered_detections

    # Apply NMS with different thresholds
    # We use the user's conf_threshold as the base for NMS, which is passed in the apply_nms function
    final_parasite_detections = apply_nms(parasite_detections, NMS_TH_PARASITE)
    final_cell_detections = apply_nms(cell_detections, NMS_TH_RBC_LEUKO)

    # --- 4. Combine Final Filtered Detections and Post-Filter ---
    final_detections = final_parasite_detections + final_cell_detections
    
    if show_only_parasites:
        final_detections = [d for d in final_detections if d[2] in parasite_IDs]

    class_counts = {name: 0 for name in class_names}
    class_confidences = {name: [] for name in class_names}

    # --- 5. Draw Boxes and Calculate Metrics ---
    for box, conf, class_id in final_detections:
        x, y, w, h = box
        detected_class_name = class_names[class_id]
        
        # Update counts and confidences
        class_counts[detected_class_name] += 1
        class_confidences[detected_class_name].append(conf)

        if show_boxes:
            color = COLOR_MAP.get(detected_class_name, COLOR_MAP['default'])
            thickness = 3 if class_id in parasite_IDs else 2
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), color, thickness)

        if show_labels:
            label_text = f"{detected_class_name}: {conf:.2f}" if show_confidence else detected_class_name
            
            # Better text visibility with background
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_cv, (x, y-text_h-8), (x+text_w, y), color, -1)
            cv2.putText(img_cv, label_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    return img_cv, class_counts, class_confidences, dim_warning

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
        # Displaying the optimized settings
        st.info(f"⚙️ Conf: {confidence_threshold:.2f} | NMS: Optimized")
    
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
            # UPDATED: The process_image call no longer includes nms_threshold
            detected_img_cv, class_counts, class_confidences, dim_warning = process_image(
                session, image, confidence_threshold, class_names,
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
                total_cells = sum(class_counts.values())
                # Use total_cells in the denominator to be inclusive of RBCs and Leukocytes, 
                # as per the WHO guide in the expandable section.
                parasitemia = (total_parasite_count/total_cells)*100 if total_cells>0 else 0.0
                
                # Diagnostic interpretation (Fixes NameError)
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
