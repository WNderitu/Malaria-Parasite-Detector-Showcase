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

# Set page configuration
st.set_page_config(
    page_title="Malaria Parasite (P.vivax) Detector v2",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- File Paths ---
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'best.onnx')
classes_path = os.path.join(base_path, 'classes.txt')

# --- Validate Files ---
if not os.path.exists(model_path):
    st.error(f"ONNX model not found at: {model_path}")
else:
    st.success("✅ ONNX model loaded successfully.")

if not os.path.exists(classes_path):
    st.error(f"Class names file not found at: {classes_path}")
else:
    st.success("✅ Class names file loaded successfully.")

# --- Load Model & Classes ---
@st.cache_resource
def load_onnx_model(model_path):
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        return session
    except Exception as e:
        st.error(f"Error loading ONNX model: {e}. Attempted path: {model_path}")
        return None

@st.cache_data
def load_class_names(classes_path):
    try:
        with open(classes_path, "r") as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        st.error(f"Error loading class names: {e}. Attempted path: {classes_path}")
        return []

session = load_onnx_model(model_path)
class_names = load_class_names(classes_path)

# --- Header ---
st.title("🔬 Malaria Parasite (P.vivax) Detector v2")

# Add model performance warning
with st.expander("⚠️ Model Performance Information", expanded=False):
    st.warning("""
    **Current Model Limitations:**
    - **Schizont Detection**: Low accuracy (18.1% precision, 9.1% recall)
    - **Gametocyte Detection**: Very poor (0% precision/recall - model struggles with this class)
    - **Ring Detection**: Moderate accuracy (46% precision, 13.6% recall)
    - **Trophozoite Detection**: Good accuracy (57% precision, 64.7% recall)
    - **Red Blood Cell Detection**: Excellent (90% precision, 95.2% recall)
    
    **Recommendations:**
    - Use results as screening tool, not diagnostic confirmation
    - Manual verification recommended for rare parasite stages
    - Best performance on trophozoite and RBC detection
    """)

# Add WHO parasitemia classification guide
with st.expander("📖 WHO Parasitemia Classification Guide", expanded=False):
    st.markdown("""
    ### Parasitemia Calculation
    **Parasitemia (%) = (Total Parasites Detected / Total Cells Detected) × 100**
    
    This represents the percentage of cells infected with malaria parasites.
    
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
    
    This tool is designed for **research and screening purposes only**. It should NOT replace:
    - Professional microscopic examination by trained technicians
    - Clinical diagnosis by qualified healthcare providers
    - Standard diagnostic protocols and confirmatory testing
    
    Always consult healthcare professionals for medical decisions.
    """)
    

# --- Dynamic Sidebar ---
st.sidebar.header("⚙️ Model & Visualization Settings")

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

# Visualization settings
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
show_labels = st.sidebar.checkbox("Show Class Labels", value=True)
show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)
show_only_parasites = st.sidebar.checkbox("Show Only Parasite Detections", value=False)
color_scheme = st.sidebar.selectbox("Color Scheme", ["Default", "High Contrast", "Pastel"], index=0)

# Add class-specific confidence thresholds
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Advanced: Per-Class Thresholds")
use_class_thresholds = st.sidebar.checkbox("Enable Class-Specific Thresholds", value=False)

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
        'leukocyte': (255, 255, 0),         # Bright Yellow (changed from white for visibility)
        'schizont': (0, 255, 255),          # Cyan
        'ring': (0, 255, 0),                # Green
        'gametocyte': (255, 0, 255),        # Magenta
        'trophozoite': (255, 165, 0),       # Orange
        'default': (128, 128, 128)
    }
    HIGH_CONTRAST_MAP = {
        'red blood cell': (255, 0, 0),
        'leukocyte': (255, 255, 0),         # Bright Yellow
        'schizont': (0, 255, 255),
        'ring': (0, 255, 0),
        'gametocyte': (255, 0, 255),
        'trophozoite': (255, 128, 0),
        'default': (255, 255, 255)
    }
    PASTEL_MAP = {
        'red blood cell': (255, 182, 193),
        'leukocyte': (255, 250, 205),       # Light Yellow (more visible than pale cream)
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
        return img_cv, class_counts, class_confidences

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
    else:
        return img_cv, class_counts, class_confidences

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
            thickness = 3 if class_id in parasite_IDs else 2  # Thicker boxes for parasites
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

    return img_cv, class_counts, class_confidences

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

# --- User Interface ---
st.header("🩸 Upload Image of Malaria Blood Smear")
uploaded_files = st.file_uploader(
    "Choose one or more image files", 
    type=['jpg','jpeg','png','bmp'], 
    accept_multiple_files=True,
    help="Upload microscopy images of blood smears (1280x1280 recommended)"
)

st.sidebar.header("📊 Chart Settings")
chart_mode = st.sidebar.radio(
    'Chart Mode',
    ['Counts', 'Percentages'],
    index=0,
    key='chart_mode_radio'
)

if uploaded_files and session and class_names:
    st.subheader(f"Processing {len(uploaded_files)} Image(s)...")
    
    if st.button("▶️ Run Detection", type="primary"):
        progress_bar = st.progress(0)
        total_images = len(uploaded_files)
        results_summary = []
        all_detections = []

        for i, file in enumerate(uploaded_files):
            image = Image.open(file)
            detected_img_cv, class_counts, class_confidences = process_image(
                session, image, confidence_threshold, nms_threshold, class_names,
                show_boxes, show_labels, show_confidence, show_only_parasites, 
                color_scheme, class_thresholds if use_class_thresholds else None
            )
            detected_img_rgb = cv2.cvtColor(detected_img_cv, cv2.COLOR_BGR2RGB)
            
            col_img, col_data = st.columns([2,1])
            
            with col_img:
                st.image(detected_img_rgb, caption=f"Processed: {file.name}", use_container_width=True)
            
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
                
                with st.expander("📋 Detailed Counts & Confidence"):
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

                # Store for CSV
                results_summary.append({
                    "Image": file.name,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Total Parasites": total_parasite_count,
                    "Total Detections": total_detections,
                    "Parasitemia (%)": f"{parasitemia:.2f}",
                    "Severity": severity,
                    **{f"{cls}": class_counts.get(cls, 0) for cls in class_names}
                })

            st.divider()
            progress_bar.progress((i+1)/total_images)

        progress_bar.empty()
        st.success("✅ Detection and quantification complete!")

        # Summary Statistics
        if len(results_summary) > 1:
            st.subheader("📊 Batch Summary Statistics")
            df_summary = pd.DataFrame(results_summary)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_parasitemia = df_summary['Parasitemia (%)'].str.rstrip('%').astype(float).mean()
                st.metric("Average Parasitemia", f"{avg_parasitemia:.2f}%")
            with col2:
                total_parasites_all = df_summary['Total Parasites'].sum()
                st.metric("Total Parasites (All Images)", total_parasites_all)
            with col3:
                positive_count = (df_summary['Total Parasites'] > 0).sum()
                st.metric("Positive Samples", f"{positive_count}/{len(results_summary)}")

        # CSV Export
        if results_summary:
            df_results = pd.DataFrame(results_summary)
            csv_buffer = io.StringIO()
            df_results.to_csv(csv_buffer, index=False)
            st.sidebar.download_button(
                label="📥 Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"malaria_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Export detailed results with timestamps"
            )

elif not session:
    st.error("❌ ONNX model could not be loaded. Please check the path and file integrity.")
else:
    st.info("👆 Upload one or more blood smear images to begin analysis")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("""
**Model:** YOLOv8n Fine-tuned on P.vivax dataset
**Input Resolution:** 1280x1280
**Classes:** RBC, Leukocyte, Schizont, Ring, Gametocyte, Trophozoite

**Note:** This tool is for research purposes only and should not replace professional medical diagnosis.
""")
