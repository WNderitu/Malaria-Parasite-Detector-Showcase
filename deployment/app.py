import streamlit as st
import onnxruntime as ort
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
import io
import altair as alt

# Set page configuration
st.set_page_config(
    page_title="Malaria Parasite (P.vivax) Detector using YOLOv8n v2",
    layout="wide"
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
    st.success(" ✅ Class names file loaded successfully.")

# --- Load Model & Classes ---
@st.cache_resource
def load_onnx_model(model_path):
    try:
        session = ort.InferenceSession(model_path)
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

st.title("🔬 Malaria Parasite (P.vivax) Detection using YOLOV8n")

# --- Dynamic Sidebar ---
st.sidebar.header("⚙️ Model & Visualization Settings")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
nms_threshold = st.sidebar.slider("NMS Threshold", 0.0, 1.0, 0.45, 0.05)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
show_labels = st.sidebar.checkbox("Show Class Labels", value=True)
show_only_parasites = st.sidebar.checkbox("Show Only Parasite Detections", value=False)
color_scheme = st.sidebar.selectbox("Color Scheme", ["Default", "High Contrast", "Pastel"], index=0)

# --- Image Processing Function ---
def process_image(session, image, conf_threshold, nms_threshold, class_names,
                  show_boxes=True, show_labels=True, show_only_parasites=False, color_scheme="Default"):
    INPUT_WIDTH, INPUT_HEIGHT = 1280, 1280
    
    # Convert PIL → numpy
    img_cv_rgb = np.array(image.convert("RGB"))
    img_resized = cv2.resize(img_cv_rgb, (INPUT_WIDTH, INPUT_HEIGHT))
    img = img_resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)   # add batch dim
    
    # Run inference
    outputs = session.run(None, {"images": img})
    preds = outputs[0][0]   # shape (33600, 10)
    
    boxes, confidences, class_ids = [], [], []
    class_counts = {name: 0 for name in class_names}
    
    parasite_IDs = {2, 3, 4, 5}
    
    # Color maps
    DEFAULT_COLOR_MAP = {
        'red blood cell': (0, 0, 255),
        'leukocyte': (255, 255, 255),
        'schizont': (0, 255, 255),
        'ring': (0, 255, 0),
        'gametocyte': (255, 0, 255),
        'trophozoite': (255, 0, 0),
        'default': (128, 128, 128)
    }
    HIGH_CONTRAST_MAP = {k: (255, 255, 0) for k in DEFAULT_COLOR_MAP}
    PASTEL_MAP = {k: (200, 180, 255) for k in DEFAULT_COLOR_MAP}
    COLOR_MAP = DEFAULT_COLOR_MAP if color_scheme=="Default" else HIGH_CONTRAST_MAP if color_scheme=="High Contrast" else PASTEL_MAP
    
    # Decode predictions
    for det in preds:
        x, y, w, h, obj_conf, *cls_scores = det
        class_id = np.argmax(cls_scores)
        class_conf = cls_scores[class_id]
        conf = obj_conf * class_conf   # final confidence

        if conf < conf_threshold:
            continue
        if class_id >= len(class_names):
            continue

        # Scale back to original image size
        x_scale = img_cv_rgb.shape[1] / INPUT_WIDTH
        y_scale = img_cv_rgb.shape[0] / INPUT_HEIGHT
        cx, cy = x * x_scale, y * y_scale
        bw, bh = w * x_scale, h * y_scale
        x1, y1 = int(cx - bw/2), int(cy - bh/2)

        boxes.append([x1, y1, int(bw), int(bh)])
        confidences.append(float(conf))
        class_ids.append(class_id)
    
    if not boxes:
        return img_cv_rgb, class_counts
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
    else:
        return img_cv_rgb, class_counts
    
    if show_only_parasites:
        indices = [i for i in indices if class_ids[i] in parasite_IDs]
    
    for i in indices:
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        detected_class_name = class_names[class_id]
        class_counts[detected_class_name] += 1
        
        if show_boxes:
            color = COLOR_MAP.get(detected_class_name, COLOR_MAP['default'])
            cv2.rectangle(img_cv_rgb, (x, y), (x+w, y+h), color, 2)
        
        if show_labels:
            label = f"{detected_class_name}: {confidences[i]:.2f}"
            cv2.putText(img_cv_rgb, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(img_cv_rgb, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255,255,255), 1, cv2.LINE_AA)
    
    return img_cv_rgb, class_counts

# --- User Interface ---
st.header(" 🩸 Upload image of blood smear slide")
uploaded_files = st.file_uploader("Choose one or more image files", type=['jpg','jpeg','png','bmp'], accept_multiple_files=True)

st.sidebar.header("📊 Chart Settings")
chart_mode = st.sidebar.radio(
    'Chart Mode',
    ['Counts', 'Percentages'],
    index=0,
    key='chart_mode_radio'
)

if uploaded_files and session and class_names:
    st.subheader(f"Processing {len(uploaded_files)} Images...")
    if st.button("  ▶️  Run detection"):
        progress_bar = st.progress(0)
        total_images = len(uploaded_files)
        results_summary = []

        for i, file in enumerate(uploaded_files):
            image = Image.open(file)
            detected_img_cv, class_counts = process_image(
                session, image, confidence_threshold, nms_threshold, class_names,
                show_boxes, show_labels, show_only_parasites, color_scheme
            )
            detected_img_rgb = cv2.cvtColor(detected_img_cv, cv2.COLOR_BGR2RGB)
            
            col_img, col_data = st.columns([2,1])
            with col_img:
                st.image(detected_img_rgb, caption=f"Processed: {file.name}", use_container_width=True)
            
            with col_data:
                st.markdown(f"#### 🧪 Results for **{file.name}**")
                
                parasite_stages = ['schizont','ring','gametocyte','trophozoite']
                total_parasite_count = sum(class_counts.get(stage,0) for stage in parasite_stages)
                total_detections = sum(class_counts.values())
                parasitemia = (total_parasite_count/total_detections)*100 if total_detections>0 else 0.0
                parasitemia_display = f"{parasitemia:.2f} %"
                               
                st.metric("**Total Parasite Count (All Stages)**", total_parasite_count)
                st.metric(
                    label='**Estimated Parasitemia Rate**',
                    value=parasitemia_display,
                    help=("Calculated as: (Total Parasite Detections / Total Cell Detections) * 100.")
                )
                st.info(f"**Total Objects Counted:** {total_detections}")
