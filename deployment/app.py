import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
import io
import altair as alt

# Set page configuration
st.set_page_config(
    page_title="Malaria Parasite (P.vivax) Detector using YOLOv8n",
    layout="wide"
)

# --- Standard Class Definitions ---
STANDARD_CLASSES = [
    'red blood cell', 
    'leukocyte', 
    'schizont', 
    'ring', 
    'gametocyte', 
    'trophozoite'
]
PARASITE_STAGES = ['schizont', 'ring', 'gametocyte', 'trophozoite']

# --- File Paths ---
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'best.onnx')
classes_path = os.path.join(base_path, 'classes.txt') 

# --- Validate Files ---
if not os.path.exists(model_path):
    st.error(f"ONNX model not found at: {model_path}")
else:
    st.success("✅ ONNX model found.")

if not os.path.exists(classes_path):
    st.warning(f"Class names file not found at: {classes_path}. Using hardcoded classes for validation.")
else:
    st.success(" ✅ Class names file found.")

# --- Load Model & Classes ---
@st.cache_resource
def load_onnx_model(model_path):
    try:
        net = cv2.dnn.readNet(model_path)
        return net
    except Exception as e:
        st.error(f"Error loading ONNX model: {e}. Attempted path: {model_path}")
        return None

@st.cache_data
def load_class_names(classes_path):
    try:
        if os.path.exists(classes_path):
             with open(classes_path, "r") as f:
                return [line.strip() for line in f.readlines()]
        else:
             return STANDARD_CLASSES 
    except Exception as e:
        st.error(f"Error loading class names from file: {e}. Using hardcoded standard list.")
        return STANDARD_CLASSES

net = load_onnx_model(model_path)
class_names = load_class_names(classes_path)

if len(class_names) != len(STANDARD_CLASSES):
    st.warning(f"Class count mismatch! File has {len(class_names)} classes, expected {len(STANDARD_CLASSES)}. Please verify 'classes.txt'.")
    class_names = STANDARD_CLASSES

st.title("🔬 Malaria Parasite (P.vivax) Detection using YOLOV8n")

# --- Dynamic Sidebar ---
st.sidebar.header("⚙️ Model & Visualization Settings")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
nms_threshold = st.sidebar.slider("NMS Threshold", 0.0, 1.0, 0.35, 0.05)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
show_labels = st.sidebar.checkbox("Show Class Labels", value=True)
show_only_parasites = st.sidebar.checkbox("Show Only Parasite Detections", value=False)
color_scheme = st.sidebar.selectbox("Color Scheme", ["Default", "High Contrast", "Pastel"], index=0)

# --- Image Processing Function ---
def process_image(net, image, conf_threshold, nms_threshold, class_names,
                  show_boxes=True, show_labels=True, show_only_parasites=False, color_scheme="Default"):
    
    INPUT_WIDTH, INPUT_HEIGHT = 1280, 1280
    
    img_cv_rgb = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img_cv_rgb, cv2.COLOR_RGB2BGR)
    
    blob = cv2.dnn.blobFromImage(img_cv, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)

    # --- Debugging lines ---
    print("Blob shape:", blob.shape)
    print("Unconnected output layers:", net.getUnconnectedOutLayersNames())
    print("All layer names:", net.getLayerNames())

    try:
        preds = net.forward()
        print("Preds type:", type(preds))
        if isinstance(preds, list):
            print("Preds[0] shape:", preds[0].shape)
        else:
            print("Preds shape:", preds.shape)
    except Exception as e:
        print("Error during forward pass:", e)
        return img_cv, {name: 0 for name in class_names}

    detections = preds[0].T
    boxes, confidences, class_ids = [], [], []
    
    parasite_IDs = {class_names.index(cls) for cls in PARASITE_STAGES if cls in class_names}

    DEFAULT_COLOR_MAP = {
        'red blood cell': (0, 0, 255),
        'leukocyte': (255, 255, 255),
        'schizont': (0, 255, 255),
        'ring': (0, 255, 0),
        'gametocyte': (255, 0, 255),
        'trophozoite': (255, 0, 0),
        'default': (128, 128, 128)
    }
    HIGH_CONTRAST_MAP = {k: (0, 0, 255) for k in DEFAULT_COLOR_MAP} 
    PASTEL_MAP = {k: (200, 180, 255) for k in DEFAULT_COLOR_MAP}

    if color_scheme == "High Contrast":
        COLOR_MAP = HIGH_CONTRAST_MAP
    elif color_scheme == "Pastel":
        COLOR_MAP = PASTEL_MAP
    else:
        COLOR_MAP = DEFAULT_COLOR_MAP

    class_counts = {name: 0 for name in class_names}

    for row in detections:
        confidence = row[4]
        if confidence > conf_threshold:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)
            if class_id >= len(class_names):
                continue
            if classes_scores[class_id] > 0.0:
                x_scale = img_cv.shape[1] / INPUT_WIDTH
                y_scale = img_cv.shape[0] / INPUT_HEIGHT
                center_x, center_y, width, height = row[0]*x_scale, row[1]*y_scale, row[2]*x_scale, row[3]*y_scale
                x, y, w, h = int(center_x - width/2), int(center_y - height/2), int(width), int(height)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    if not boxes:
        print("No boxes detected.")
        return img_cv, class_counts

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    print("Boxes before NMS:", len(boxes), "Boxes after NMS:", len(indices))

    if len(indices) > 0:
        indices = indices.flatten()
    else:
        return img_cv, class_counts

    if show_only_parasites:
        indices = [i for i in indices if class_ids[i] in parasite_IDs]

    for i in indices:
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        detected_class_name = class_names[class_id]
        class_counts[detected_class_name] += 1

        if show_boxes:
            color = COLOR_MAP.get(detected_class_name, COLOR_MAP['default'])
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), color, 2)

        if show_labels:
            label = f"{detected_class_name}: {confidences[i]:.2f}"
            cv2.putText(img_cv, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA) 
            cv2.putText(img_cv, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return img_cv, class_counts
