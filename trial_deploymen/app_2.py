import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
import io

# ====================================================================
# A. CONFIGURATION: UPDATE THESE VALUES FOR ONNX DEPLOYMENT
# ====================================================================

# 1. Path to your ONNX model weights
# CRITICAL: Using os.path.dirname(__file__) makes path handling robust in deployment
base_path = os.path.dirname(__file__)
MODEL_FILENAME = 'best (2).onnx'
model_path = os.path.join(base_path, MODEL_FILENAME)

# 2. Optimal Thresholds determined from your iterative search
# NOTE: Adjusted default slider value to 0.25 (a more typical starting point) 
# The actual value is still set as OPTIMAL_CONFIDENCE for use in the help text.
OPTIMAL_CONFIDENCE = 0.25 # Set a reasonable clinical default (was 0.001)
OPTIMAL_IOU = 0.4        # A common default for mAP50 (was 0.4)
IMAGE_SIZE = 1280         

# 3. Class names (must match your data.yaml)
CLASS_NAMES = ['red blood cell', 'leukocyte', 'schizont', 'ring', 'gametocyte', 'trophozoite']
PARASITE_STAGES = ['schizont', 'ring', 'gametocyte', 'trophozoite']

# ====================================================================
# B. APPLICATION SETUP
# ====================================================================

st.set_page_config(
    page_title="P. vivax Detector",
    page_icon="🔬",
    layout="wide"
)

# Cache the model loading for fast execution
@st.cache_resource
def load_model():
    """Loads the YOLOv8 ONNX model once."""
    if not os.path.exists(model_path):
        st.error(f"Model not found. Please ensure '{MODEL_FILENAME}' is committed to the root directory.")
        return None
    
    try:
        # Load the ONNX model using YOLO()
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading ONNX model: {e}")
        st.info("Check if the model file is accessible, properly formatted, and if system dependencies (packages.txt) are correct.")
        return None

st.title("🔬 P. vivax Malaria Parasite Detection (YOLOv8 ONNX)")
st.markdown(
    """
    This tool uses a fine-tuned YOLOv8 model to rapidly screen blood smear images for *P. vivax* parasite stages.
    Tune the detection thresholds on the left sidebar for optimal clinical sensitivity.
    """
)

# Load the model and display initial status
model = load_model()
if model:
    st.sidebar.success("Model Status: Loaded")
else:
    st.sidebar.error("Model Status: Failed to Load")

# --- Sidebar for Threshold Control ---
st.sidebar.header("Model Settings")

# Use OPTIMAL_CONFIDENCE for the initial value, but ensure it's at least 0.01 
# for the slider's min_value constraint.
initial_conf_value = max(0.01, OPTIMAL_CONFIDENCE)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold (Min Score)", 
    min_value=0.01, max_value=1.0, 
    value=initial_conf_value, 
    step=0.01, 
    help=f"Minimum confidence score for a detection. Recommended for high recall: ~{OPTIMAL_CONFIDENCE:.2f}."
)
iou_threshold = st.sidebar.slider(
    "IoU Threshold (Non-Max Suppression)", 
    min_value=0.1, max_value=1.0, 
    value=OPTIMAL_IOU, 
    step=0.05, 
    help=f"Intersection over Union threshold for filtering overlapping boxes. Default: {OPTIMAL_IOU:.2f}."
)
st.sidebar.markdown(f"**Model Input Size:** {IMAGE_SIZE}x{IMAGE_SIZE}")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload a Blood Smear Image", type=['jpg', 'jpeg', 'png'])

# ====================================================================
# C. INFERENCE LOGIC
# ====================================================================

if uploaded_file is not None and model is not None:
    # 1. Load the image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)
    
    # 2. Run prediction
    with st.spinner('Analyzing blood smear for parasites...'):
        results = model.predict(
            source=image, 
            imgsz=IMAGE_SIZE, 
            conf=conf_threshold, 
            iou=iou_threshold, 
            save=False, 
            verbose=False,
        )

    # 3. Process and Display Results
    if results and results[0].boxes:
        
        # Display the result image with boxes drawn
        im_array = results[0].plot() 
        im = Image.fromarray(im_array[..., ::-1])  
        
        with col2:
            st.subheader("Detection Results")
            st.image(im, caption='Parasite Detections (Conf. > {:.2f})'.format(conf_threshold), use_column_width=True)

        # Count detected classes
        counts = results[0].boxes.cls.unique(return_counts=True)
        detected_classes = {
            CLASS_NAMES[int(c)]: int(count) 
            for c, count in zip(counts[0], counts[1])
        }

        # Display the clinical findings
        st.markdown("---")
        st.markdown("### Clinical Findings Summary")
        
        positive_detections = {k: v for k, v in detected_classes.items() if k in PARASITE_STAGES}
        
        if positive_detections:
            st.success("🚨 **Positive Findings: Malaria Parasites Detected**")
            
            # Create a table of parasite counts
            parasite_data = {
                'Parasite Stage': list(positive_detections.keys()),
                'Count': list(positive_detections.values())
            }
            # Use st.table for clearer display of key results
            st.table(parasite_data)
            
            total_parasites = sum(positive_detections.values())
            st.markdown(f"**Total Parasite Objects Detected:** **{total_parasites}**")
            
            # Additional context for non-parasite detections
            non_parasite_detections = {k: v for k, v in detected_classes.items() if k not in PARASITE_STAGES}
            if non_parasite_detections:
                 st.caption(f"Also detected: {non_parasite_detections.get('red blood cell', 0)} RBCs and {non_parasite_detections.get('leukocyte', 0)} Leukocytes.")

            
        else:
            st.info("No P. vivax parasite stages detected at the current **Confidence Threshold**. ")
            st.caption("If parasites are visible, try lowering the Confidence Threshold in the sidebar to increase sensitivity (Recall).")

    else:
        st.warning("No objects were detected. Try adjusting the threshold settings in the sidebar.")
