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
# --- File Paths ---
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'best(2).onnx')

# --- Validate Files ---
# Check if files exist
if not os.path.exists(model_path):
    st.error(f"ONNX model not found at: {model_path}")
else:
    st.success("✅ ONNX model loaded successfully.")

# 2. Optimal Thresholds determined from your iterative search
OPTIMAL_CONFIDENCE = 0.001
OPTIMAL_IOU = 0.4        
IMAGE_SIZE = 1280           

# 3. Class names (must match your data.yaml)
CLASS_NAMES = ['red blood cell', 'leukocyte', 'schizont', 'ring', 'gametocyte', 'trophozoite']

# ====================================================================
# B. APPLICATION SETUP
# ====================================================================

# Cache the model loading for fast execution
@st.cache_resource
def load_model():
    """Loads the YOLOv8 model (including ONNX engines) once."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at relative path: {MODEL_PATH}")
        st.warning("Please verify the MODEL_PATH variable and commit 'best.onnx' to your GitHub repo.")
        return None
    
    try:
        # YOLO() handles both .pt and .onnx file types automatically
        model = YOLO(MODEL_PATH)
        st.success("ONNX Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Check if the model file is accessible and properly formatted.")
        return None

st.title("🔬 P. vivax Malaria Parasite Detection (YOLOv8 ONNX)")
st.markdown("---")

# Load the model
model = load_model()

# --- Sidebar for Threshold Control ---
st.sidebar.header("Model Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.01, max_value=1.0, 
    value=OPTIMAL_CONFIDENCE, 
    step=0.01, 
    help=f"Minimum confidence score for a detection. Default: {OPTIMAL_CONFIDENCE}."
)
iou_threshold = st.sidebar.slider(
    "IoU Threshold (NMS)", 
    min_value=0.1, max_value=1.0, 
    value=OPTIMAL_IOU, 
    step=0.05, 
    help=f"IoU threshold for Non-Max Suppression. Default: {OPTIMAL_IOU}."
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
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.subheader("Detection Results")

    # 2. Run prediction
    # NOTE: When using ONNX, the model.predict() runs the inference engine directly.
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
        st.image(im, caption='Parasite Detections', use_column_width=True)

        # Count detected classes
        counts = results[0].boxes.cls.unique(return_counts=True)
        detected_classes = {
            CLASS_NAMES[int(c)]: int(count) 
            for c, count in zip(counts[0], counts[1])
        }

        # Display the clinical findings
        st.markdown("### Clinical Findings Summary")
        
        parasite_stages = ['schizont', 'ring', 'gametocyte', 'trophozoite']
        positive_detections = {k: v for k, v in detected_classes.items() if k in parasite_stages}
        
        if positive_detections:
            st.success("🚨 **Positive Findings: Malaria Parasites Detected**")
            
            # Create a table of parasite counts
            parasite_data = {
                'Parasite Stage': list(positive_detections.keys()),
                'Count': list(positive_detections.values())
            }
            st.dataframe(parasite_data, hide_index=True)
            st.warning(f"Note: Total of {sum(positive_detections.values())} parasite objects detected at this threshold.")
            
        else:
            st.info("No P. vivax parasite stages detected. (Check the confidence threshold if expected.)")

    else:
        st.warning("No objects were detected at the current confidence and IoU thresholds. Try lowering the thresholds in the sidebar.")
