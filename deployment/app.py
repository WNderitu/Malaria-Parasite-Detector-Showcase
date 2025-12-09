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
# CRITICAL: This list MUST match the order of classes in your 'classes.txt'
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

# Ensure the loaded class names match the expected number
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
    
    # RESOLUTION CHANGE: Updated to 1280x1280
    INPUT_WIDTH, INPUT_HEIGHT = 1280, 1280
    
    # 1. Convert PIL image to BGR numpy array for OpenCV consistency
    img_cv_rgb = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img_cv_rgb, cv2.COLOR_RGB2BGR)
    
    # 2. Prepare blob for inference
    blob = cv2.dnn.blobFromImage(img_cv, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Add a visual indicator in the console logs for the resolution used
    # print(f"Processing image at {INPUT_WIDTH}x{INPUT_HEIGHT} resolution.") 
    
    preds = net.forward()
    detections = preds[0].T

    boxes, confidences, class_ids = [], [], []
    
# --- UPDATED SECTION FOR PARASITE IDs ---
    # We rely on the globally defined PARASITE_STAGES list and dynamically find their indices
    # based on the order of the loaded class_names list.
    PARASITE_STAGES = ['schizont', 'ring', 'gametocyte', 'trophozoite']
    
    # We only include the ID if the class name is actually in the loaded list.
    parasite_IDs = {class_names.index(cls) for cls in PARASITE_STAGES if cls in class_names}
    
    # For reference, based on the standard list, parasite_IDs should be {2, 3, 4, 5}


    # Color maps (Defined in BGR format for OpenCV drawing)
    # BGR: (B, G, R)
    DEFAULT_COLOR_MAP = {
        'red blood cell': (0, 0, 255),    # RED
        'leukocyte': (255, 255, 255),     # WHITE
        'schizont': (0, 255, 255),        # YELLOW
        'ring': (0, 255, 0),              # GREEN
        'gametocyte': (255, 0, 255),      # MAGENTA/PURPLE
        'trophozoite': (255, 0, 0),       # BLUE
        'default': (128, 128, 128)        # GRAY
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
        return img_cv, class_counts

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
    else:
        return img_cv, class_counts

    if show_only_parasites:
        # This line now uses the correct, dynamically generated parasite_IDs set
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
            # Draw label with black outline and white fill for visibility
            cv2.putText(img_cv, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA) 
            cv2.putText(img_cv, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return img_cv, class_counts

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

if uploaded_files and net and class_names:
    st.subheader(f"Processing {len(uploaded_files)} Images...")
    if st.button("  ▶️  Run detection"):
        progress_bar = st.progress(0)
        total_images = len(uploaded_files)
        results_summary = []

        for i, file in enumerate(uploaded_files):
            image = Image.open(file)
            detected_img_cv, class_counts = process_image(
                net, image, confidence_threshold, nms_threshold, class_names,
                show_boxes, show_labels, show_only_parasites, color_scheme
            )
            detected_img_rgb = cv2.cvtColor(detected_img_cv, cv2.COLOR_BGR2RGB)
            
            col_img, col_data = st.columns([2,1])
            with col_img:
                st.image(detected_img_rgb, caption=f"Processed: {file.name}", use_container_width=True)
            
            with col_data:
                st.markdown(f"#### 🧪 Results for **{file.name}**")
                
                # --- Parasitemia Calculation ---
                total_parasite_count = sum(class_counts.get(stage,0) for stage in PARASITE_STAGES)
                total_detections = sum(class_counts.values())
                parasitemia = (total_parasite_count/total_detections)*100 if total_detections>0 else 0.0
                parasitemia_display = f"{parasitemia:.2f} %"
                
                st.metric("**Total Parasite Count (All Stages)**", total_parasite_count)
                st.metric(
                    label='**Estimated Parasitemia Rate**',
                    value=parasitemia_display,
                    help=("Calculated as: (Total Parasite Detections / Total Cell Detections) * 100. It estimates the proportion of infected cells among all detected cells.")
                )
                st.info(f"**Total Objects Counted:** {total_detections}")

                # Class Count Overview
                st.markdown("##### 🧫 Class Counts per Image")
                cols = st.columns(3)
                all_classes_display = STANDARD_CLASSES 

                for idx, class_name in enumerate(all_classes_display):
                    count = class_counts.get(class_name, 0)
                    with cols[idx % 3]:
                        st.caption(class_name.title())
                        st.markdown(f"**{count}**")

                # Bar Chart
                counts_df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
                if not counts_df.empty:
                    total = counts_df["Count"].sum()
                    counts_df["Percentage"] = (counts_df["Count"] / total) * 100 if total > 0 else 0
                    
                    if chart_mode == 'Counts':
                        x_field = "Count"
                        x_title = "Number of Detections"
                        chart_title = "Detection Counts per Class"
                    else:
                        x_field = "Percentage"
                        x_title = "Detections (%)"
                        chart_title = "Detection % per Class"
                    
                    # Build chart
                    chart = (
                        alt.Chart(counts_df)
                        .mark_bar()
                        .encode(
                            x=alt.X(f"{x_field}:Q", title=x_title),
                            y=alt.Y("Class:N", sort='-x', title="Class Name"),
                            color=alt.Color("Class:N",legend=None),
                            tooltip=[
                                alt.Tooltip('Class:N',title="Class"),
                                alt.Tooltip("Count:Q", title="Count"),
                                alt.Tooltip("Percentage:Q", title="Percentage", format=".2f")
                            ]
                        )
                        .properties(width="container",height=300,title=chart_title)
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("No detections found to visualize.")
                    
                # Append results for CSV
                results_summary.append({
                    "Image": file.name,
                    "Total Parasites": total_parasite_count,
                    "Total Detections": total_detections,
                    "Parasitemia (%)": f"{parasitemia:.2f}",
                    **{f"Count_{cls}": count for cls, count in class_counts.items()}
                })

            st.divider()
            progress_bar.progress((i+1)/total_images)

        progress_bar.empty()
        st.success(" ✅ Detection and quantification complete!")

        # --- CSV Export ---
        if results_summary:
            df_results = pd.DataFrame(results_summary)
            csv_buffer = io.StringIO()
            df_results.to_csv(csv_buffer, index=False)
            st.sidebar.download_button(
                label="📥 Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name="malaria_detection_results.csv",
                mime="text/csv",
                help="Export per-image counts and parasitemia rates."
            )
elif not net:
    st.error(" ❌ ONNX model could not be loaded. Please check the path and file integrity.")
