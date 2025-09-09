# ============================================
# Streamlit App: Defect Detection + Reasoning + Bounding Boxes
# ============================================

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# -----------------------------
# 1️⃣ Defect dictionary
# -----------------------------
DEFECT_INFO = {
    "crazing": {
        "Description": "Fine cracks on the surface of the material.",
        "Causes": "Rapid cooling, improper stress relief, or material fatigue.",
        "Prevention": "Controlled cooling, heat treatment, and proper material handling."
    },
    "inclusion": {
        "Description": "Foreign material trapped inside the metal.",
        "Causes": "Contamination during casting or welding.",
        "Prevention": "Clean environment, proper material handling, filtration."
    },
    "patches": {
        "Description": "Surface areas with inconsistent texture or coating.",
        "Causes": "Uneven material application, curing issues, or contamination.",
        "Prevention": "Uniform coating procedures, proper curing, surface preparation."
    }
}

# -----------------------------
# 2️⃣ Load YOLOv8 model
# -----------------------------
yolo = YOLO("best.pt")
 # path to your YOLOv8 weights

# -----------------------------
# 3️⃣ Streamlit Interface
# -----------------------------
st.title(" Manufacturing Defect Detection & Reasoning")
st.write("Upload an image to detect defects, see bounding boxes, and get reasoning.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Read image
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    # Run YOLO detection
    results = yolo(img_np)

    # Draw bounding boxes
    img_boxes = img_np.copy()
    detected_defects = []

    for box in results[0].boxes:
        cls = int(box.cls.cpu().numpy())
        conf = float(box.conf.cpu().numpy())
        label = results[0].names[cls]

        # Save detected defect
        detected_defects.append({"name": label, "confidence": conf})

        # Draw bounding box
        xyxy = box.xyxy.cpu().numpy()[0].astype(int)  # [x1, y1, x2, y2]
        cv2.rectangle(img_boxes, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(img_boxes, f"{label} {conf:.2f}", (xyxy[0], xyxy[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display image with boxes
    st.subheader(" Detected Defects with Bounding Boxes")
    st.image(img_boxes, use_column_width=True)

    # Show reasoning
    st.subheader(" Defect Reasoning")
    if not detected_defects:
        st.success("No defects detected!")
    else:
        for det in detected_defects:
            label = det["name"]
            conf = det["confidence"]
            info = DEFECT_INFO.get(label, {
                "Description": "Unknown",
                "Causes": "Unknown",
                "Prevention": "Unknown"
            })
            st.markdown(f"**Defect:** {label} ({conf:.2f})")
            st.markdown(f"- **Description:** {info['Description']}")
            st.markdown(f"- **Causes:** {info['Causes']}")
            st.markdown(f"- **Prevention:** {info['Prevention']}")
            st.markdown("---")

