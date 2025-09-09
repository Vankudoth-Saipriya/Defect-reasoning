import streamlit as st
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from transformers import pipeline
from PIL import Image
import os

# --- Fix PyTorch UnpicklingError ---
torch.serialization.add_safe_globals([DetectionModel])

# --- Load YOLO Model ---
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file '{MODEL_PATH}' not found in repo.")
    st.stop()

yolo = YOLO(MODEL_PATH)

# --- Load Reasoning Model ---
reasoner = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

# --- Streamlit UI ---
st.title("🔍 Defect Detection & Reasoning")
st.write("Upload an image to detect defects and get reasoning.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # --- YOLO Inference ---
    results = yolo(image)
    detections = results[0].boxes

    # --- Annotated Image ---
    annotated_img = results[0].plot()  # numpy array
    st.image(annotated_img, caption="Detected Defects", use_column_width=True)

    # --- Extract Defects ---
    defect_list = []
    for box in detections:
        cls = int(box.cls.cpu().numpy())
        conf = float(box.conf.cpu().numpy())
        label = yolo.names[cls]
        defect_list.append(f"{label} ({conf:.2f})")

    if defect_list:
        st.subheader("🔍 Detected Defects")
        for defect in defect_list:
            st.write(f"- {defect}")
    else:
        st.warning("No defects detected!")

    # --- Reasoning ---
    defects_text = ", ".join(defect_list)
    reasoning_prompt = f"""
    You are a manufacturing quality inspector.
    Analyze these defects: {defects_text}.
    Provide a structured reasoning:

    - Defect Name:
      - Description:
      - Likely Causes:
      - Prevention Tips:
    """

    st.subheader("🧠 Reasoning")
    try:
        reasoning = reasoner(reasoning_prompt, max_length=300)[0]['generated_text']
        st.write(reasoning)
    except Exception as e:
        st.error(f"Reasoning generation failed: {e}")
