"""
app.py

Streamlit interface: upload a QR image -> decode -> run both branches ->
show fused Low/Medium/High risk verdict with reasons and a Grad-CAM heatmap.

Run (from project root):
    streamlit run app.py
"""

import os
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st
from pyzbar.pyzbar import decode as zbar_decode
from PIL import Image
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "branch_a_image"))
sys.path.append(os.path.join(os.path.dirname(__file__), "branch_b_payload"))
sys.path.append(os.path.join(os.path.dirname(__file__), "fusion"))

try:
    import tensorflow as tf
    from gradcam import predict_with_gradcam                       # noqa: E402
    HAVE_TF = True
except ImportError:
    HAVE_TF = False
    
from payload_pipeline import analyze_payload                   # noqa: E402
from fusion import fuse                                        # noqa: E402

CNN_MODEL_PATH = os.path.join(os.path.dirname(__file__), "branch_a_image", "qr_cnn_model.keras")

st.set_page_config(page_title="QR Scam Detector", page_icon="🔍", layout="centered")
st.title("🔍 AI-Based QR Code Scam Detection")
st.caption("Combines CNN-based image tampering detection with payload risk analysis.")


@st.cache_resource
def load_cnn_model():
    if HAVE_TF and os.path.exists(CNN_MODEL_PATH):
        return tf.keras.models.load_model(CNN_MODEL_PATH)
    return None


def decode_qr(image_path: str):
    img = cv2.imread(image_path)
    decoded = zbar_decode(img)
    if not decoded:
        return None
    return decoded[0].data.decode("utf-8", errors="ignore")


source = st.radio("Input method", ["Upload image", "Use webcam"])
image_path = None

if source == "Upload image":
    uploaded = st.file_uploader("Upload a QR code image", type=["png", "jpg", "jpeg"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(uploaded.read())
            image_path = tmp.name
else:
    cam_image = st.camera_input("Scan a QR code")
    if cam_image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(cam_image.getvalue())
            image_path = tmp.name

if image_path:
    st.image(image_path, caption="Input image", width=300)
    payload = decode_qr(image_path)

    if payload is None:
        st.error("Could not decode a QR code in this image.")
    else:
        st.write(f"**Decoded payload:** `{payload}`")

        with st.spinner("Analyzing..."):
            payload_result = analyze_payload(payload)

            cnn_model = load_cnn_model()
            if cnn_model is not None:
                image_risk_score, heatmap_overlay = predict_with_gradcam(cnn_model, image_path)
            else:
                image_risk_score, heatmap_overlay = 0.0, None
                st.warning("CNN model not found - train it first (see README). Using image_risk_score=0.")

            result = fuse(image_risk_score, payload_result)

        st.divider()
        band_color = {"Low": "green", "Medium": "orange", "High": "red"}[result["risk_band"]]
        st.markdown(f"## Risk level: :{band_color}[{result['risk_band']}]")
        st.progress(result["final_score"])
        st.write(f"Final risk score: **{result['final_score']}**  |  "
                 f"Payload type: **{result['payload_type']}**  |  "
                 f"Image score: **{result['image_risk_score']}**  |  "
                 f"Payload score: **{result['payload_risk_score']}**")

        st.subheader("Why this verdict:")
        for r in result["reasons"]:
            st.write(f"- {r}")

        if heatmap_overlay is not None:
            st.subheader("Image anomaly heatmap (Grad-CAM)")
            st.image(heatmap_overlay, caption="Warmer colors = regions driving the tampering prediction", width=300)
