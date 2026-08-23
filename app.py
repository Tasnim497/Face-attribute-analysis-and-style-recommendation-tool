import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

from preprocessing import (
    preprocess_face,
    preprocess_eye,
    recommend_haircut,
    recommend_lashes,
    get_gradcam,
    make_gradcam_overlay
)

FACE_MODEL_PATH = "face_efficientnet.keras"
EYE_MODEL_PATH = "eye_efficientnet.keras"

face_classes = ["heart", "oblong", "oval", "round", "square"]
eye_classes = ["almond", "downturned", "hooded", "round", "upturned"]
# Change these if your real class order is different.

@st.cache_resource
def load_models():
    face_model = tf.keras.models.load_model(FACE_MODEL_PATH)
    eye_model = tf.keras.models.load_model(EYE_MODEL_PATH)
    return face_model, eye_model

def predict_one(model, img_rgb, class_names):
    batch = np.expand_dims(img_rgb.astype("float32"), axis=0)
    probs = model(batch, training=False).numpy()[0]
    idx = int(np.argmax(probs))

    return {
        "label": class_names[idx],
        "confidence": float(probs[idx]),
        "probabilities": {
            class_names[i]: float(probs[i]) for i in range(len(class_names))
        }
    }

st.title("Face Shape and Eye Shape Predictor")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Image could not be loaded.")
    else:
        original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        face_in = preprocess_face(img_bgr).astype("float32")
        eye_in = preprocess_eye(img_bgr).astype("float32")

        st.image(original_rgb, caption="Original Uploaded Image", use_container_width=True)

        with st.expander("Show preprocessing outputs"):
            st.image(face_in.astype("uint8"), caption="Face Input Used by Face Model")
            st.image(eye_in.astype("uint8"), caption="Eye Input Used by Eye Model")

        if st.button("Predict"):
            face_model, eye_model = load_models()

            face_res = predict_one(face_model, face_in, face_classes)
            eye_res = predict_one(eye_model, eye_in, eye_classes)

            face_rec = recommend_haircut(face_res["label"])
            eye_rec = recommend_lashes(eye_res["label"])

            st.subheader("Results")

            st.markdown("### Face")
            st.write(f"**Prediction:** {face_res['label'].title()}")
            st.write(f"**Confidence:** {face_res['confidence']:.2f}")
            st.write(f"**Haircuts:** {', '.join(face_rec['haircuts'])}")
            st.write(face_rec["description"])

            st.markdown("---")

            st.markdown("### Eye")
            st.write(f"**Prediction:** {eye_res['label'].title()}")
            st.write(f"**Confidence:** {eye_res['confidence']:.2f}")
            st.write(f"**Lashes:** {', '.join(eye_rec['lash_styles'])}")
            st.write(eye_rec["description"])

            face_tensor = np.expand_dims(face_in.astype("float32"), axis=0)
            face_tensor = np.expand_dims(face_in.astype("float32"), axis=0)
            eye_tensor = np.expand_dims(eye_in.astype("float32"), axis=0)

            face_heatmap = get_gradcam(face_model, face_tensor)
            eye_heatmap = get_gradcam(eye_model, eye_tensor)

            face_overlay = make_gradcam_overlay(face_in, face_heatmap)
            eye_overlay = make_gradcam_overlay(eye_in, eye_heatmap)

            st.subheader("Grad-CAM")
            st.image(face_overlay, caption="Face Model Grad-CAM", use_container_width=True)
            st.image(eye_overlay, caption="Eye Model Grad-CAM", use_container_width=True)