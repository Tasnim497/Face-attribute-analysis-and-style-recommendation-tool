
import os
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np

# -- Global config
IMG_SIZE   = (224, 224)   # uniform for face AND eye -- matches EfficientNet pretrain
BATCH_SIZE = 32
MODEL_DIR  = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def preprocess_face(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]
    scale   = min(1.0, 600.0 / max(h, w))
    small   = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))
    gray    = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    faces   = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

    if len(faces) > 0:
        xs, ys, ws, hs = max(faces, key=lambda box: box[2] * box[3])
        x  = max(0, int(xs / scale))
        y  = max(0, int(ys / scale))
        fw = int(ws / scale)
        fh = int(hs / scale)
        mx, my = int(fw * 0.15), int(fh * 0.20)
        crop = img_rgb[
            max(0, y - my): min(h, y + fh + my),
            max(0, x - mx): min(w, x + fw + mx)
        ]
    else:
        side = min(h, w)
        cy, cx = h // 2, w // 2
        crop = img_rgb[cy - side//2: cy + side//2, cx - side//2: cx + side//2]

    if crop.size == 0:
        crop = img_rgb

    return cv2.resize(crop, IMG_SIZE)


def preprocess_eye(img_bgr):
    """
    Eye preprocessing for FULL FACE images:
    1. Detect face.
    2. Crop upper-middle face area where the eyes usually are.
    3. Resize to IMG_SIZE.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    scale = min(1.0, 600.0 / max(h, w))
    small = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

    if len(faces) > 0:
        xs, ys, ws, hs = max(faces, key=lambda box: box[2] * box[3])
        x = max(0, int(xs / scale))
        y = max(0, int(ys / scale))
        fw = int(ws / scale)
        fh = int(hs / scale)

        eye_y1 = y + int(0.22 * fh)
        eye_y2 = y + int(0.55 * fh)
        eye_x1 = x + int(0.10 * fw)
        eye_x2 = x + int(0.90 * fw)

        crop = img_rgb[
            max(0, eye_y1):min(h, eye_y2),
            max(0, eye_x1):min(w, eye_x2)
        ]
    else:
        eye_y1 = int(0.25 * h)
        eye_y2 = int(0.55 * h)
        eye_x1 = int(0.15 * w)
        eye_x2 = int(0.85 * w)
        crop = img_rgb[eye_y1:eye_y2, eye_x1:eye_x2]

    if crop.size == 0:
        crop = img_rgb

    return cv2.resize(crop, IMG_SIZE)

def recommend_haircut(face_shape):

    face_shape = str(face_shape).strip().lower()

    if face_shape == "oval":
        return {
            "haircuts": ["Pixie Cut", "Bob", "Layers", "Shag"],
            "description": "Balanced face shape — most styles fit well."
        }

    elif face_shape == "round":
        return {
            "haircuts": ["Long Layers", "Side Fringe", "Asymmetry", "Lob"],
            "description": "Goal is to elongate the face."
        }

    elif face_shape == "square":
        return {
            "haircuts": ["Soft Waves", "Curtain Bangs", "Textured Lob"],
            "description": "Goal is to soften strong jawline."
        }

    elif face_shape == "heart":
        return {
            "haircuts": ["Lob", "Side Bangs", "Curtain Bangs", "Soft Layers"],
            "description": "Balance forehead and narrow chin."
        }

    elif face_shape == "oblong":
        return {
            "haircuts": ["Blunt Fringe", "Bob", "Shoulder Length", "Layers"],
            "description": "Goal is to reduce perceived face length."
        }

    else:
        return {
            "haircuts": [],
            "description": "Face shape not recognized."
        }
    
def recommend_lashes(eye_shape):

    eye_shape = str(eye_shape).strip().lower()

    if eye_shape == "round":
        return {
            "lash_styles": ["Cat Eye", "Elongated Cat Eye"],
            "description": "Round eyes benefit from elongation using cat-eye styles."
        }

    elif eye_shape == "hooded":
        return {
            "lash_styles": ["Cat Eye soft", "Open Eye"],
            "description": "Goal is lift and visibility for hooded eyes."
        }

    elif eye_shape == "upturned":
        return {
            "lash_styles": ["Cat Eye", "Hybrid Lashes", "Volume Lashes"],
            "description": "Naturally lifted eyes suit dramatic lash styles."
        }

    elif eye_shape == "downturned":
        return {
            "lash_styles": ["Doll Eye", "Open Eye"],
            "description": "Goal is lifting outer corners visually."
        }

    elif eye_shape == "almond":
        return {
            "lash_styles": ["Natural", "Open Eye", "Cat Eye", "Doll Eye", "Fox Eye"],
            "description": "Most versatile eye shape."
        }

    else:
        return {
            "lash_styles": [],
            "description": "Eye shape not recognized."
        }
    

GRADCAM_LAYER = "top_conv"

def get_gradcam(model, img_tensor, layer_name=GRADCAM_LAYER):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor, training=False)
        pred_idx = tf.argmax(preds[0])
        loss = preds[:, pred_idx]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    mx = tf.reduce_max(heatmap)
    if mx > 0:
        heatmap = heatmap / mx

    return heatmap.numpy()


def make_gradcam_overlay(img_rgb, heatmap, alpha=0.4):
    img_rgb = img_rgb.astype("uint8")

    heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        img_rgb,
        1 - alpha,
        heatmap_color,
        alpha,
        0
    )

    return overlay