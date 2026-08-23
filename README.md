<h1 align="center">Face Attribute Analysis and Style Recommendation Tool</h1>

<p align="center">
  A deep learning-based computer vision system for facial attribute analysis and style recommendation.
</p>

<hr>

<h2>Project Overview</h2>

<p>
This project presents a complete computer vision pipeline for face-shape and eye-shape classification.
It covers dataset preparation, image preprocessing, model training, evaluation, Grad-CAM explainability,
and GUI-based prediction.
</p>

<p>
The system accepts a user-provided facial image, preprocesses the relevant regions, and uses two separate
CNN-based models to predict face shape and eye shape. Based on the predictions, the system provides
suitable haircut and eyelash recommendations.
</p>

<p>The project focuses on:</p>

<ul>
  <li>Face Shape Classification</li>
  <li>Eye Shape Classification</li>
  <li>Grad-CAM Explainability</li>
  <li>Haircut and Eyelash Recommendations</li>
</ul>

<hr>

<h2>Datasets</h2>

<h3>1. Face Dataset — <code>finaset</code></h3>

<ul>
  <li>Classification dataset retrieved from Kaggle.</li>
  <li>Used to train the face-shape classification model.</li>
</ul>

<h3>2. Eye Dataset — <code>mergedd_classification_dataset</code></h3>

<p>
Created by converting two eye-detection datasets from Roboflow into classification datasets
and merging them class-wise.
</p>

<p align="center">
  <b>Eye Detection Dataset 1</b>
  &nbsp; + &nbsp;
  <b>Eye Detection Dataset 2</b>
  <br>
  ↓
  <br>
  <b>Classification Conversion</b>
  <br>
  ↓
  <br>
  <b>Class-wise Merging</b>
  <br>
  ↓
  <br>
  <code>mergedd_classification_dataset</code>
</p>

<hr>

<h2>Models</h2>

<p>
Two EfficientNetB0 models pretrained on ImageNet are used for face-shape and eye-shape classification.
</p>

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Architecture</th>
      <th>Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Face Shape Classification</td>
      <td>EfficientNetB0</td>
      <td><code>face_efficientnet.keras</code></td>
    </tr>
    <tr>
      <td>Eye Shape Classification</td>
      <td>EfficientNetB0</td>
      <td><code>eye_efficientnet.keras</code></td>
    </tr>
  </tbody>
</table>

<hr>

<h2>Implementation Pipeline</h2>

<h3>1. Dataset Preparation</h3>

<p>
Two separate classification datasets are used for face shape and eye shape. Each dataset is divided
into training, validation, and testing subsets with class-specific folders.
</p>

<pre>
Dataset
├── train
│   ├── class_1
│   ├── class_2
│   └── ...
├── validation
│   ├── class_1
│   ├── class_2
│   └── ...
└── test
    ├── class_1
    ├── class_2
    └── ...
</pre>

<h3>2. Image Preprocessing</h3>

<p>
OpenCV Haar Cascade is used for face detection. The detected region is then processed according
to the classification task.
</p>

<pre>
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
</pre>

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Input Region</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Face Shape</td>
      <td>Cropped Full Face</td>
      <td>Focuses on jawline, cheeks, chin, and forehead</td>
    </tr>
    <tr>
      <td>Eye Shape</td>
      <td>Cropped Eye Region</td>
      <td>Focuses on eyelids, eye corners, opening, and crease</td>
    </tr>
  </tbody>
</table>

<p>
All images are resized to <b>224 × 224 pixels</b> before being passed to the models.
</p>

<h3>3. TensorFlow Data Pipeline</h3>

<p>
A TensorFlow data pipeline loads images from class folders, assigns numeric labels, resizes images,
converts them to tensors, batches the data, and prefetches batches for efficient training.
</p>

<p>
The same class-label order is maintained during training, evaluation, and GUI prediction.
</p>

<h3>4. Data Augmentation</h3>

<p>
Augmentation is applied only to the training sets to improve generalization.
</p>

<ul>
  <li>Horizontal flipping</li>
  <li>Rotation</li>
  <li>Zooming</li>
  <li>Translation</li>
  <li>Brightness adjustment</li>
  <li>Contrast adjustment</li>
</ul>

<p>
Validation and test images are not augmented to ensure reliable performance evaluation.
</p>

<h3>5. Class Weighting</h3>

<p>
Class weights are calculated from the training-set distribution. Underrepresented classes receive
higher weights to reduce class imbalance and improve balanced learning.
</p>

<h3>6. Model Architecture</h3>

<p>
Two separate CNN-based models are developed using EfficientNetB0 pretrained on ImageNet as the
feature-extraction backbone.
</p>

<p>The custom classification head includes:</p>

<ul>
  <li>Global Average Pooling</li>
  <li>Dense Layers</li>
  <li>Batch Normalization</li>
  <li>ReLU Activation</li>
  <li>Dropout</li>
  <li>Softmax Output Layer</li>
</ul>

<p align="center">
  <b>Input Image</b>
  <br>
  ↓
  <br>
  <b>EfficientNetB0 Backbone</b>
  <br>
  ↓
  <br>
  <b>Global Average Pooling</b>
  <br>
  ↓
  <br>
  <b>Dense + Batch Normalization + ReLU + Dropout</b>
  <br>
  ↓
  <br>
  <b>Softmax Classification</b>
</p>

<h3>7. Training Strategy</h3>

<p>
Training is performed in three stages to gradually adapt the pretrained network to the project-specific
classification tasks.
</p>

<table>
  <thead>
    <tr>
      <th>Stage</th>
      <th>Training Strategy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Stage 1</td>
      <td>EfficientNetB0 backbone frozen; classification head trained.</td>
    </tr>
    <tr>
      <td>Stage 2</td>
      <td>Top backbone layers unfrozen and fine-tuned with a smaller learning rate.</td>
    </tr>
    <tr>
      <td>Stage 3</td>
      <td>Additional layers fine-tuned using an even smaller learning rate.</td>
    </tr>
  </tbody>
</table>

<p>
The training process uses model checkpointing, early stopping, and learning-rate reduction
to preserve the best model and improve training stability.
</p>

<h3>8. Model Evaluation</h3>

<p>
The trained models are evaluated on the test sets using both standard prediction and
test-time augmentation (TTA).
</p>

<ul>
  <li>Accuracy</li>
  <li>Macro F1-Score</li>
  <li>Classification Report</li>
  <li>Confusion Matrix</li>
</ul>

<p>
For TTA, multiple augmented versions of each test image are predicted and their probabilities
are averaged to obtain the final prediction.
</p>

<h3>9. Explainability with Grad-CAM</h3>

<p>
Grad-CAM is used to visualize the image regions that contribute most to the model's predictions.
</p>

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Expected Focus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Face Shape Model</td>
      <td>Face outline, jawline, cheeks, forehead, and chin</td>
    </tr>
    <tr>
      <td>Eye Shape Model</td>
      <td>Eyelids, eye corners, eye opening, and crease</td>
    </tr>
  </tbody>
</table>

<p>
This helps assess whether the models are learning relevant facial features rather than
irrelevant background or lighting information.
</p>

<h3>10. Model Saving and GUI Integration</h3>

<p>
The best-performing models are saved in <code>.keras</code> format and loaded by the Streamlit GUI.
</p>

<pre>
FACE_MODEL_PATH = "face_efficientnet.keras"
EYE_MODEL_PATH = "eye_efficientnet.keras"
</pre>

<p>
The GUI allows users to upload an unseen facial image. The same preprocessing pipeline used during
training is applied before prediction.
</p>

<p>
Preprocessing and recommendation functions are stored in <code>preprocessing.py</code> and imported
into the main GUI application.
</p>

<pre>
from preprocessing import (
    preprocess_face,
    preprocess_eye,
    recommend_haircut,
    recommend_lashes,
    get_gradcam,
    make_gradcam_overlay
)
</pre>

<p>
The predicted face and eye shapes are then mapped to suitable haircut and eyelash recommendations.
</p>

<hr>

<h2>System Workflow</h2>

<p align="center">
  <b>User Image</b>
  <br>
  ↓
  <br>
  <b>Face Detection</b>
  <br>
  ↓
  <br>
  ┌─────────────────────────────┐
  <br>
  <b>Face Region</b> &nbsp;&nbsp;&nbsp;&nbsp; <b>Eye Region</b>
  <br>
  ↓ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ↓
  <br>
  <b>Face Model</b> &nbsp;&nbsp;&nbsp;&nbsp; <b>Eye Model</b>
  <br>
  ↓ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ↓
  <br>
  <b>Face Shape</b> &nbsp;&nbsp;&nbsp;&nbsp; <b>Eye Shape</b>
  <br>
  ↓ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ↓
  <br>
  └──────────────┬──────────────┘
  <br>
  ↓
  <br>
  <b>Style Recommendations</b>
</p>

<hr>

<h2>Project Structure</h2>

<pre>
Face-Attribute-Analysis-and-Style-Recommendation-Tool/
│
├── app.py
├── preprocessing.py
├── face_eye_classifier.ipynb
│
├── models/
│   ├── face_efficientnet.keras
│   └── eye_efficientnet.keras
│
├── datasets/
│   ├── finaset/
│   └── mergedd_classification_dataset/
│
└── README.md
</pre>

<p>
Large datasets and model weights are stored externally due to GitHub file-size limitations.
</p>

<hr>

<h2>Datasets and Model Weights</h2>

<p>
The complete datasets and trained model weights are available through the following Google Drive folder:
</p>

<p align="center">
  <a href="https://drive.google.com/drive/folders/1ucjoWXbGEnMUNXvXbB9lb7Wd822pZZCx?usp=sharing">
    Google Drive — Datasets and Model Weights
  </a>
</p>

<table>
  <thead>
    <tr>
      <th>File / Folder</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>finaset</code></td>
      <td>Face-shape classification dataset from kaggle</td>
    </tr>
    <tr>
      <td><code>mergedd_classification_dataset</code></td>
      <td>Merged eye-shape classification dataset from Roboflow</td>
    </tr>
    <tr>
      <td><code>face_efficientnet.keras</code></td>
      <td>Trained face-shape model</td>
    </tr>
    <tr>
      <td><code>eye_efficientnet.keras</code></td>
      <td>Trained eye-shape model</td>
    </tr>
  </tbody>
</table>

<hr>

<h2>Technologies Used</h2>

<table>
  <tr>
    <td><b>Programming</b></td>
    <td>Python</td>
  </tr>
  <tr>
    <td><b>Deep Learning</b></td>
    <td>TensorFlow, Keras, EfficientNetB0</td>
  </tr>
  <tr>
    <td><b>Computer Vision</b></td>
    <td>OpenCV, Haar Cascade</td>
  </tr>
  <tr>
    <td><b>Data Processing</b></td>
    <td>NumPy, Pandas</td>
  </tr>
  <tr>
    <td><b>Evaluation</b></td>
    <td>Scikit-learn, Matplotlib</td>
  </tr>
  <tr>
    <td><b>Explainability</b></td>
    <td>Grad-CAM</td>
  </tr>
  <tr>
    <td><b>GUI</b></td>
    <td>Streamlit</td>
  </tr>
</table>

<hr>

<hr>

<h2>Results and Evaluation</h2>

<h3>Face Shape Classifier</h3>

<ul>
  <li><b>Accuracy:</b> 0.73</li>
  <li><b>F1 Score:</b> 0.73</li>
</ul>

<p>
The face-shape classifier achieved an accuracy and F1 score of <b>0.73</b>
on the test set.
</p>


<h3>Eye Shape Classifier</h3>

<ul>
  <li><b>Accuracy:</b> 0.425</li>
  <li><b>F1 Score:</b> 0.416</li>
</ul>

<p>
The eye-shape classifier achieved an accuracy of <b>0.425</b> and an F1 score
of <b>0.416</b>. The relatively lower performance is expected due to the
considerably small dataset. In addition, eye-shape classification can be
challenging because some eyes may share characteristics of multiple classes.
For example, an eye may appear both almond and upturned, or hooded and round,
making the classification task more challenging for the model.
</p>

<h2>References</h2>

<p>
The following sources were used for the datasets and for developing the haircut
and eyelash recommendation rules:
</p>

<p>
  <b> Face Shape Dataset:</b><br>
  <a href="https://www.kaggle.com/datasets/niten19/face-shape-dataset">
    https://www.kaggle.com/datasets/niten19/face-shape-dataset
  </a>
</p>

<p>
  <b>Eye Shape Datasets:</b><br>

  <a href="https://universe.roboflow.com/data-vm3yw/eye-shape-rgpxt">
    https://universe.roboflow.com/data-vm3yw/eye-shape-rgpxt
  </a><br>

  <a href="https://universe.roboflow.com/test1-sjwxa/eye-shape-2">
    https://universe.roboflow.com/test1-sjwxa/eye-shape-2
  </a><br>

  <a href="https://universe.roboflow.com/eyeshapes/eye-shapes/dataset/2">
    https://universe.roboflow.com/eyeshapes/eye-shapes/dataset/2
  </a>
</p>

<p>

<p>
  <b> Haircut Recommendations:</b><br>
  <a href="https://kenarissalon.com/blog/f/women%E2%80%99s-guide-to-choosing-the-best-haircuts-based-on-face-shape">
    https://kenarissalon.com/blog/f/women%E2%80%99s-guide-to-choosing-the-best-haircuts-based-on-face-shape
  </a>
</p>

<p>
  <b> Eyelash Recommendations:</b><br>
  <a href="https://belolash.com/blogs/news/what-are-the-best-lash-styles-for-different-eye-shapes">
    https://belolash.com/blogs/news/what-are-the-best-lash-styles-for-different-eye-shapes
  </a>
</p>

<h3>Datasets and Model Weights</h3>

<p>
The complete datasets and trained model weights used in this project are available
in the following Google Drive folder:
</p>

<p>
  <a href="https://drive.google.com/drive/folders/1ucjoWXbGEnMUNXvXbB9lb7Wd822pZZCx?usp=sharing">
    Google Drive — Datasets and Model Weights
  </a>
</p>
