🎥 AI Video Detector

Deep Learning Model to Classify AI-Generated vs Real Videos

📌 Project Overview

This project builds a deep learning-based video classification system that determines whether a given video is AI-generated or real.

With the rapid advancement of generative AI tools (e.g., deepfakes, synthetic avatars, AI-rendered footage), detecting artificially generated videos has become increasingly important. This project approaches the problem using:

Convolutional Neural Networks (CNNs)

3D Convolutions for Spatiotemporal Learning

Temporal modeling from video frame sequences

The final model takes a video as input and outputs a binary prediction:

0 → Real Video

1 → AI-Generated Video

🧠 Methodology
1️⃣ Frame Extraction

Videos are first converted into a sequence of frames.

Key Features:

Evenly spaced frame sampling (default: 30 frames per video)

Frame resizing (default: 224×224 resolution)

Efficient OpenCV-based frame extraction

Converts video into a structured tensor suitable for deep learning

This ensures:

Standardized input dimensions

Reduced memory usage

Temporal consistency across videos

2️⃣ Dataset Processing

Videos are organized into two folders:

dataset/
│
├── real/
└── ai/

The dataset processing pipeline:

Iterates through both directories

Extracts frames from each video

Assigns labels:

0 for real

1 for AI-generated

Converts data into NumPy arrays

Splits dataset using train_test_split

This creates:

Training set

Validation set

Properly labeled feature tensors

3️⃣ Model Architecture
🏗 Lightweight 3D CNN

The model uses 3D Convolutions (Conv3D) instead of 2D CNNs.

Why 3D CNN?

Unlike image classification, videos contain:

Spatial information (frame content)

Temporal information (motion across frames)

3D CNNs learn:

Spatial patterns (textures, artifacts)

Temporal inconsistencies (AI flickering, unnatural motion)

Architecture Highlights:

Conv3D layers

MaxPooling3D layers

ReLU activations

Binary classification output (Sigmoid activation)

Optimized for GPU memory efficiency (Kaggle-compatible)

This design balances:

Performance

Speed

Memory efficiency

4️⃣ Data Augmentation

To improve generalization and prevent overfitting, random augmentations are applied:

Random horizontal flipping

Random brightness adjustment

Frame-level transformations

This helps the model:

Avoid memorizing specific videos

Handle variations in lighting and orientation

Generalize better to unseen data

5️⃣ Training Configuration
Optimizer

Adam (learning rate = 0.001)

Loss Function

Binary Crossentropy

Metrics

Accuracy

Precision

Recall

AUC (Area Under ROC Curve)

Callbacks Used

EarlyStopping

Prevents overfitting

ModelCheckpoint

Saves best-performing model

ReduceLROnPlateau

Adjusts learning rate dynamically

TensorBoard

Enables training visualization

📊 Evaluation

The project includes detailed evaluation tools:

📈 Training Visualization

Accuracy curves

Loss curves

📋 Classification Metrics

Precision

Recall

F1-score

AUC score

🔎 Confusion Matrix

Helps analyze:

False Positives (Real predicted as AI)

False Negatives (AI predicted as Real)

📉 ROC Curve

Evaluates classification threshold performance.

💾 Model Saving & Deployment

After training:

model.save('ai_video_detector_final.keras')

The saved model can be reloaded and used for predictions:

loaded_model = load_model('ai_video_detector_final.keras')
Prediction Pipeline
predict_video(video_path)

Steps:

Extract frames

Preprocess frames

Feed into trained model

Return prediction score

🚀 How to Run
1️⃣ Install Dependencies
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn tqdm
2️⃣ Organize Dataset
dataset/
├── ai/
└── real/
3️⃣ Train Model

Run the notebook:

Classifier.ipynb
4️⃣ Make Predictions
predict_video("sample_video.mp4")
📦 Tech Stack

TensorFlow / Keras

NumPy

OpenCV

Scikit-learn

Matplotlib

Seaborn

TQDM

🎯 Key Features

✔ 3D CNN for temporal learning
✔ Memory-efficient architecture
✔ Data augmentation pipeline
✔ Training visualization
✔ Confusion matrix & ROC analysis
✔ Saved deployable model

🧩 Future Improvements

Use pretrained video backbones (e.g., I3D, SlowFast)

Add attention mechanisms

Try Vision Transformers for video

Implement real-time inference pipeline

Deploy via FastAPI or Streamlit

Train on larger deepfake datasets

⚠ Limitations

Performance depends heavily on dataset quality

Requires sufficient GPU memory for large-scale training

May struggle with high-quality AI-generated videos

Binary classification only (AI vs Real)

📌 Conclusion

This project demonstrates a complete deep learning pipeline for detecting AI-generated videos using spatiotemporal modeling. It combines:

Efficient preprocessing

3D CNN-based architecture

Strong evaluation techniques

Deployment-ready saving and prediction pipeline

It serves as a solid foundation for further research in deepfake detection and AI media forensics.
