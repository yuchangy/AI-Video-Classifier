🎥 AI Video Detector
Deep Learning Model to Classify AI-Generated vs Real Videos
📌 Project Overview

This project builds a deep learning–based video classification system that determines whether a video is AI-generated or authentic. With rapid advancements in generative AI and deepfake technologies, distinguishing synthetic media from real footage has become increasingly challenging. This project addresses that problem by designing a spatiotemporal model capable of identifying subtle visual artifacts and motion inconsistencies that commonly appear in AI-generated videos.

The final model performs binary classification:

0 → Real Video

1 → AI-Generated Video

🧠 Methodology

The pipeline begins by extracting evenly spaced frames from each video to create a consistent temporal representation. Each video is converted into a fixed-length tensor of resized frames, ensuring standardized inputs and efficient memory usage during training. The dataset is organized into two labeled categories (real and AI-generated), then processed into NumPy arrays and split into training and validation sets.

To effectively capture both spatial and temporal patterns, the model uses 3D Convolutional Neural Networks (Conv3D) instead of traditional 2D CNNs. While 2D CNNs analyze individual frames independently, 3D CNNs learn motion patterns across consecutive frames, allowing the model to detect:

Temporal inconsistencies in movement

Flickering or unnatural transitions

Texture irregularities and generation artifacts

This design enables the model to analyze videos holistically rather than as isolated images.

🏗 Model Architecture

The architecture consists of stacked Conv3D and MaxPooling3D layers with ReLU activations, followed by dense layers and a sigmoid output for binary classification. The model is trained using the Adam optimizer with binary crossentropy loss.

To improve generalization and stability, training incorporates:

Early stopping to prevent overfitting

Learning rate scheduling

Model checkpointing to save the best-performing weights

Performance is evaluated using accuracy, precision, recall, and AUC to ensure balanced assessment beyond simple accuracy.

📊 Evaluation

Model performance is analyzed using training and validation accuracy/loss curves to monitor convergence. A confusion matrix provides insight into false positives and false negatives, while an ROC curve evaluates classification performance across probability thresholds. These metrics help measure how effectively the model distinguishes between real and AI-generated videos.

💾 Deployment & Usage

After training, the model is saved in a deployable format and can be reloaded for inference. The prediction pipeline:

Extracts frames from a new video

Preprocesses frames into the required tensor format

Generates a probability score using the trained model

This structure allows the detector to be integrated into media verification tools, moderation systems, or research applications.

🛠 Tech Stack

TensorFlow / Keras

OpenCV

NumPy

Scikit-learn

Matplotlib & Seaborn

🎯 Conclusion

This project demonstrates a complete end-to-end deep learning pipeline for detecting AI-generated video content. By combining structured preprocessing, spatiotemporal modeling with 3D CNNs, rigorous evaluation, and deployment-ready inference, it provides a practical and scalable foundation for research in deepfake detection and AI media forensics.
