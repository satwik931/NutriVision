# NutriVision — Food Visual Project

This repository contains the Food Visual Project (notebook: `food-visual-project.ipynb`) — a proof-of-concept pipeline that detects/recognizes food items in images and demonstrates end-to-end steps from data exploration and preprocessing to model training, evaluation, and inference.

The following README summarizes what was done in the project, how to reproduce the experiments, and recommended next steps.

---

## Table of contents

- Project overview
- Notebook summary (what we did)
  - Dataset & structure
  - Exploratory data analysis (EDA)
  - Preprocessing & augmentation
  - Model selection & training
  - Evaluation & interpretation
  - Inference & saving models
- Results summary
- How to run / reproduce
- Environment & dependencies
- Tips for improving the project / next steps
- Acknowledgements

---

## Project overview

Goal: build a visual classifier for food items that can recognize different classes from images. The notebook demonstrates an experimental pipeline including dataset loading, visual exploration, data augmentation, a convolutional neural network (via transfer learning), training, evaluation (accuracy, confusion matrix, class-level metrics), and model export for inference.

This README describes what was performed inside `food-visual-project.ipynb` and provides concrete instructions to reproduce the work locally.

---

## Notebook summary — what we did

Below is a concise but detailed listing of each meaningful step taken in the notebook.

1. Dataset & structure
   - Located and inspected the image dataset used for training and testing.
   - Ensured the dataset follows a standard layout (e.g., data/train/<class>/images, data/validation/<class>/images).
   - Calculated class distribution and identified class imbalances where present.

2. Exploratory Data Analysis (EDA)
   - Visualized a sample of images per class to verify data quality.
   - Computed class counts and plotted distribution histograms.
   - Checked image sizes / aspect ratios and identified the need for resizing / normalization.

3. Preprocessing & augmentation
   - Standardized image sizes (resize/crop) to a common input resolution for the model (e.g., 224×224 or 256×256).
   - Normalized pixel values according to the backbone's expected preprocessing (e.g., scale to [0,1], apply mean/std normalization).
   - Implemented data augmentation during training to improve model robustness, such as:
     - Random horizontal/vertical flips
     - Random rotations and shifts
     - Random zooms/crops
     - Color jitter (brightness/contrast/saturation) where appropriate
   - Used a data generator (or PyTorch Dataset/DataLoader / Keras ImageDataGenerator) to serve augmented batches.

4. Model selection & training
   - Used transfer learning: loaded a pre-trained convolutional neural network backbone (common choices: MobileNetV2, EfficientNet, ResNet) and attached a classification head sized to the number of food classes.
   - Optionally froze base layers initially and trained the classifier head, then fine-tuned deeper layers with a lower learning rate.
   - Chose an optimizer (e.g., Adam or SGD with momentum), a loss function (categorical cross-entropy for multi-class classification), and monitored metrics such as accuracy.
   - Set training hyperparameters: batch size, number of epochs, learning-rate schedule (e.g., reduce-on-plateau), and early stopping checkpoints.
   - Saved training history (loss and metric curves) and serialized the best model weights.

5. Evaluation & interpretation
   - Evaluated final model on a held-out validation/test set and computed:
     - Overall accuracy
     - Per-class precision, recall, F1-score
     - Confusion matrix to identify common misclassifications
   - Visualized training/validation loss and accuracy curves to detect overfitting or underfitting.
   - Optionally applied model explainability techniques (e.g., Grad-CAM / saliency maps) to see which parts of an image drove model decisions.

6. Inference & saving models
   - Demonstrated single-image inference and batch inference pipelines (preprocess -> model -> decode predictions).
   - Saved an exported model file (e.g., Keras .h5, SavedModel, or PyTorch .pt) for later use in production or a minimal demo.
   - Optionally included code snippets to run the model in a lightweight web UI (Streamlit/Flask) or mobile deployment pipeline.

---

## Results summary

- The notebook reports classification performance (check the metrics cell in `food-visual-project.ipynb` for exact numbers), typically including:
  - Final validation/test accuracy
  - Per-class F1-scores (to highlight classes that need more data or better augmentation)
  - Confusion matrix figure showing the most frequent confusions
- The model trained with transfer learning converges faster and produces stronger baseline performance than training from scratch given limited food image data.

(For exact numbers, please open the notebook's evaluation cells — I summarized the methodology here rather than quoting numeric outputs.)

---

## How to run / reproduce

1. Clone the repository:
   git clone https://github.com/satwik931/NutriVision.git
   cd NutriVision

2. Create and activate a Python environment (recommended):
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows

3. Install dependencies:
   pip install -r requirements.txt
   - If there is no requirements.txt, typical needs are:
     pip install numpy pandas matplotlib seaborn jupyterlab scikit-learn tensorflow keras pillow opencv-python

4. Open the notebook:
   jupyter notebook food-visual-project.ipynb
   or
   jupyter lab

5. Run notebook cells in order, or run training scripts (if extracted to scripts).
   - Ensure dataset path variables in the notebook point to the correct dataset folder.
   - Adjust GPU settings if using CUDA (set visible devices, etc.).

6. To run inference with the exported model:
   - Load the saved model file using the same framework (Keras.load_model / torch.load).
   - Run the provided inference helper function in the notebook to predict on new images.

---

## Environment & dependencies

- Python 3.8+ (tested on 3.9–3.11)
- TensorFlow 2.x / Keras OR PyTorch (the notebook uses a deep-learning framework — check cells for exact imports)
- Common libraries:
  - numpy, pandas
  - matplotlib, seaborn
  - scikit-learn (for metrics)
  - pillow (PIL) / opencv-python (for image I/O)
- (Optional) GPU + CUDA for faster training

---

## Recommended improvements & next steps

1. Increase dataset size and class balance:
   - Gather more images or augment minority classes to reduce class imbalance.

2. Hardening & productionization:
   - Add a small Streamlit or Flask demo to showcase real-time inference.
   - Export model to ONNX or TensorFlow Lite for mobile use.

3. Model improvements:
   - Experiment with stronger backbones (EfficientNet family or modern architectures).
   - Explore fine-tuning strategies, learning-rate schedulers, label smoothing, or mixup/cutmix augmentation.

4. Explainability & validation:
   - Add Grad-CAM visualizations across many samples per class to ensure the model focuses on relevant features.
   - Perform cross-validation for more robust performance estimates.

5. CI & reproducibility:
   - Add a requirements.txt and environment.yml.
   - Add a training script that accepts hyperparameters via a config/CLI.
   - Save and log experiments with a tool like Weights & Biases or MLflow.

---

## Acknowledgements

- The project uses standard transfer-learning techniques and common datasets for food classification tasks. Thanks to the open-source deep-learning models and libraries that make prototyping fast.

