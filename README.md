# Facial Expression Recognizer

A lightweight project for recognising facial expressions using a convolutional neural network (CNN) and a demo web application.  
This repository contains model definition and training logic, utility scripts, a demo Flask web-app, plus templates / static assets.

---

## Project Structure

- `emotion_cnn.py` – defines the CNN model and includes training / prediction logic.  
- `utils.py` – helper functions for loading data, preprocessing images, building batches etc.  
- `app.py` – Flask web application that serves the demo UI, allows image upload / webcam input and shows predicted expression.  
- `templates/` & `static/` – HTML templates and static files (CSS/JS/images) for the web interface.  
- `models/` – a directory for saving trained model weights (currently included as empty or placeholder).  
- `requirements.txt` – Python dependencies needed for the project.  
- `__pycache__/` – compiled Python files (ignored in version control).  

---

## Quick Start

### 1. Clone the repository  
```bash
git clone https://github.com/Sameershahh/Facial_Expression_Recognizer.git
cd Facial_Expression_Recognizer
```
### 2. Set up a virtual environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate        # on Mac/Linux
# .venv\Scripts\activate         # on Windows
pip install -r requirements.txt
```

### 3. Prepare your dataset 
This project expects a dataset of facial-expression-labelled images. For example, you can use a dataset like FER2013 (downloadable from Kaggle) or any custom dataset arranged in a folder-per-class structure.
A suggested folder structure:
```bash
data/
  train/
    happy/
    sad/
    angry/
    neutral/
    [other_classes]/
  val/
    happy/
    sad/
    angry/
    neutral/
    [other_classes]/
```

### 4. Train the model
```bash
python emotion_cnn.py --mode train --data_dir data/train --val_dir data/val --epochs 30 --batch_size 32 --save_dir models/
```

### 5. Run the demo 
```bash
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000
```

### 6. Perform a prediction on a single image 
```bash
python app.py --predict --image path/to/image.jpg --model models/best_model.h5
```

##  Model & Architecture

The CNN model defined in `emotion_cnn.py` uses **convolutional layers**, **pooling**, **dropout**, and a **final softmax layer** to classify facial expressions into multiple classes.

### Model Details
- **Number of Classes:** *N (depends on dataset, e.g., 7 for FER2013 – Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)*  
- **Input Image Size:** *48×48 grayscale* (or adjust if your dataset differs)  
- **Architecture Summary:**
  - Multiple convolutional blocks (Conv2D → ReLU → MaxPooling)
  - Dropout layers to prevent overfitting
  - Fully-connected (Dense) layers
  - Final Softmax layer for classification
- **Training Details:**
  - Optimizer: *Adam*
  - Learning Rate: *1e-3 (default, adjustable)*
  - Loss Function: *Categorical Crossentropy*
  - Epochs: *Typically 25–50*
  - Batch Size: *32*
  - Framework: *TensorFlow / Keras*

If you already have saved model weights inside `models/`, include them in the repository or provide a download link for convenience.

---

##  Evaluation

Evaluate the trained model using a **separate validation/test dataset** to measure generalization.

### Recommended Evaluation Steps
- Compute **accuracy per class**
- Generate a **confusion matrix**
- *(Optional)* Plot **ROC** or **Precision-Recall** curves for each class

You can extend `emotion_cnn.py` by adding a `--mode evaluate` argument to handle evaluation and automatically save metrics.

---

##  Usage Notes

- The web demo (`app.py`) expects a **detected face** in the uploaded or webcam image.  
  If no face is detected, prediction results may be inaccurate.
- For **real-time performance** (webcam), resize frames and limit FPS.
- Model performance depends on dataset quality and diversity; extreme lighting or poses may reduce accuracy.

---

##  Requirements & Environment

- **Python:** 3.8 or newer  
- **Dependencies:** listed in `requirements.txt`
- **GPU Support:** Recommended for faster training (especially for large datasets)
- For full reproducibility:
  ```bash
  pip freeze > requirements.txt

