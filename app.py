from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from emotion_cnn import EmotionCNN
from utils import detect_face
import base64
import io

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load model
model = EmotionCNN()
model.load_state_dict(torch.load('models/emotion_model.pth', map_location=torch.device('cpu')))
model.eval()

# Transforms (same as training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if image upload or webcam
        if 'image' in request.files:
            file = request.files["image"]
            if file.filename == "":
                return render_template("index.html", error="No selected file")

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], "output.jpg")
            file.save(filepath)

        elif 'webcam_image' in request.form:
            webcam_data = request.form['webcam_image']
            if webcam_data.startswith('data:image'):
                header, encoded = webcam_data.split(",", 1)
                decoded = base64.b64decode(encoded)
                image = Image.open(io.BytesIO(decoded)).convert("RGB")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], "output.jpg")
                image.save(filepath)
            else:
                return render_template("index.html", error="Invalid webcam image data")

        else:
            return render_template("index.html", error="No input received")

        # Preprocess the image
        image = Image.open(filepath).convert("RGB")
        face = detect_face(np.array(image))

        if face is None:
            return render_template("index.html", error="No face detected")

        face_pil = Image.fromarray(face)
        input_tensor = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            emotion = emotion_labels[predicted.item()]

        return render_template("index.html", emotion=emotion, image_url="static/output.jpg")

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)