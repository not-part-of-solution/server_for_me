from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import gdown

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Ограничение 5MB

# Ссылка на модель
MODEL_PATH = 'cat_model.h5'
if not os.path.exists(MODEL_PATH):
    file_id = '1ha9UT3lJgkJLwv1hOAp0HwBd76nXmapT'
    gdown.download(f'https://drive.google.com/uc?id={file_id}', MODEL_PATH, quiet=False)

# Параметры
IMAGE_SIZE = (224, 224)
CLASS_LABELS = {
    0: 'unknown',
    1: 'Авель',
    2: 'Бисмарк',
    3: 'Буся',
    4: 'Винстон',
    5: 'Ева',
    6: 'Изумруд',
    7: 'Муся',
    8: 'Нора'
}

# Загрузка модели
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/", methods=["GET"])
def index():
    return "Flask-сервер запущен. Используй POST /predict для распознавания изображения."

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400

    file = request.files['image']

    try:
        image = Image.open(file.stream).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)[0]
        class_id = int(np.argmax(predictions))
        confidence = float(predictions[class_id])
        label = CLASS_LABELS.get(class_id, 'unknown')

        return jsonify({'class': label, 'confidence': confidence})

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
