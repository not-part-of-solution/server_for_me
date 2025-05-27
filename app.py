from flask import Flask, request, jsonify
import cv2
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import gdown

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Ограничение файла 5MB

# URL модели
MODEL_URL = 'https://drive.google.com/file/d/1ha9UT3lJgkJLwv1hOAp0HwBd76nXmapT/view?usp=drive_link'
MODEL_PATH = 'cat_model.h5'

# Скачивание модели, если нет
if not os.path.exists(MODEL_PATH):
    file_id = '1ha9UT3lJgkJLwv1hOAp0HwBd76nXmapT'
    gdown.download(f'https://drive.google.com/uc?id={file_id}', MODEL_PATH, quiet=False)

# Параметры
IMAGE_SIZE = (224, 224)
UNKNOWN_THRESHOLD = 0.6

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

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

@app.route("/", methods=["GET"])
def index():
    return "Flask сервер запущен. Используй POST /predict для распознавания изображения."

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400

    file = request.files['image']
    try:
        image = Image.open(BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({'error': 'Invalid image format'}), 400

    image = image.resize(IMAGE_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)[0]
    class_id = int(np.argmax(predictions))
    confidence = float(predictions[class_id])
    label = CLASS_LABELS.get(class_id, 'unknown')

    # Чистим память
    del image, predictions
    import gc
    gc.collect()

    return jsonify({'class': label, 'confidence': confidence})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
