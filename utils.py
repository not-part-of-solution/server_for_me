import cv2
import numpy as np

IMAGE_SIZE = (224, 224)

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0  # нормализация
    return np.expand_dims(img, axis=0)
