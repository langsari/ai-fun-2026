import sys
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
CLASS_NAMES = ["cats", "dogs"]

# โหลดทั้ง 3 โมเดล
model1 = tf.keras.models.load_model("saved_models/SimpleCNN.keras")
model2 = tf.keras.models.load_model("saved_models/MobileNetV2.keras")
model3 = tf.keras.models.load_model("saved_models/ResNet50.keras")

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(image_path):
    img = load_image(image_path)

    p1 = model1.predict(img, verbose=0)[0][0]
    p2 = model2.predict(img, verbose=0)[0][0]
    p3 = model3.predict(img, verbose=0)[0][0]

    # เฉลี่ย
    avg = (p1*0.1 + p2*0.2 + p3*0.7)

    if avg >= 0.5:
        label = "dogs"
        confidence = avg
    else:
        label = "cats"
        confidence = 1 - avg

    return label, confidence, (p1, p2, p3)

if __name__ == "__main__":
    path = sys.argv[1]

    label, conf, probs = predict(path)

    print(f"Prediction: {label}")
    print(f"Confidence: {conf:.4f}")
    print(f"SimpleCNN: {probs[0]:.4f}")
    print(f"MobileNetV2: {probs[1]:.4f}")
    print(f"ResNet50: {probs[2]:.4f}")