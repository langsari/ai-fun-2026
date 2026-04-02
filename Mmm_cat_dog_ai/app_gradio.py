import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image

MODEL_PATH = "saved_models/ResNet50.keras"
CLASS_NAMES = ["cats", "dogs"]
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

def predict(image):
    if image is None:
        return "No image uploaded.", None

    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr, verbose=0)[0][0])

    if prob >= 0.5:
        pred = CLASS_NAMES[1]
        confidence = prob
    else:
        pred = CLASS_NAMES[0]
        confidence = 1 - prob

    return f"Prediction: {pred} | Confidence: {confidence:.4f}", {
        "cats": float(1 - prob),
        "dogs": float(prob)
    }

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Cat/Dog Image"),
    outputs=[
        gr.Textbox(label="Result"),
        gr.Label(label="Class Probability")
    ],
    title="Cat vs Dog AI Classifier",
    description="Upload an image to classify whether it is a cat or a dog using the best trained model."
)

if __name__ == "__main__":
    demo.launch()
