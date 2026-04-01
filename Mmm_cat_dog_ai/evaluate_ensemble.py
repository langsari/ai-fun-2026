import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TEST_DIR = "dataset/test"
RESULT_DIR = "results"
MODEL_DIR = "saved_models"
CLASS_NAMES = ["cats", "dogs"]

os.makedirs(RESULT_DIR, exist_ok=True)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE
test_ds_plain = test_ds.prefetch(AUTOTUNE)
test_ds_mobilenet = test_ds.map(
    lambda x, y: (mobilenet_preprocess(tf.cast(x, tf.float32)), y)
).prefetch(AUTOTUNE)
test_ds_resnet = test_ds.map(
    lambda x, y: (resnet_preprocess(tf.cast(x, tf.float32)), y)
).prefetch(AUTOTUNE)

simple_model = tf.keras.models.load_model(f"{MODEL_DIR}/SimpleCNN.keras")
mobilenet_model = tf.keras.models.load_model(f"{MODEL_DIR}/MobileNetV2.keras")
resnet_model = tf.keras.models.load_model(f"{MODEL_DIR}/ResNet50.keras")

def collect_preds(model, dataset):
    y_true, y_prob = [], []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0).flatten()
        y_prob.extend(preds.tolist())
        y_true.extend(labels.numpy().astype("int32").tolist())
    return np.array(y_true), np.array(y_prob)

y_true_1, p1 = collect_preds(simple_model, test_ds_plain)
y_true_2, p2 = collect_preds(mobilenet_model, test_ds_mobilenet)
y_true_3, p3 = collect_preds(resnet_model, test_ds_resnet)

assert np.array_equal(y_true_1, y_true_2) and np.array_equal(y_true_2, y_true_3), "Label order mismatch"
y_true = y_true_1

ensemble_prob = (p1 + p2 + p3) / 3.0
ensemble_pred = (ensemble_prob > 0.5).astype("int32")

report = classification_report(
    y_true,
    ensemble_pred,
    target_names=CLASS_NAMES,
    output_dict=True
)
cm = confusion_matrix(y_true, ensemble_pred)

summary = {
    "model": "EnsembleAvg",
    "accuracy": float(report["accuracy"]),
    "precision": float(report["weighted avg"]["precision"]),
    "recall": float(report["weighted avg"]["recall"]),
    "f1_score": float(report["weighted avg"]["f1-score"])
}

with open(f"{RESULT_DIR}/ensemble_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

with open(f"{RESULT_DIR}/ensemble_classification_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

np.savetxt(f"{RESULT_DIR}/ensemble_confusion_matrix.csv", cm, fmt="%d", delimiter=",")

print("Ensemble result:")
print(summary)
