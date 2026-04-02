import os
import json
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_CNN = 5
EPOCHS_TRANSFER = 3

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

RESULT_DIR = "results"
MODEL_DIR = "saved_models"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = train_ds.class_names
print("Class names:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds_plain = train_ds.prefetch(AUTOTUNE)
val_ds_plain = val_ds.prefetch(AUTOTUNE)
test_ds_plain = test_ds.prefetch(AUTOTUNE)

# -----------------------------
# Data augmentation
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# -----------------------------
# Model 1: Simple CNN
# -----------------------------
def build_simple_cnn():
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Rescaling(1./255),
        data_augmentation,

        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# -----------------------------
# Model 2: MobileNetV2
# -----------------------------
def build_mobilenet():
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = mobilenet_preprocess(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# -----------------------------
# Model 3: ResNet50
# -----------------------------
def build_resnet():
    base_model = ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = resnet_preprocess(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# -----------------------------
# Plot history
# -----------------------------
def plot_history(history, model_name):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/{model_name}_accuracy.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/{model_name}_loss.png")
    plt.close()

# -----------------------------
# Evaluate model
# -----------------------------
def evaluate_model(model, test_dataset, model_name):
    y_true = []
    y_pred = []

    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        preds = (preds > 0.5).astype("int32").flatten()

        y_true.extend(labels.numpy().astype("int32"))
        y_pred.extend(preds)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True
    )

    cm = confusion_matrix(y_true, y_pred)

    with open(f"{RESULT_DIR}/{model_name}_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    np.savetxt(f"{RESULT_DIR}/{model_name}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    return {
        "model": model_name,
        "accuracy": float(report["accuracy"]),
        "precision": float(report["weighted avg"]["precision"]),
        "recall": float(report["weighted avg"]["recall"]),
        "f1_score": float(report["weighted avg"]["f1-score"])
    }

# -----------------------------
# Train function
# -----------------------------
def train_and_evaluate(model, model_name, epochs):
    print(f"\\n===== Training {model_name} =====")
    start_time = time.time()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{MODEL_DIR}/{model_name}.keras",
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    history = model.fit(
        train_ds_plain,
        validation_data=val_ds_plain,
        epochs=epochs,
        callbacks=callbacks
    )

    train_time = time.time() - start_time
    plot_history(history, model_name)

    result = evaluate_model(model, test_ds_plain, model_name)
    result["train_time_sec"] = round(train_time, 2)
    return result

# -----------------------------
# Main
# -----------------------------
def main():
    all_results = []

    simple_cnn = build_simple_cnn()
    all_results.append(train_and_evaluate(simple_cnn, "SimpleCNN", EPOCHS_CNN))

    mobilenet = build_mobilenet()
    all_results.append(train_and_evaluate(mobilenet, "MobileNetV2", EPOCHS_TRANSFER))

    resnet = build_resnet()
    all_results.append(train_and_evaluate(resnet, "ResNet50", EPOCHS_TRANSFER))

    print("\\n===== Summary =====")
    for r in all_results:
        print(r)

    with open(f"{RESULT_DIR}/summary_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()