from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

layers = keras.layers
models = keras.models
callbacks = keras.callbacks
applications = keras.applications
utils = keras.utils

MobileNetV2 = applications.MobileNetV2
EarlyStopping = callbacks.EarlyStopping
ModelCheckpoint = callbacks.ModelCheckpoint
ReduceLROnPlateau = callbacks.ReduceLROnPlateau
image_dataset_from_directory = utils.image_dataset_from_directory

# =========================================================
# Configuration
# =========================================================

DATA_DIR = r"C:\Users\LENOVO\Desktop\satellite_image_land_classification\data\EuroSAT"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "land_classifier_model.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "class_labels.json")

IMAGE_SIZE: Tuple[int, int] = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2
SEED = 42
LEARNING_RATE = 1e-3

# If your laptop is weak, use:
# IMAGE_SIZE = (96, 96)
# BATCH_SIZE = 16
# EPOCHS = 5


# =========================================================
# Utility Functions
# =========================================================

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_directories() -> None:
    """Create required directories if they do not exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)


def validate_dataset_structure(data_dir: str) -> None:
    """
    Validate dataset folder structure.

    Args:
        data_dir: Path to dataset root

    Raises:
        FileNotFoundError: If dataset path does not exist
        ValueError: If no class folders are found
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset path not found: {data_dir}")

    class_dirs = [p for p in Path(data_dir).iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(
            "No class folders found inside dataset directory.\n"
            "Expected structure like:\n"
            "EuroSAT/Forest, EuroSAT/River, etc."
        )

    print(f"Dataset found at: {data_dir}")
    print(f"Number of class folders found: {len(class_dirs)}")
    print("Classes:")
    for class_dir in sorted(class_dirs):
        print(f" - {class_dir.name}")


def count_images(data_dir: str) -> int:
    """
    Count total image files in dataset.

    Args:
        data_dir: Dataset path

    Returns:
        Number of image files
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    total = 0
    for root, _, files in os.walk(data_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_exts:
                total += 1
    return total


def save_class_labels(class_names: list[str], labels_path: str) -> None:
    """
    Save class labels as JSON.

    Args:
        class_names: List of class labels
        labels_path: Output JSON path
    """
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)
    print(f"Class labels saved to: {labels_path}")


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """
    Plot training accuracy and loss.

    Args:
        history: Keras training history object
    """
    hist = history.history

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(hist["accuracy"], label="Train Accuracy")
    plt.plot(hist["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(hist["loss"], label="Train Loss")
    plt.plot(hist["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


# =========================================================
# Dataset Loading
# =========================================================

def get_augmentation_layer() -> tf.keras.Sequential:
    """
    Build data augmentation layer.

    Returns:
        Keras Sequential augmentation pipeline
    """
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="train_augmentation",
    )
    return augmentation


def preprocess_train_batch(
    images: tf.Tensor,
    labels: tf.Tensor,
    augmentation_layer: tf.keras.Sequential,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Apply augmentation + MobileNetV2 preprocessing on training batches.

    Args:
        images: Batch of images
        labels: Batch of labels
        augmentation_layer: Keras augmentation pipeline

    Returns:
        Preprocessed images and labels
    """
    images = tf.cast(images, tf.float32)
    images = augmentation_layer(images, training=True)
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
    return images, labels


def preprocess_val_batch(
    images: tf.Tensor,
    labels: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Apply MobileNetV2 preprocessing on validation batches.

    Args:
        images: Batch of images
        labels: Batch of labels

    Returns:
        Preprocessed images and labels
    """
    images = tf.cast(images, tf.float32)
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
    return images, labels


def load_datasets(data_dir: str):
    """
    Load train and validation datasets from directory.

    Args:
        data_dir: Dataset folder path

    Returns:
        train_ds, val_ds, class_names
    """
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    class_names = train_ds.class_names

    print("\nClass names loaded:")
    for idx, name in enumerate(class_names):
        print(f"{idx}: {name}")

    augmentation_layer = get_augmentation_layer()

    train_ds = train_ds.map(
        lambda x, y: preprocess_train_batch(x, y, augmentation_layer),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    val_ds = val_ds.map(
        preprocess_val_batch,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Performance optimization
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


# =========================================================
# Model Building
# =========================================================

def build_model(num_classes: int) -> tf.keras.Model:
    """
    Build transfer learning model using MobileNetV2.

    Args:
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze base model initially

    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="land_classifier_mobilenetv2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =========================================================
# Training
# =========================================================

def get_callbacks():
    """
    Create callbacks for training.

    Returns:
        List of callbacks
    """
    callback_list = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]
    return callback_list


def train_model():
    """Main training pipeline."""
    print("Starting training pipeline...\n")

    # Step 1: setup
    set_seed(SEED)
    ensure_directories()
    validate_dataset_structure(DATA_DIR)

    total_images = count_images(DATA_DIR)
    print(f"\nTotal images found: {total_images}")

    if total_images == 0:
        raise ValueError("No images found in dataset directory.")

    # Step 2: load dataset
    train_ds, val_ds, class_names = load_datasets(DATA_DIR)

    # Step 3: save class labels
    save_class_labels(class_names, LABELS_PATH)

    # Step 4: build model
    model = build_model(num_classes=len(class_names))
    print("\nModel Summary:")
    model.summary()

    # Step 5: train
    print("\nTraining started...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=get_callbacks(),
        verbose=1,
    )

    # Step 6: evaluate
    print("\nEvaluating model on validation data...")
    val_loss, val_accuracy = model.evaluate(val_ds, verbose=1)
    print(f"\nValidation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Step 7: save final model
    model.save(MODEL_PATH)
    print(f"\nFinal model saved to: {MODEL_PATH}")

    # Step 8: plot history
    plot_training_history(history)

    print("\nTraining completed successfully.")


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"\nError: {e}")