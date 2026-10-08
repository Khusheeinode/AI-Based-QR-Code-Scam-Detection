"""
train_cnn.py

Trains the Branch A image classifier: genuine vs tampered QR image.
Uses transfer learning on MobileNetV2.

Run (from project root):
    python branch_a_image/train_cnn.py
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 8
EPOCHS_FINE_TUNE = 6
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "qr_cnn_model.keras")


def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary",
    )
    class_names = train_ds.class_names  # ['genuine', 'tampered'] alphabetical
    print("Class order (label 0 / 1):", class_names)

    normalization = layers.Rescaling(1.0 / 127.5, offset=-1)  # MobileNetV2 expects [-1, 1]
    train_ds = train_ds.map(lambda x, y: (normalization(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization(x), y)).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False  # freeze for the head-training phase

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    return model, base


def main():
    train_ds, val_ds, class_names = build_datasets()
    model, base = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )

    print("\n--- Phase 1: training classification head ---")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

    print("\n--- Phase 2: fine-tuning top layers of MobileNetV2 ---")
    base.trainable = True
    for layer in base.layers[:-30]:   # keep most of the backbone frozen
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),  # low LR for fine-tuning
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE_TUNE)

    model.save(MODEL_OUT)
    print(f"\nSaved model to {MODEL_OUT}")
    print("Remember: class_names =", class_names, "-> label 1 is:", class_names[1])


if __name__ == "__main__":
    main()
