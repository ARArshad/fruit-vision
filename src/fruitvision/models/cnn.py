"""Convolutional neural network architecture for fruit classification."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape: tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """Create a CNN classifier using Keras Sequential API."""

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            data_augmentation,
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPool2D(),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPool2D(),
            layers.Dropout(0.25),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPool2D(),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
