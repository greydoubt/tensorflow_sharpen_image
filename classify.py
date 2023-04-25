import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def classify_dataset(input_dir, output_dir):
    """
    Classifies a directory of preprocessed images as noisy or not noisy.

    Args:
        input_dir: A string representing the path to the input directory.
        output_dir: A string representing the path to the output directory.

    Returns:
        A list of the filenames of the images classified as not noisy.
    """
    # Set up the input data pipeline
    datagen = ImageDataGenerator(rescale=1./255)
    input_data = datagen.flow_from_directory(
        input_dir,
        target_size=(256, 256),
        batch_size=32,
        class_mode="binary"
    )

    # Define the model architecture
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(256, 256, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Train the model on the input data
    model.fit(
        input_data,
        epochs=10
    )

    # Save the model to the output directory
    model.save(os.path.join(output_dir, "model"))

    # Get the filenames of the images classified as not noisy
    not_noisy_filenames = input_data.filenames[np.where(model.predict(input_data) > 0.5)[0]]

    # Save the filenames to a log file
    with open(os.path.join(output_dir, "sharpest.txt"), "w") as f:
        f.write("\n".join(not_noisy_filenames))

    # Return the list of filenames
    return not_noisy_filenames
