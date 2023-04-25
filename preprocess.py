import tensorflow as tf
import os

def preprocess_image(image_path, resize_dim):
    """
    Preprocesses a single image by resizing it to the specified dimensions and normalizing its pixel values.

    Args:
        image_path: A string representing the path to the image file.
        resize_dim: A tuple representing the dimensions to resize the image to.

    Returns:
        The preprocessed image as a TensorFlow tensor.
    """
    # Load the image from the file
    image = tf.io.read_file(image_path)
    # Decode the image to a tensor with 3 color channels
    image = tf.image.decode_png(image, channels=3)
    # Resize the image to the specified dimensions
    image = tf.image.resize(image, resize_dim)
    # Convert the pixel values to floats between 0 and 1
    image = tf.cast(image, tf.float32) / 255.0
    return image


def preprocess_image_dataset(input_dir, resize_dim, batch_size):
    """
    Preprocesses a directory of images and creates a TensorFlow dataset.

    Args:
        input_dir: A string representing the path to the input directory.
        resize_dim: A tuple representing the dimensions to resize the images to.
        batch_size: An integer representing the batch size for the dataset.

    Returns:
        A TensorFlow dataset containing the preprocessed images.
    """
    # Get a list of all the image file names in the input directory
    file_names = os.listdir(input_dir)
    file_names = [f for f in file_names if f.endswith(".png")]

    # Create a list of file paths for the image files
    file_paths = [os.path.join(input_dir, f) for f in file_names]

    # Preprocess the images and create a TensorFlow dataset
    images = [preprocess_image(f, resize_dim) for f in file_paths]
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
