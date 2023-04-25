import os
import preprocess
import classify
import fft

# Set up the directories
ORIGINALS_DIR = "originals"
PREPROCESSED_DIR = "preprocessed"
CLASSIFICATION_DIR = "classification"
FFT_DIR = "fft"

# Preprocess the images
preprocess.preprocess_images(ORIGINALS_DIR, PREPROCESSED_DIR)

# Classify the images as noisy or not noisy
sharp_images = classify.classify_dataset(PREPROCESSED_DIR, CLASSIFICATION_DIR)

# Apply Fast Fourier Transform to sharpen the images
fft.fft_images(PREPROCESSED_DIR, FFT_DIR, sharp_images)

# Save the sharpened images to a new directory
if not os.path.exists("sharpened"):
    os.makedirs("sharpened")
fft.save_images(FFT_DIR, "sharpened")
