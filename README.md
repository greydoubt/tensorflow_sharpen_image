# tensorflow_sharpen_image
classify images based on sharpness and use the output to apply FFT to sharpen the entire dataset

This code first preprocesses the original images, then classifies the preprocessed images as noisy or not noisy using the classify_dataset function from classify.py. The function saves the model to the CLASSIFICATION_DIR directory and also returns a list of the filenames of the images classified as not noisy, which are considered the sharpest images.

The code then applies Fast Fourier Transform to sharpen the preprocessed images using the fft_images function from fft.py, passing in the list of sharp images returned by classify_dataset. Finally, the sharpened images are saved to a new directory called "sharpened" using the save_images function from fft.py.
