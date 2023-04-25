import os
import numpy as np
import cv2

def fft_images(input_dir, output_dir, sharp_images):
    """
    Apply Fast Fourier Transform to the images in input_dir and save the
    sharpened images to output_dir. Only the images in sharp_images will be
    sharpened.
    """
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Load the image and apply Fast Fourier Transform
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        
        # Apply high-pass filter to sharpen the image if it is not noisy
        if filename in sharp_images:
            rows, cols = img.shape
            crow, ccol = rows // 2, cols // 2
            fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
        else:
            # Apply low-pass filter to remove noise
            fshift[:10, :] = 0
            fshift[-10:, :] = 0
            fshift[:, :10] = 0
            fshift[:, -10:] = 0
        
        # Apply inverse Fourier Transform to get the image back
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Save the sharpened image to output_dir
        cv2.imwrite(output_path, img_back)

def save_images(input_dir, output_dir):
    """
    Save the images in input_dir to output_dir with the same filenames.
    """
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(output_path, img)

def save_sharp_images(sharp_images, filename):
    """
    Save the list of sharp images to a file.
    """
    with open(filename, "w") as f:
        for img in sharp_images:
            f.write(img + "\n")
