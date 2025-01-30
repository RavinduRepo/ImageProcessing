import cv2
import numpy as np
from scipy.signal import convolve2d

def apply_motion_blur(image, psf_size, shift):
    """Applies motion blur to an image using a PSF of given size and shift."""
    PSF = np.zeros((psf_size, psf_size))
    center = psf_size // 2
    PSF[center, :] = 1  # Horizontal line for motion blur
    PSF = np.roll(PSF, shift, axis=1)  # Shift horizontally by the given amount
    
    # Normalize PSF for consistent brightness
    PSF /= PSF.sum()
    
    # Convert image to float64 and normalize
    image_double = image.astype(np.float64) / 255.0
    blurred = convolve2d(image_double, PSF, mode='same', boundary='wrap')
    return blurred, PSF
