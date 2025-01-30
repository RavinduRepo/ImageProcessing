import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from read_image import read_image
from apply_motion_blur import apply_motion_blur
from apply_wiener_deblur import apply_wiener_deblur_to_motion_blur, apply_wiener_deblur_to_noisy_motion_blur
from display_images import display_images


file_path = 'cameraman.tif'  # Change this to the path of your image file
original_image = read_image(file_path)
blurred_image, PSF = apply_motion_blur(original_image, psf_size=21, shift=11)
restored_image_motion_blur = apply_wiener_deblur_to_motion_blur(blurred_image, PSF)
display_images(original_image, blurred_image, restored_image_motion_blur)
restored_image_noisy_motion_blur = apply_wiener_deblur_to_noisy_motion_blur(blurred_image, PSF, noise_var=0.01, signal_var=np.var(original_image))
display_images(original_image, blurred_image, restored_image_noisy_motion_blur)
