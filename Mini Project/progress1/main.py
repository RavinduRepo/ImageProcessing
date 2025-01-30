import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from read_image import read_image
from apply_motion_blur import apply_motion_blur
from apply_wiener_deblur import apply_wiener_deblur_to_motion_blur, apply_wiener_deblur_to_noisy_motion_blur
from display_images import display_images
from apply_gaussian_noise_to_blur import add_gaussian_noise

file_path = 'cameraman.tif'  # Change this to the path of your image file
original_image = read_image(file_path) # Read the file
                                                                    ## cehck later why no effect on this
blurred_image, PSF = apply_motion_blur(original_image, psf_size=21, shift=11) # Apply motion blur

# Display original image, psf and blurred image
display_images({
    'Original Image': original_image,
    'Point Spread Function (PSF)': PSF,
    'Blurred Image': blurred_image
})

restored_image_motion_blur = apply_wiener_deblur_to_motion_blur(blurred_image, PSF)
display_images({
    'Original Image': original_image,
    'Blurred Image': blurred_image,
    'Restored Image (Motion Blur)': restored_image_motion_blur
})

noice_blur_image = add_gaussian_noise(blurred_image, mean=0, var=0.0001)
restored_image_motion_blur = apply_wiener_deblur_to_motion_blur(noice_blur_image, PSF)
display_images({
    'Original Image': original_image,
    'Blurred noice Image': noice_blur_image,
    'Restored Image (Motion Blur)': restored_image_motion_blur
})

restored_image_noisy_motion_blur = apply_wiener_deblur_to_noisy_motion_blur(noice_blur_image, PSF, noise_var=0.01, signal_var=np.var(original_image))
display_images({
    'Original Image': original_image,
    'Blurred Image': blurred_image,
    'Noiced Image': noice_blur_image,
    'Restored Image (Noisy Motion Blur)': restored_image_noisy_motion_blur
})
