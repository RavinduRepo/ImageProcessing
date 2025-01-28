import numpy as np

def add_gaussian_noise(image, mean=0, var=0.0001):
    sigma = np.sqrt(var)  # Standard deviation
    gaussian_noise = np.random.normal(mean, sigma, image.shape)  # Generate noise
    noisy_image = image + gaussian_noise  # Add noise to the image
    noisy_image = np.clip(noisy_image, 0, 1)  # Ensure pixel values are within [0, 1]
    return noisy_image
