import numpy as np

def apply_wiener_deblur_to_motion_blur(blurred_image, PSF, noise_var=0):
    signal_var = np.var(blurred_image)  # Estimate signal variance

    # Zero-padding PSF to match the size of the blurred image
    PSF_padded = np.pad(PSF, [(0, blurred_image.shape[0] - PSF.shape[0]), 
                              (0, blurred_image.shape[1] - PSF.shape[1])], mode='constant')

    # Apply Wiener deconvolution in the frequency domain
    F_blurred = np.fft.fft2(blurred_image)
    F_PSF = np.fft.fft2(PSF_padded)
    F_PSF_conj = np.conj(F_PSF)
    F_PS = np.abs(F_PSF) ** 2

    # Wiener filter
    Wiener_filter = F_PSF_conj / (F_PS + (noise_var / signal_var))
    F_restored = Wiener_filter * F_blurred

    # Inverse FFT to get the restored image
    restored = np.fft.ifft2(F_restored)
    return np.abs(restored)


def apply_wiener_deblur_to_noisy_motion_blur(blurred_noisy, PSF, noise_var, signal_var):
    # Zero-pad the PSF to match the size of the blurred image
    PSF_padded = np.pad(PSF, [(0, blurred_noisy.shape[0] - PSF.shape[0]),
                              (0, blurred_noisy.shape[1] - PSF.shape[1])], mode='constant')

    # Perform FFT of the blurred noisy image and PSF
    F_blurred_noisy = np.fft.fft2(blurred_noisy)
    F_PSF = np.fft.fft2(PSF_padded)
    F_PSF_conj = np.conj(F_PSF)

    # Calculate the Wiener filter
    NSR = noise_var / signal_var  # Noise-to-signal ratio
    F_Wiener_filter = F_PSF_conj / (np.abs(F_PSF)**2 + NSR)

    # Apply the Wiener filter to the blurred image in the frequency domain
    F_restored = F_Wiener_filter * F_blurred_noisy

    # Inverse FFT to get the restored image
    restored = np.fft.ifft2(F_restored)
    restored = np.abs(restored)

    return restored
