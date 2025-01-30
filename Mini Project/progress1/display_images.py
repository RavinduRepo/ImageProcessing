import matplotlib.pyplot as plt

def display_images(original, blurred, restored):
    """Displays the original, blurred, and restored images in a single figure."""
    plt.figure(figsize=(12, 4))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Blurred Image
    plt.subplot(1, 3, 2)
    plt.imshow(blurred, cmap='gray')
    plt.title("Blurred Image")
    plt.axis('off')

    # Restored Image
    plt.subplot(1, 3, 3)
    plt.imshow(restored, cmap='gray')
    plt.title("Restored Image")
    plt.axis('off')

    plt.show()
    