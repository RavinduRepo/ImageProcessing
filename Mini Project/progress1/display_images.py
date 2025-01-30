import matplotlib.pyplot as plt

def display_images(images, num_images=None):
    """Displays the images with their variable names as titles in a single figure."""
    if num_images is None:
        num_images = len(images)
    
    plt.figure(figsize=(12, 4))
    
    for i, (name, image) in enumerate(images.items(), 1):
        if i > num_images:
            break
        plt.subplot(1, num_images, i)
        plt.imshow(image, cmap='gray')
        plt.title(name.replace('_', ' ').title())
        plt.axis('off')
    
    plt.show()
