import cv2

def read_image(file_path):
    """Reads the grayscale image from the given file path."""
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
