from PIL import Image, ImageEnhance, ImageOps
import random
import cv2
import numpy as np



def rotate_image(image, max_angle=15):
    """
    Rotate image by a random angle with a white background for remaining areas.
    """
    angle = random.uniform(-max_angle, max_angle)
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=(255, 255, 255))
    return rotated


def add_noise(image, noise_level=0.05):
    """
    Add random noise to image
    """
    noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    return noisy


def adjust_brightness(image, max_factor=0.3):
    """
    Adjust image brightness
    """
    factor = random.uniform(1 - max_factor, 1 + max_factor)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_image)
    enhanced = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)


def adjust_contrast(image, max_factor=0.3):
    """
    Adjust image contrast
    """
    factor = random.uniform(1 - max_factor, 1 + max_factor)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)


def mirror_image(image):
    """
    Mirror image horizontally
    """
    return cv2.flip(image, 1)


def add_blur(image, max_kernel=3):
    """
    Add Gaussian blur
    """
    kernel_size = random.randrange(1, max_kernel, 2)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def add_shear(image, intensity=0.2):
    """
    Apply shear transformation
    """
    shear = random.uniform(-intensity, intensity)
    height, width = image.shape[:2]
    shear_matrix = np.float32([[1, shear, 0], [0, 1, 0]])
    sheared = cv2.warpAffine(image, shear_matrix, (width, height), borderValue=(255, 255, 255))
    return sheared


def crop_and_resize(image, max_crop=0.1):
    """
    Random crop and resize back to original size
    """
    height, width = image.shape[:2]
    crop_percent = random.uniform(0, max_crop)

    crop_height = int(height * (1 - crop_percent))
    crop_width = int(width * (1 - crop_percent))

    start_x = random.randint(0, width - crop_width)
    start_y = random.randint(0, height - crop_height)

    cropped = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
    resized = cv2.resize(cropped, (width, height))
    return resized
