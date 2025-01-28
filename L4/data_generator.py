import os
import shutil
import numpy as np
from pathlib import Path
from typing import List, Union
from utils import *


class DataGenerator:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def generate_balanced_dataset(self,
                                  input_dir: Union[str, Path],
                                  output_dir: Union[str, Path],
                                  target_samples_per_class: int=900) -> None:
        """
        Create a balanced dataset with specified number of samples per class

        Args:
            input_dir (Union[str, Path]): Path to input directory containing class folders
            output_dir (Union[str, Path]): Path to output directory where balanced dataset will be created
            target_samples_per_class (int, optional): Target number of samples per class. Defaults to 900.

        """
        # Ensure clean folder is added
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        # Main loop
        for class_name in self.class_names:
            print(f"Processing class {class_name}")

            # Getting directories
            input_class_dir = os.path.join(input_dir, class_name)
            output_class_dir = os.path.join(output_dir, class_name)
            os.mkdir(output_class_dir)

            # Get the list of originals images
            original_images = [f for f in os.listdir(input_class_dir) if f.endswith((".jpg", ".png", "jpeg"))]
            DataGenerator.copy_original_images(input_class_dir, output_class_dir, original_images)

            # Generate the rest images
            DataGenerator.generate_augmented_images(input_class_dir, output_class_dir, original_images, class_name, target_samples_per_class)


    @staticmethod
    def generate_augmented_images(input_class_dir: Union[str, Path],
                                  output_class_dir: Union[str, Path],
                                  original_images: List[str],
                                  class_name: str,
                                  target_samples_per_class: int=900) -> None:

        # Create augmented images
        current_count = len(original_images)
        while current_count < target_samples_per_class:
            for img_name in original_images:
                if current_count >= target_samples_per_class:
                    break

                image = cv2.imread(os.path.join(input_class_dir, img_name))
                augmented = DataGenerator.apply_random_augmentations(image, class_name)

                aug_name = f"aug_{current_count}_{img_name}"
                cv2.imwrite(os.path.join(output_class_dir, aug_name), augmented)
                current_count += 1


    @staticmethod
    def copy_original_images(input_class_dir: Union[str, Path],
                             output_class_dir: Union[str, Path],
                             original_images: List[Union[str, Path]]) -> None:
        """
        Copy original images to new directory
        """
        for img_name in original_images:
            shutil.copy2(
                os.path.join(input_class_dir, img_name),
                os.path.join(output_class_dir, img_name)
            )


    @staticmethod
    def apply_random_augmentations(image: np.ndarray, class_name: str) -> np.ndarray:
        """
        Apply a random combination of augmentation techniques

        Args:
            image (np.ndarray): Input image as a NumPy array
            class_name (str): Name of the class for conditional augmentation

        Returns:
            np.ndarray: Augmented image
        """

        augmentation_funcs = [
            rotate_image,
            add_noise,
            adjust_brightness,
            adjust_contrast,
            add_blur,
            add_shear,
            crop_and_resize,
            mirror_image,
        ]


        if class_name in ["iv", "vi"]:
            augmentation_funcs = augmentation_funcs[:-1]

        num_augmentations = random.randint(2, 4)
        selected_augmentations = random.sample(augmentation_funcs, num_augmentations)


        augmented = image.copy()
        for func in selected_augmentations:
            augmented = func(augmented)


        return augmented