import os
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

"""
This script is used to generate more images for the dataset by using existing images and
shifting them pixel-wise by 5 left and right and saving those shifted images.
Also, for every shifted image, an image with gaussian noise is also saved.
"""


def shift(input_img_path):
    """
    The main function reads the images from the original dataset_OLD and removes the images that were already shifted.
    The unshifted images are then sent do the function that shifts them.
    """
    # # Get the path to the dataset directory.
    images_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent.parent)))
    images_path = os.path.abspath(os.path.join(images_path, "resources", "input_images", "input_images_rows_v3"))
    images_path = os.path.join(images_path, input_img_path.split('\\')[-1][:-4])
    # Get all the files in said directory.
    row_images = [img_name for img_name in os.listdir(images_path)]
    row_images = [img_name for img_name in row_images if img_name.endswith('.png')]

    unprocessed_images = []  # A list that contains only the images that need to be processed.
    for index, img_name in enumerate(row_images):  # Iterate through the image names.
        # If a name contains a 'plus' or a 'minus', it means that it is already shifted.
        if "plus" not in img_name:
            if "minus" not in img_name:
                # If a name if a substring of another name in the same list, the image was already processed.
                # Explanation: example: 'img01_minus1px.png' means that 'img01.png' was processed.
                if not any(img_name in s for i, s in enumerate(row_images) if i != index):
                    # Append only unprocessed images.
                    unprocessed_images.append(img_name)

    # Iterate through unprocessed images.
    for index, img_name in enumerate(unprocessed_images):
        # Generate shifted images based on a unprocessed image.
        generate_more_images(img_name, images_path, index + 1, len(unprocessed_images))


def generate_more_images(img_name, path, current_index, total):
    """
    This function generates shifted images and images with added noise (gaussian).
    :param img_name: Image name.
    :param path: Path to the directory that contains the image.
    :param current_index: Current index (out of all images) of the image that is being processed.
    :param total: Total number of images.
    """
    SHIFT = [5, 10, 12, 15]  # List with shift values (pixels).

    img_path = os.path.join(path, img_name)  # Construct the path to the image.
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read the image.
    image_h, image_w = img.shape  # Get the image dimensions.

    for index, s in enumerate(SHIFT):
        # Print out the current progress.
        print("'\rImage({} of {}): '{}', Iteration: {} out of {}."
              .format(current_index + 1, total, img_name, index, len(SHIFT)), end='')

        px_right = deepcopy(img[0:image_h, (image_w - s):image_w])  # Get the rightmost column of pixels.
        px_left = deepcopy(img[0:image_h, 0:s])  # Get the leftmost column of pixels.

        img_plus_x = deepcopy(img)
        img_minus_x = deepcopy(img)

        # Add a column of pixels (1 pixel wide) to the right.
        img_plus_x = np.concatenate((img_plus_x[0: image_h, s:image_w], px_right), axis=1)
        # Add a column of pixels (1 pixel wide) to the left.
        img_minus_x = np.concatenate((px_left, img_minus_x[0:image_h, 0:image_w - s]), axis=1)

        # Generate image names.
        img_plus_x_name = img_name[:-4] + "_plus_" + str(s) + "x" + ".png"
        img_minus_x_name = img_name[:-4] + "_minus_" + str(s) + "x" + ".png"

        # Save
        full_path = os.path.join(path, img_plus_x_name)
        cv2.imwrite(full_path, img_plus_x)
        full_path = os.path.join(path, img_minus_x_name)
        cv2.imwrite(full_path, img_minus_x)

    print()
