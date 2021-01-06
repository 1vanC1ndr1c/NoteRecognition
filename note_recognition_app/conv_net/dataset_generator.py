import os
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from skimage.util import random_noise, img_as_ubyte

"""
This script is used to generate more images for the dataset by using existing images and
shifting them pixel-wise by 10 pixels in every direction and saving those shifted images.
Also, for every shifted image, an image with gaussian noise is also saved.
"""


def main():
    """
    The main function reads the images from the original dataset_OLD and removes the images that were already shifted.
    The unshifted images are then sent do the function that shifts them.
    """
    # Get the path to the dataset directory.
    dataset_images_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent.parent)))
    dataset_images_path = os.path.abspath(os.path.join(dataset_images_path, "resources", "dataset"))
    # Get all the files in said directory.
    dataset_images = [img_name for img_name in os.listdir(dataset_images_path)]

    # reset_dataset(dataset_images_path, dataset_images)

    unprocessed_images = []  # A list that contains only the images that need to be processed.
    for index, img_name in enumerate(dataset_images):  # Iterate through the image names.
        # If a name contains a 'plus' or a 'minus', it means that it is already shifted.
        if "plus" not in img_name and "minus" not in img_name:
            # If a name if a substring of another name in the same list, the image was already processed.
            # Explanation: example: 'img01_minus1px.png' means that 'img01.png' was processed.
            if not any(img_name in s for i, s in enumerate(dataset_images) if i != index):
                # Append only unprocessed images.
                unprocessed_images.append(img_name)

    # Iterate through unprocessed images.
    for index, img_name in enumerate(unprocessed_images):
        # Generate shifted images based on a unprocessed image.
        generate_more_images(img_name, dataset_images_path, index + 1, len(unprocessed_images))


def generate_more_images(img_name, path, current_index, total):
    """
    This function generates shifted images and images with added noise (gaussian).
    :param img_name: Image name.
    :param path: Path to the directory that contains the image.
    :param current_index: Current index (out of all images) of the image that is being processed.
    :param total: Total number of images.
    """

    NO_OF_ITER = 2  # Number of iterations per image. Change to change the size of the new dataset.

    img_path = os.path.join(path, img_name)  # Construct the path to the image.
    img = cv2.imread(img_path)  # Read the image.
    image_w, image_h = img.shape[:-1]  # Get the image dimensions.

    px_right = img[0:image_h, image_w - 1:image_w]  # Get the rightmost column of pixels.
    px_left = img[0:image_h, 0:1]  # Get the leftmost column of pixels.
    px_up = img[image_h - 1:image_h, 0:image_w]  # Get the top row of pixels.
    px_down = img[0:1, 0:image_w]  # Get the bottom row of pixels.

    img_plus_x = deepcopy(img)  # Copy the original image. This image will have pixels added to the right.
    img_minus_x = deepcopy(img)  # Copy the original image. This image will have pixels added to the left.

    for x in range(0, NO_OF_ITER):
        # Print out the current progress.
        print("'\rImage({} of {}): '{}', Iteration: {} out of {}."
              .format(current_index + 1, total, img_name, x, NO_OF_ITER), end='')

        # Add a column of pixels (1 pixel wide) to the right.
        img_plus_x = np.concatenate((img_plus_x[0: image_h, 1:image_w], px_right), axis=1)
        # Add a column of pixels (1 pixel wide) to the left.
        img_minus_x = np.concatenate((px_left, img_minus_x[0:image_h, 0:image_w - 1]), axis=1)

        img_plus_x_plus_y = deepcopy(img_plus_x)  # Image that goes to the right and up.
        img_plus_x_minus_y = deepcopy(img_plus_x)  # Image that goes to the right and down.
        img_minus_x_plus_y = deepcopy(img_minus_x)  # Image that goes to the right and up.
        img_minus_x_minus_y = deepcopy(img_minus_x)  # Image that goes to the right and down.

        for y in range(0, NO_OF_ITER):
            # Add a row of pixels (1 pixel tall) to the top.
            img_plus_x_plus_y = np.concatenate((img_plus_x_plus_y[1:image_h, 0:image_w], px_up), axis=0)
            # Add a row of pixels (1 pixel tall) to the bottom.
            img_plus_x_minus_y = np.concatenate((px_down, img_plus_x_minus_y[0:image_h - 1, 0:image_w]), axis=0)
            # Add a row of pixels (1 pixel tall) to the top.
            img_minus_x_plus_y = np.concatenate((img_minus_x_plus_y[1:image_h, 0:image_w], px_up), axis=0)
            # Add a row of pixels (1 pixel tall) to the bottom.
            img_minus_x_minus_y = np.concatenate((px_down, img_minus_x_minus_y[0:image_h - 1, 0:image_w]), axis=0)

            # Generate images that contain noise.
            img_plus_x_plus_y_noise = img_as_ubyte(random_noise(img_plus_x_plus_y, mode='gaussian'))
            img_plus_x_minus_y_noise = img_as_ubyte(random_noise(img_plus_x_minus_y, mode='gaussian'))
            img_minus_x_plus_y_noise = img_as_ubyte(random_noise(img_minus_x_plus_y, mode='gaussian'))
            img_minus_x_minus_y_noise = img_as_ubyte(random_noise(img_minus_x_minus_y, mode='gaussian'))

            # Generate image names.
            img_plus_x_plus_y_name = img_name[:-4] + "_plus_" + str(x) + "x_" + "plus_" + str(y) + "y" + ".png"
            img_plus_x_minus_y_name = img_name[:-4] + "_plus_" + str(x) + "x_" + "minus_" + str(y) + "y" + ".png"
            img_minus_x_plus_y_name = img_name[:-4] + "_minus_" + str(x) + "x_" + "plus_" + str(y) + "y" + ".png"
            img_minus_x_minus_y_minus = img_name[:-4] + "_minus_" + str(x) + "x_" + "minus_" + str(y) + "y" + ".png"
            img_plus_x_plus_y_name_noise = img_plus_x_plus_y_name[:-4] + "_noise" + ".png"
            img_plus_x_minus_y_name_noise = img_plus_x_minus_y_name[:-4] + "_noise" + ".png"
            img_minus_x_plus_y_name_noise = img_minus_x_plus_y_name[:-4] + "_noise" + ".png"
            img_minus_x_minus_y_minus_noise = img_minus_x_minus_y_minus[:-4] + "_noise" + ".png"

            # Save all the new images.
            full_path = os.path.join(path, img_plus_x_plus_y_name)
            cv2.imwrite(full_path, img_plus_x_plus_y)

            full_path = os.path.join(path, img_plus_x_minus_y_name)
            cv2.imwrite(full_path, img_plus_x_minus_y)

            full_path = os.path.join(path, img_minus_x_plus_y_name)
            cv2.imwrite(full_path, img_minus_x_plus_y)

            full_path = os.path.join(path, img_minus_x_minus_y_minus)
            cv2.imwrite(full_path, img_minus_x_minus_y)

            full_path = os.path.join(path, img_plus_x_plus_y_name_noise)
            cv2.imwrite(full_path, img_plus_x_plus_y_noise)

            full_path = os.path.join(path, img_plus_x_minus_y_name_noise)
            cv2.imwrite(full_path, img_plus_x_minus_y_noise)

            full_path = os.path.join(path, img_minus_x_plus_y_name_noise)
            cv2.imwrite(full_path, img_minus_x_plus_y_noise)

            full_path = os.path.join(path, img_minus_x_minus_y_minus_noise)
            cv2.imwrite(full_path, img_minus_x_minus_y_noise)
    print()


def reset_dataset(dataset_images_path, dataset_images):
    for index, img_name in enumerate(dataset_images):
        if img_name.endswith(".png"):
            if "plus" in img_name:
                img_path = os.path.join(dataset_images_path, img_name)
                os.remove(img_path)
            elif "minus" in img_name:
                img_path = os.path.join(dataset_images_path, img_name)
                os.remove(img_path)


if __name__ == '__main__':
    sys.exit(main())
