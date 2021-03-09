import os
from pathlib import Path
import shutil

import cv2
import numpy as np

from copy import deepcopy

from note_recognition_app_v2.console_output.console_output_constructor import construct_output


def generate_train_element(input_image_path, element_positions):
    """
    For input image and element positions on the input image, generate masks using those positions.
    """

    construct_output(indent_level=0, message="Row Generating the dataset for finding element positions.")
    # Get the image name from the input path.
    img_name = input_image_path.split('\\')[-1][:-4]
    construct_output(indent_level=1, message="Adding {} to the dataset".format(img_name))
    # Generate the path to the training directory.
    train_dataset_input = os.path.join(str(Path(__file__).parent.parent),
                                       'positions_detection', 'resources', 'train', img_name)
    # Generate the path to the original image directory.
    images_folder = os.path.join(train_dataset_input, 'images')
    # Generate the path to the masks directory.
    masks_folder = os.path.join(train_dataset_input, 'masks')

    construct_output(indent_level=2, message="Creating the needed directories.")
    try:  # Generate needed directory structure.
        os.mkdir(train_dataset_input)
        os.mkdir(images_folder)
        os.mkdir(masks_folder)
    except FileExistsError as _:
        pass

    construct_output(indent_level=2, message="Saving a non-transparent image into {}.".format(images_folder))
    # Get the original image.
    org_omg = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    img_w, img_h = org_omg.shape[:-1]  # Get the dimensions of the original image.
    trans_mask = org_omg[:, :, 3] == 0  # Remove any transparency.
    org_omg[trans_mask] = [255, 255, 255, 255]
    img_gray = cv2.cvtColor(org_omg, cv2.COLOR_BGR2GRAY)  # Convert to BR2GRAY (grayscale mode).
    _, img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    img_name = img_name + '.png'
    img_name = os.path.join(images_folder, img_name)
    cv2.imwrite(img_name, img_gray)  # Save the non-transparent original image into the dataset.

    # Construct a black image with the same dimensions.
    black_image = np.zeros((img_w, img_h), np.uint8)

    # Iterate over found elements.
    construct_output(indent_level=2, message="***Creating masks.***")
    for index, element_position in enumerate(element_positions):
        construct_output(indent_level=2, message="Creating mask {}.".format(index))
        # Generate a a copy of a black image.
        temp_black_img = deepcopy(black_image)
        # Get the individual element position.
        y_start = element_position[0][0]
        y_end = element_position[0][1]
        x_start = element_position[1][0]
        x_end = element_position[1][1]
        # Extract the element.
        element = img_gray[y_start: y_end, x_start:x_end]
        # Make a white rectangle over the element position.
        element.fill(255)
        # Save that rectangle onto the black image.
        temp_black_img[y_start: y_end, x_start:x_end] = img_gray[y_start: y_end, x_start:x_end]

        # Generate mask name
        # (mask being the black image containing a single white rectangle in the place of the element).
        mask_name = 'mask' + str(index) + '.png'
        mask_name = os.path.join(masks_folder, mask_name)
        # Save the mask.
        cv2.imwrite(mask_name, temp_black_img)
        construct_output(indent_level=3,
                         message="Mask {} saved into {}.".format('mask' + str(index) + '.png', masks_folder))
