import os
import sys
from os import listdir
from os.path import isfile, join
from pathlib import Path

import cv2

from note_recognition_app_v2.console_output.console_output_constructor import construct_output
from note_recognition_app_v2.image_segmentation_dataset_generator.generator import generate_train_element
from note_recognition_app_v2.image_segmentation_dataset_generator.img_resizer import ResizeWithAspectRatio
from note_recognition_app_v2.image_segmentation_dataset_generator.row_splitter import split_into_rows
from note_recognition_app_v2.image_segmentation_dataset_generator.single_element_template_matcher import \
    extract_elements_by_template_matching


def generator_main():
    """
    Main function for image processing.
    Calls on module for row splitting (first), and then module for individual elements extraction(second).
    Results are then used for generating the dataset.
    """

    # Get the path to the input images.
    input_images_path = os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'input_images')
    output_path = os.path.join(str(Path(__file__).parent.parent), 'positions_detection', 'resources', 'train')
    output_path_check = [f for f in listdir(output_path) if not isfile(join(input_images_path, f))]

    # Get all the images in said folder.
    input_images = [f for f in listdir(input_images_path) if isfile(join(input_images_path, f))]
    # Iterate through those images.
    for input_image in input_images:
        if input_image[:-4] in output_path_check:  # Skip already existing images.
            continue

        input_image_path = os.path.join(input_images_path, input_image)  # Construct the path to the individual image.

        construct_output(indent_level="block", message="Processing the resources image ({}).".format(input_image))

        row_positions = split_into_rows(input_image_path)  # Firstly, extract rows.
        # Then, extract elements from those rows.
        x_coords_by_row_number = extract_elements_by_template_matching(input_image)

        element_positions = list()  # element_positions = list(tuple(Y_UP, Y_DOWN, X_LEFT, X_RIGHT)
        for c in x_coords_by_row_number:
            element_positions.append((row_positions[c[0]], c[1]))

        # draw_results(img_name=input_image, element_positions=element_positions)

        # Use the gathered element positions to generate masks for training.
        generate_train_element(input_image_path, element_positions)


def draw_results(img_name, element_positions):
    img_location = os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'input_images')
    img_location = os.path.join(img_location, img_name)

    img = cv2.imread(img_location, cv2.IMREAD_UNCHANGED)  # Read the image.
    result = img
    if len(img.shape) == 3:
        trans_mask = img[:, :, 3] == 0  # Remove any transparency.
        img[trans_mask] = [255, 255, 255, 255]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to BR2GRAY (grayscale mode).
        th, img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        result = img_gray

    img_gray = result

    color = (255, 0, 0)
    thickness = 2
    for el in element_positions:
        start_point = (el[1][0], el[0][0])
        end_point = (el[1][1], el[0][1])
        img_gray = cv2.rectangle(img_gray, start_point, end_point, color, thickness)

    cv2.imshow("elements", ResizeWithAspectRatio(img_gray, height=900))
    cv2.moveWindow('elements', 200, 200)
    cv2.waitKey()

    construct_output(indent_level="block", message="Input image processing done.")


if __name__ == '__main__':
    sys.exit(generator_main())
