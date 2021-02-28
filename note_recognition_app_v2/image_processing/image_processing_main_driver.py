import os
from pathlib import Path

import cv2

from note_recognition_app_v2.console_output.console_output_constructor import construct_output
from note_recognition_app_v2.image_processing.row_splitter import split_into_rows
from note_recognition_app_v2.image_processing.img_resizer import ResizeWithAspectRatio
from note_recognition_app_v2.image_processing.single_element_template_matcher import \
    extract_elements_by_template_matching


def process_image(input_image_path):
    """
    Main function for image processing.
    Calls on module for row splitting (first), and then module for individual elements extraction(second).
    :param input_image_path: Path to the image that needs to be processed.
    """
    img_name = input_image_path[input_image_path.rfind('\\') + 1:]  # Extract image name from the given path.
    construct_output(indent_level="block", message="Processing the input image ({}).".format(img_name))

    row_positions = split_into_rows(input_image_path)  # Firstly, extract rows.
    x_coords_by_row_number = extract_elements_by_template_matching(img_name)  # Then, extract elements from those rows.

    # element_positions = list(tuple(Y_UP, Y_DOWN, X_LEFT, X_RIGHT)
    element_positions = list()
    for c in x_coords_by_row_number:
        element_positions.append((row_positions[c[0]], c[1]))

    img_location = os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'input_images')
    img_location = os.path.join(img_location, img_name)

    img = cv2.imread(img_location, cv2.IMREAD_UNCHANGED)  # Read the image.
    trans_mask = img[:, :, 3] == 0  # Remove any transparency.
    img[trans_mask] = [255, 255, 255, 255]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to BR2GRAY (grayscale mode).
    th, img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

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
