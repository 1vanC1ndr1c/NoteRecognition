import sys
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path

from note_recognition_app_v2.console_output.console_output_constructor import construct_output
from note_recognition_app_v2.image_segmentation_dataset_generator.row_splitter import split_into_rows
from note_recognition_app_v2.image_segmentation_dataset_generator.single_element_template_matcher import \
    extract_elements_by_template_matching
from note_recognition_app_v2.image_segmentation_dataset_generator.generator import generate_train_element


def main():
    """
    Main function for image processing.
    Calls on module for row splitting (first), and then module for individual elements extraction(second).
    Results are then used for generating the dataset.
    """

    # Get the path to the input images.
    input_images_path = os.path.abspath(os.path.join(str(Path(__file__).parent), 'resources', 'input_images'))

    # Get all the images in said folder.
    input_images = [f for f in listdir(input_images_path) if isfile(join(input_images_path, f))]

    # Iterate through those images.
    for input_image in input_images:
        input_image_path = os.path.join(input_images_path, input_image)  # Construct the path to the individual image.

        construct_output(indent_level="block", message="Processing the resources image ({}).".format(input_image))

        row_positions = split_into_rows(input_image_path)  # Firstly, extract rows.
        # Then, extract elements from those rows.
        x_coords_by_row_number = extract_elements_by_template_matching(input_image)

        element_positions = list()  # element_positions = list(tuple(Y_UP, Y_DOWN, X_LEFT, X_RIGHT)
        for c in x_coords_by_row_number:
            element_positions.append((row_positions[c[0]], c[1]))

        # Use the gathered element positions to generate masks for training.
        generate_train_element(input_image_path, element_positions)


if __name__ == '__main__':
    sys.exit(main())
