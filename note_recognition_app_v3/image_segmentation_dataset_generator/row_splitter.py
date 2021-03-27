import os
from pathlib import Path

import cv2
import numpy as np

from note_recognition_app_v3.console_output.console_output_constructor import construct_output


def split_into_rows(img_path, save=False):
    """
    This function splits the resources image into separate rows of note lines.
    :param img_path: Path to the image.
    :param save: Indicates whether the images need to be saved.
    :return: boolean: True if successful, false otherwise.
    """
    try:
        construct_output(indent_level=0, message="Row splitting.")
        img_name = img_path[img_path.rfind('\\') + 1:]  # Extract image name from the given path.
        # Directory name for the directory that will hold the rows of the resources image.
        dir_name = os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'input_images')
        dir_name = os.path.join(dir_name, 'input_images_rows', img_name[:-4])
        if save is True:
            try:  # Try creating a directory.
                construct_output(indent_level=1, message="Creating a folder for the image: {}".format(dir_name))
                os.mkdir(dir_name)
            except OSError as e:
                construct_output(indent_level=1, message="Folder already exists: {}".format(dir_name))

        construct_output(indent_level=1, message="Reading the resources image {}.".format(img_name))
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read the image.
        if len(img.shape) == 3:
            trans_mask = img[:, :, 3] == 0  # Remove any transparency.
            img[trans_mask] = [255, 255, 255, 255]
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to BR2GRAY (grayscale mode).
            # Make a black and white image based on a threshold.
            th, img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            image_w, image_h = img_gray.shape[::-1]  # Image dimensions.
        else:
            img_gray = img
            image_w, image_h = img_gray.shape

        template_path = os.path.abspath(
            os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'templates', 'line_start_templates'))
        row_templates = [file for file in os.listdir(template_path)]

        construct_output(indent_level=1, message="Finding rows in the image.")
        t_locations = []  # List that will contain all the locations that were found by template matching.
        t_dimensions = []  # List that will contain all the dimensions of the found templates on the locations.
        for t in row_templates:  # Iterate through all of the vertical line templates.
            template = cv2.imread(template_path + "\\" + t, 0)  # Read the template from the path.
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)  # Convert the template into a gray image.
            threshold = 0.80  # The threshold to determine whether a part of an image is similar enough to the template.
            locations = np.where(res >= threshold)  # Locations in the image where the template matching found results.
            template_w, template_h = template.shape[::-1]  # Dimensions of the current template.
            # list(zip(*locations[::-1])) -> Match the 'x' and 'y' coordinates into a tuple them and save into a list.
            # Iterate through locations to remove elements already found by previous templates.
            for point in list(zip(*locations[::-1])):
                if len(t_locations) == 0:  # Save the first template matching results without checking.
                    t_locations.append(point)
                    t_dimensions.append((template_w, template_h))  # Also save the template dimensions.
                else:  # Check if 'v_line_locations' already contains the new point +/- 6 px, don't add if true.
                    if not np.intersect1d(list(zip(*t_locations))[1], list(range(point[1] - 6, point[1] + 6))):
                        t_locations.append(point)
                        t_dimensions.append((template_w, template_h))

        construct_output(indent_level=1, message="Saving the found rows into folder: {}".format(dir_name))

        row_positions = list()
        for index, el in enumerate(t_locations):  # Iterate through found locations.
            row_position = tuple((el[1] - 40, el[1] + t_dimensions[index][1] + 40))
            row_positions.append(row_position)

            if save is True:
                img_slice_name_and_path = dir_name + "/row" + str(index) + ".png"  # Generate a path and a name.
                # Cut the part of the img.
                img_slice = img_gray[el[1] - 40:el[1] + t_dimensions[index][1] + 40:, 0:image_w]
                cv2.imwrite(img_slice_name_and_path, img_slice)  # Save that part of the image.

        return row_positions

    except Exception as e:  # Catch exception.
        print(e)
        exit(-1)  # If any exceptions caught, return False.
    construct_output(indent_level=0, message="Row splitting done.")
