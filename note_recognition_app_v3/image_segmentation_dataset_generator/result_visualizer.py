import os
from pathlib import Path

import cv2
import numpy as np

from note_recognition_app_v3.image_segmentation_dataset_generator.img_resizer import ResizeWithAspectRatio


def generate_result_img(slices):
    """ This function is only used to draw the results of element extraction.
        It is not needed for the application to work correctly.
        It only generates and image that can be displayed to view the results.
    :param slices: List of image input_images_individual_elements extracted from the original image.

    :return: list : A generated image displaying all the input_images_individual_elements in (possibly) multiple rows.
    """

    # Read the image separator that will be put between  individual elements (a ordinary vertical green line).
    # Generate the path to the vertical separator.
    sp_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'display'))
    sp_path = os.path.abspath(os.path.join(sp_path, 'img_separator_ver.png'))
    column_splitter = cv2.imread(sp_path)  # Read the image.
    # Resize it to be the same height as the individual elements (otherwise the concatenation will not work).
    column_splitter = ResizeWithAspectRatio(column_splitter, height=200)

    # Read the image separator that will be put between individual elements (a ordinary horizontal green line).
    # Generate the path to the horizontal separator.
    sp_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'display'))
    sp_path = os.path.abspath(os.path.join(sp_path, 'img_separator_hor.png'))
    row_splitter = cv2.imread(sp_path)  # Read the image.
    # Resize it to be the same width as the individual elements (otherwise the concatenation will not work).
    row_splitter = ResizeWithAspectRatio(row_splitter, width=200 * 5)

    slice_rows = []  # Individual rows (200 px tall) of the newly generated image.
    # Put the first slice in the first row in the first position and add a green line after it.
    slice_row = np.concatenate((slices[0], column_splitter), axis=1)
    for index, el in enumerate(slices):  # Iterate through all the input_images_individual_elements.
        # If it is not the first one (already added) and if it is not the every 5th element...
        if index != 0 and index % 5 != 0:
            slice_row = np.concatenate((slice_row, el), axis=1)  # Add a new element into the current row.
            slice_row = np.concatenate((slice_row, column_splitter), axis=1)  # Add a green line after the new element.

        if index % 5 == 0 and index > 0:  # After 5 elements, go into a new line.
            slice_rows.append(slice_row)  # Append the current (full containing 5 elements) row into the list.
            slice_row = slices[index]  # Put the current element into a new row.
            slice_row = np.concatenate((slice_row, column_splitter), axis=1)  # Add a green line after it.

        elif index == len(slices) - 1:  # If it is the last element...
            slice_rows.append(slice_row)  # Append it into the list.

    row_w = slice_rows[0].shape[1]  # Get the width of the first row.
    # Into the final image concatenate the first row from 'slice_rows' and add a green line under it.
    final_img = np.concatenate((slice_rows[0], ResizeWithAspectRatio(row_splitter, width=row_w)), axis=0)
    for index, row in enumerate(slice_rows):  # Iterate through all the rows.
        current_row_w = row.shape[1]  # Get the current row width.
        # If it is not the first (already added)
        # and if it's width is the same as the 1st row (meaning it is also a full row)...
        if index != 0 and current_row_w == row_w:
            # Add the row.
            final_img = np.concatenate((final_img, row), axis=0)
            # Add a green line under that row.
            final_img = np.concatenate((final_img, ResizeWithAspectRatio(row_splitter, width=row_w)), axis=0)
        # If the current row is smaller that the first one (meaning it is the last row which is not full)...
        elif current_row_w < row_w:
            # Calculate the needed expansion to make it a full row (roughly 5 * 200 px + the green lines between).
            w_to_expand = row_w - current_row_w
            # Generate the path to the full black image that will be used to fill the rest of the row.
            f_path = os.path.abspath(
                os.path.join(str(Path(__file__).parent.parent.parent),
                             'resources',
                             'display'))
            f_path = os.path.abspath(os.path.join(f_path, 'filler.png'))
            filler = cv2.imread(f_path)  # Read the image.
            filler = cv2.resize(filler, (w_to_expand, 200))  # Resize it to be the needed width and 200 px tall.
            row = np.concatenate((row, filler), axis=1)  # Add the filler to the end of the row.
            # Now, when the row is the right width, add it to the final image.
            final_img = np.concatenate((final_img, row), axis=0)

    return final_img  # Return the generated final image.
