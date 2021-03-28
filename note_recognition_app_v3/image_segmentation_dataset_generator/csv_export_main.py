import csv
import os
import sys
from os import listdir
from os.path import isfile, join
from pathlib import Path

import cv2

from note_recognition_app_v3.image_segmentation_dataset_generator import img_shifter
from note_recognition_app_v3.image_segmentation_dataset_generator.position_aggregator import get_positions
from note_recognition_app_v3.image_segmentation_dataset_generator.row_splitter import split_into_rows


def main():
    """
    This script converts the input element information gathered from opencv template matching into csv file
    that can be used in vott and later in yolov3.
    """

    # Get the path to the images.
    input_images_path = os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'input_images')
    # Get the path to json files.
    csv_path = os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'input_images', 'csv_info')
    csv_file_path = os.path.join(csv_path, 'Annotations-export.csv')
    # Get all the images names in aforementioned file.
    input_images = [f for f in listdir(input_images_path) if isfile(join(input_images_path, f))]
    # Filter just the .png files.
    input_images = [img for img in input_images if img.endswith('.png')]

    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        csv_writer.writerow(["image", "xmin", "ymin", "xmax", "ymax", "label"])

    # Iterate through all the images.
    for input_img_name in input_images:
        # Get the full path to the current image.
        input_img_path = os.path.join(input_images_path, input_img_name)

        # Split the image into rows.
        _ = split_into_rows(input_img_path, save=True)

        # Get additional images by shifting the original images.
        img_shifter.shift(input_img_path)

        # Get the path to the rows of the current image.
        saved_rows_path = os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'input_images')
        saved_rows_path = os.path.join(saved_rows_path, 'input_images_rows')
        saved_rows_path = os.path.join(saved_rows_path, input_img_name[:-4])

        # Get all row names.
        row_names = [f for f in listdir(saved_rows_path) if isfile(join(saved_rows_path, f))]
        row_names = [r for r in row_names if r.endswith('.png')]

        for row_name in row_names:
            row_path = os.path.join(saved_rows_path, row_name)
            row = cv2.imread(row_path, cv2.IMREAD_UNCHANGED)  # Read the image.

            # Get element positions and list that indicates if the element is recognized or not.
            element_positions, recognized_list = get_positions(row_path, row_name)

            # Since row names are 'rowX' X is the row number, add the image name in front of it.
            row_name = input_img_name[:-4] + '_' + row_name

            for index, el in enumerate(element_positions):  # Iterate over all the elements.
                start_x, start_y = (el[1][0], el[0][0])
                end_x, end_y = (el[1][1], el[0][1])

                # Convert the points from numpy numbers.
                start_x = int(start_x)
                start_y = int(start_y)
                end_x = int(end_x)
                end_y = int(end_y)

                with open(csv_file_path, mode='a', newline='') as csv_file:
                    _writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

                    _writer.writerow([row_name, start_x, start_y, end_x, end_y,
                                      "recognized" if recognized_list[index] is True else "not_recognized"])

            cv2.imwrite(os.path.join(csv_path, row_name), row)


if __name__ == '__main__':
    sys.exit(main())
