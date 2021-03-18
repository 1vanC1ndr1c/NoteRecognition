import os
import string
import sys
from os import listdir
from os.path import isfile, join
from pathlib import Path
import csv

import cv2

from note_recognition_app_v3.image_segmentation_dataset_generator.position_aggregator import get_positions


def main():
    """
    This script converts the input element information gathered from opencv template matching into csv file
    that can be used in vott and later in yolov3.
    """

    # Get the path to the images.
    input_images_path = os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'input_images')
    # Get the path to json files.
    csv_path = os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'input_images', 'csv_info')
    csv_path = os.path.join(csv_path, 'Annotations-export.csv')
    # Get all the images names in aforementioned file.
    input_images = [f for f in listdir(input_images_path) if isfile(join(input_images_path, f))]
    # Filter just the .png files.
    input_images = [img for img in input_images if img.endswith('.png')]

    with open(csv_path, mode='w', newline='') as csv_file:
        employee_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        employee_writer.writerow(["image", "xmin", "ymin", "xmax", "ymax", "label"])

    # Iterate through all the images.
    for input_img_name in input_images:

        _id = 1  # Id of the file stored within the json file.

        # Get the full path to the current image.
        input_img_path = os.path.join(input_images_path, input_img_name)
        img = cv2.imread(input_img_path, cv2.IMREAD_UNCHANGED)  # Read the image.
        trans_mask = img[:, :, 3] == 0  # Remove any transparency.
        img[trans_mask] = [255, 255, 255, 255]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to BR2GRAY (grayscale mode).
        # Make a black and white image based on a threshold.
        th, img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Get element positions and list that indicates if the element is recognized or not.
        element_positions, recognized_list = get_positions(input_img_path, input_img_name)

        for index, el in enumerate(element_positions):  # Iterate over all the elements.

            # point = (x, y)
            start_x, start_y = (el[1][0], el[0][0])
            end_x, end_y = (el[1][1], el[0][1])

            # Convert the points from numpy numbers.
            start_x = int(start_x)
            start_y = int(start_y)
            end_x = int(end_x)
            end_y = int(end_y)

            with open(csv_path, mode='a', newline='') as csv_file:
                print('Writing into .csv')
                employee_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

                employee_writer.writerow([input_img_name, start_x, start_y, end_x, end_y,
                                          "recognized" if recognized_list[index] is True else "not_recognized"])


if __name__ == '__main__':
    sys.exit(main())
