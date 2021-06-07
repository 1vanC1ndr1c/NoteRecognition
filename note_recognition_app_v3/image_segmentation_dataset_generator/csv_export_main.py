import csv
import os
import sys
from os import listdir
from os.path import isfile, join
from pathlib import Path
import random
import numpy as np

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
        saved_rows_path = os.path.join(saved_rows_path, 'input_images_rows_v3')
        saved_rows_path = os.path.join(saved_rows_path, input_img_name[:-4])

        # Get all row names.
        row_names = [f for f in listdir(saved_rows_path) if isfile(join(saved_rows_path, f))]
        row_names = [r for r in row_names if r.endswith('.png')]

        for row_name in row_names:
            row_path = os.path.join(saved_rows_path, row_name)
            row = cv2.imread(row_path, cv2.IMREAD_UNCHANGED)  # Read the image.

            # Get element positions and list that indicates if the element is recognized or not.
            element_positions, recognized_list = get_positions(row_path, row_name)

            for index in range(5):
                print(f'{" " * 20} Doing rng image for the row {row_name} ({index + 1} / 5.)')
                rng_pos, rng_list, rng_row = get_randomized_rows(
                    element_positions,
                    recognized_list,
                    row)
                if rng_list is None:
                    continue

                rng_row_name = input_img_name[:-4] + '_rng_' + str(index) + '_' + row_name

                for el_index, el in enumerate(rng_pos):  # Iterate over all the elements.
                    start_x, start_y = (el[1][0], el[0][0])
                    end_x, end_y = (el[1][1], el[0][1])

                    # Convert the points from numpy numbers.
                    start_x = int(start_x)
                    start_y = int(start_y)
                    end_x = int(end_x)
                    end_y = int(end_y)

                    with open(csv_file_path, mode='a', newline='') as csv_file:
                        _writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                        _writer.writerow([rng_row_name, start_x, start_y, end_x, end_y,
                                          "recognized" if rng_list[el_index] is True else "not_recognized"])

                cv2.imwrite(os.path.join(csv_path, rng_row_name), rng_row)

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


def get_randomized_rows(element_positions, recognized_list, row):
    # Randomize the rows and list order.
    tmp = list(zip(element_positions[1:], recognized_list[1:]))
    random.shuffle(tmp)

    # Unpack it.
    new_el_pos, new_rec_list = zip(*tmp)

    new_el_pos = list(new_el_pos)
    new_el_pos.insert(0, element_positions[0])
    new_el_pos = tuple(new_el_pos)

    new_rec_list = list(new_rec_list)
    new_rec_list.insert(0, recognized_list[0])
    new_rec_list = tuple(new_rec_list)

    # Get the new image with that order.
    new_positions = []

    row_hgt, row_width = row.shape
    # Create new img.
    new_row = np.ones((row_hgt, 1), np.uint8)

    # Add the new element position (first element).
    new_positions.append((new_el_pos[0][0], (new_el_pos[0][1][0], new_el_pos[0][1][1])))

    # Add the part of the row before the first element to the image.
    new_row = np.concatenate((new_row, row[0:row_hgt, 0:element_positions[0][1][0]]), axis=1)
    # Add the first element to the image.
    new_row = np.concatenate((new_row, row[0:row_hgt, element_positions[0][1][0]:element_positions[0][1][1]]), axis=1)

    # Calculate the average distance between the elements.
    sum_dist = 0
    index = 0
    for index, (el1, el2) in enumerate(zip(element_positions[:-1], element_positions[1:])):
        dist = el2[1][0] - el1[1][1]
        sum_dist = sum_dist + dist
    avg_dist = sum_dist // index if index > 0 else 9
    if avg_dist > 10:
        avg_dist = 9
    # Add some empty space to the image.
    empty_space = row[0:row_hgt, element_positions[0][1][1]:element_positions[0][1][1] + avg_dist]
    new_row = np.concatenate((new_row, empty_space), axis=1)

    for index, el in enumerate(new_el_pos[1:]):
        width = el[1][1] - el[1][0]
        new_positions.append((el[0], (new_row.shape[1], new_row.shape[1] + width)))
        # Add an element
        new_row = np.concatenate((new_row, row[0:row_hgt, el[1][0]:el[1][1]]), axis=1)
        # Add some empty space.
        if index < len(new_el_pos) - 2:
            new_row = np.concatenate((new_row, empty_space), axis=1)

    if max([x[1][1] for x in new_positions]) >= row.shape[1]:
        return None, None, None

    right_pad = row.shape[1] - new_row.shape[1]
    if right_pad < 0:
        return None, None, None

    if right_pad > 0:
        right_pad = np.ones((row_hgt, right_pad), np.uint8)
        new_row = np.concatenate((new_row, right_pad), axis=1)

    if new_row.shape != row.shape:
        return None, None, None

    return list(new_positions), list(new_rec_list), new_row


if __name__ == '__main__':
    sys.exit(main())
