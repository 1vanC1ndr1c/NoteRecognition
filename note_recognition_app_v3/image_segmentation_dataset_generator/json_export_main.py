import os
import string
import sys
from os import listdir
from os.path import isfile, join
from pathlib import Path
import json

import random

import cv2

from note_recognition_app_v3.image_segmentation_dataset_generator.position_aggregator import get_positions


def main():
    """
    This script converts the input element information gathered from opencv template matching into json files
    that can be used in vott and later in yolov3.
    """

    # Get the path to the images.
    input_images_path = os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'input_images')
    # Get the path to json files.
    json_files_path = os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'input_images', 'json_info')

    # Get all the images names in aforementioned file.
    input_images = [f for f in listdir(input_images_path) if isfile(join(input_images_path, f))]
    # Filter just the .png files.
    input_images = [img for img in input_images if img.endswith('.png')]

    # Get all the images names in aforementioned file.
    json_files = [f for f in listdir(json_files_path) if isfile(join(json_files_path, f))]
    # Filter just the .json files.
    json_files = [j for j in json_files if j.endswith('.json')]

    # Iterate through all the images.
    for input_img_name in input_images:
        json_file_name = input_img_name[:-4] + '.json'
        if json_file_name in json_files:
            continue
        _id = 1  # Id of the file stored within the json file.

        # Get the full path to the current image.
        input_img_path = os.path.join(input_images_path, input_img_name)
        img = cv2.imread(input_img_path, cv2.IMREAD_UNCHANGED)  # Read the image.
        trans_mask = img[:, :, 3] == 0  # Remove any transparency.
        img[trans_mask] = [255, 255, 255, 255]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to BR2GRAY (grayscale mode).
        # Make a black and white image based on a threshold.
        th, img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        image_w, image_h = img_gray.shape[::-1]  # Image dimensions.

        # Construct a dictionary based on the format of the json file.
        img_info_asset = dict(
            format=input_img_name[-3:],
            id=_id,
            name=input_img_name,
            path=input_img_path,
            size=dict(
                width=image_w,
                height=image_h),
            state=2,
            type=1)

        # Get element positions and list that indicates if the element is recognized or not.
        element_positions, recognized_list = get_positions(input_img_path, input_img_name)
        el_ids = []  # Ids of the elements found in the image.
        regions = []  # Regions is defined by the json file.

        for index, el in enumerate(element_positions):  # Iterate over all the elements.

            # point = (x, y)
            start_x, start_y = (el[1][0], el[0][0])
            end_x, end_y = (el[1][1], el[0][1])

            # Convert the points from numpy numbers.
            start_x = int(start_x)
            start_y = int(start_y)
            end_x = int(end_x)
            end_y = int(end_y)

            # Get all the needed information for json generation.
            el_height = abs(end_y - start_y)
            el_width = abs(end_x - start_x)
            el_left = start_x
            el_top = start_y

            while True:  # Iterate while a unique id has not been found.
                # Get a random sequence of lowercase letters and numbers. Length is 9.
                el_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))
                if el_id not in el_ids:  # Check if the id already exist.
                    # If it does not, append it to the list of already existing file names.
                    el_ids.append(el_id)
                    # Stop the loop.
                    break
            if el_id is None:
                # If id or json file name is non existing, stop the program.
                raise Exception('Generating element ID gone wrong!')

            # Construct a dictionary based on the format of the json file.
            region = dict(
                id=el_id,
                type="RECTANGLE",
                tags=["recognized" if recognized_list[index] is True else "not_recognized"],
                boundingBox=dict(height=el_height,
                                 width=el_width,
                                 left=el_left,
                                 top=el_top),
                points=[dict(x=start_x, y=start_y),
                        dict(x=end_x, y=start_y),
                        dict(x=end_x, y=end_y),
                        dict(x=start_x, y=end_y)])

            # Save the data.
            regions.append(region)

        # Store all the data into one dictionary.
        json_dict = dict(asset=img_info_asset, regions=regions)

        # Get the full path to the save location.
        json_file_full_path = os.path.join(json_files_path, json_file_name)
        with open(json_file_full_path, "w") as fout:
            # Save the file.
            json.dump(json_dict, fout, indent=4)
            cv2.imwrite(os.path.join(json_files_path, input_img_name), img_gray)


if __name__ == '__main__':
    sys.exit(main())
