import os
import sys
from os import listdir
from os.path import isfile, join
from pathlib import Path

import cv2

from note_recognition_app_v3.console_output import console_output_constructor
from note_recognition_app_v3.image_segmentation_dataset_generator.position_aggregator import get_positions


def main():
    input_images_path = os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'input_images')

    input_images = [f for f in listdir(input_images_path) if isfile(join(input_images_path, f))]
    input_images = [img for img in input_images if img.endswith('.png')]

    for input_img in input_images:
        input_img_path = os.path.join(input_images_path, input_img)
        cv2.imread(input_img_path, cv2.IMREAD_UNCHANGED)

        element_positions = get_positions(input_img_path, input_img)

        # TODO save into json


if __name__ == '__main__':
    sys.exit(main())
