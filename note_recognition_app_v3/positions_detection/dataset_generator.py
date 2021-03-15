import sys
import os
import cv2
from os import listdir
from os.path import isfile, join

from pathlib import Path


def main():
    input_images_path = os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'input_images')

    input_images = [f for f in listdir(input_images_path) if isfile(join(input_images_path, f))]
    input_images = [img for img in input_images if img.endswith('.png')]

    for input_img in input_images:
        cv2.imread(os.path.join(input_images_path, input_img), cv2.IMREAD_UNCHANGED)
        # ...etc


if __name__ == '__main__':
    sys.exit(main())
