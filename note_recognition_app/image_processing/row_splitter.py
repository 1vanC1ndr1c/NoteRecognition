import os
from pathlib import Path

import cv2
import numpy as np


def split_into_rows(img_name):
    try:
        dir_name = "resources/" + img_name[:-4]
        try:
            os.mkdir(dir_name)
        except OSError as e:
            pass

        path = os.path.join(os.getcwd(), 'resources', img_name)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        trans_mask = img[:, :, 3] == 0
        img[trans_mask] = [255, 255, 255, 255]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th, img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        image_w, image_h = img_gray.shape[::-1]

        template_path = os.path.abspath(
            os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'templates', 'line_start_templates'))
        row_templates = [file for file in os.listdir(template_path)]

        t_locations = []
        t_dimensions = []
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

        for index, el in enumerate(t_locations):
            img_slice_name = dir_name + "/row" + str(index) + ".png"
            img_slice = img_gray[el[1] - 40:el[1] + t_dimensions[index][1] + 40:, 0:image_w]
            cv2.imwrite(img_slice_name, img_slice)
    except Exception as e:
        print(e)
        return False
    return True
