import os
from operator import itemgetter
from pathlib import Path

import cv2
import numpy as np

from note_recognition_app.info_output.output_constructor import construct_output


def extract_elements_by_template_matching(img_name):
    """
    Main function for element extraction that calls on all the sub-functions.
    :param img_name: Name of the input image from which image rows where extracted.
    """
    img_location = os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'input_images')
    img_location = os.path.join(img_location, 'input_images_rows', img_name[:-4])
    rows_numerated = [row for row in os.listdir(img_location) if row.startswith("row")]

    construct_output(indent_level=0, message="Finding individual elements in the saved rows.")
    construct_output(indent_level=1, message="Reading extracted rows.")
    for row_number, row_img in enumerate(rows_numerated):
        construct_output(indent_level=2, message="Reading row number {}.".format(row_number))
        img_rgb = cv2.imread(img_location + "/" + row_img)  # Read the image.
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)  # Convert it into a gray image.
        image_w, image_h = img_gray.shape[::-1]  # Image dimensions.

        # Construct the path to the templates.
        template_path = os.path.abspath(
            os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'templates'))
        # List all the templates that are not within the 'line_start_templates' subdirectory.
        template_names = [t for t in os.listdir(template_path) if not str(t).startswith("line_start_templates")]

        # (1) Find templates using template matching.
        # Use the 'match_templates' function to get the locations(list),dimensions(list).
        # Also, get list of booleans on whether the templates are recognized by their names (such as clef_g),
        # or if they need to be processed by a conv. network.
        # Replace the values in 'template_names' with names of the found templates
        t_loc, t_dim, t_recognized_list, found_t_names = match_templates(template_names, template_path, img_gray)

        # (2) Get the start and end coordinates of the templates.
        construct_output(indent_level=2, message="Matching the row elements with the templates.")
        templates_start_end = [(x[0], x[0] + t_dim[index][0]) for index, x in enumerate(t_loc)]

        # (3) Save the images in the standard size (200x200 px). Return value only used for visualisation.
        construct_output(indent_level=2, message="Saving found elements in the row {}.".format(row_number))
        slices = save_elements_standard_size(templates_start_end, img_rgb,
                                             image_h, image_w,
                                             img_name, str(row_number),
                                             found_t_names, t_recognized_list
                                             )
        construct_output(indent_level=0, message="Finding individual elements in the saved rows done.")

        # RESULT DRAWING (TESTING).
        # from copy import deepcopy
        # from note_recognition_app.image_processing.img_resizer import ResizeWithAspectRatio
        # from note_recognition_app.image_processing.row_splitter_result_visualizer import generate_result_img
        # Draw the results of (2). Leave for checking purposes.
        # tmp_img = deepcopy(img_rgb)
        # for el in templates_start_end:
        #     cv2.rectangle(tmp_img, (el[0], 20), (el[1], image_h - 20), (255, 0, 255), 2)
        # cv2.imshow('Found elements.', ResizeWithAspectRatio(tmp_img, width=1000))
        # cv2.moveWindow('Found elements.', 0, 200)
        # Draw the results of (3). Leave for checking purposes.
        # result_img = generate_result_img(input_images_individual_elements)
        # cv2.imshow('Final Result.', ResizeWithAspectRatio(result_img, width=500))
        # cv2.moveWindow('Final Result.', 0, 400)
        # cv2.waitKey()


def match_templates(template_list, path, img):
    """ This function searches the image for  templates. If a template if found in the image, it's location
        is saved.

    :param template_list: List containing the names of all the templates that need to be searched for.
    :param path: The path to the said templates.
    :param img: The image that the search is being done upon.
    :return: list, list: Returns lists containing the positions of found templates, list containing the
            sizes of those templates, list of booleans specifying if the templates can be categorized only by.
            their names, and the list names of all the templates found at at a certain iteration.
    """
    t_loc = []  # Locations of the found templates.
    t_dim = []  # Dimensions of the found templates.
    t_recognized_list = []  # List of booleans marking if a template at a specified index is recognized by name.
    t_found_names = []  # Names of found template files.

    for template_name in template_list:  # Iterate through all of the vertical line templates.
        template = cv2.imread(path + "\\" + template_name, 0)  # Read the template from the path.
        # Check if the template will need recognizing in the conv. network(templates that are not determined by name).
        current_template_recognized = check_template_type(template_name)
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)  # Convert the template into a gray image.
        threshold = 0.8  # The threshold to determine whether a part of an image is similar enough to the template.
        locations = np.where(res >= threshold)  # Locations in the image where the template matching found results.
        template_w, template_h = template.shape[::-1]  # Dimensions of the current template.
        # list(zip(*locations[::-1])) -> Match the 'x' and 'y' coordinates into a tuple them and save into a list.
        # Iterate through locations to remove elements already found by previous templates.
        for point in list(zip(*locations[::-1])):
            if len(t_loc) == 0:  # Save the first template matching results without checking.
                t_loc.append(point)
                t_dim.append((template_w, template_h))  # Also save the template dimensions.
                t_recognized_list.append(current_template_recognized)
                t_found_names.append(template_name)
            else:  # Check if 't_locations' already contains the new point +/- (template_width) px, don't add if true.
                template_range = list(range(point[0] - int(template_w / 2), point[0] + int(template_w / 2)))
                intersect = np.intersect1d(list(zip(*t_loc))[0], template_range)
                if len(intersect) == 0:
                    t_loc.append(point)
                    t_dim.append((template_w, template_h))
                    t_recognized_list.append(current_template_recognized)
                    t_found_names.append(template_name)

    # Sort the results from the left side of the image to the right side of the image.
    sort_by_x_coord = sorted(zip(t_loc, t_dim, t_recognized_list, t_found_names), key=itemgetter(0, 0, 0, 0))
    t_loc, t_dim, t_recognized_list, t_found_names = zip(*sort_by_x_coord)

    return t_loc, t_dim, t_recognized_list, t_found_names


def save_elements_standard_size(max_size_templates, img_rgb, image_h, image_w, img_name, row_no, template_names,
                                template_recognized_list):
    """ This function takes the input templates and resizes them to standard dimensions (200 px x 200 px).

    :param max_size_templates: List of templates to be resized.
    :param img_rgb: Original image used to resize templates.
    :param image_h: Original image height.
    :param image_w: Original image width.
    :param img_name: Original image name, used in naming the input_images_individual_elements produced.
    :param row_no: Row number.
    :param template_names: Names of the template files that were found.
    :param template_recognized_list: List that indicates if the templates are recognized just by their name.
    :return: list: A list that contains the position of the slice within the original image.
    """

    dir_name = os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'input_images')

    dir_name = os.path.join(dir_name, 'input_images_individual_elements', img_name[:-4])
    try:  # Try creating a directory.
        construct_output(indent_level=1, message="Creating a folder for the image elements: {}".format(dir_name))
        os.mkdir(dir_name)
    except OSError as e:
        construct_output(indent_level=1, message="Folder already exists: {}".format(dir_name))

    slices = []
    file_reset = False  # Boolean flag to check if a file has been reset when the image of the same name is read.
    for index, el in enumerate(max_size_templates):  # Iterate through the images.
        right_coord = el[1]  # Rightmost coordinate of the image.
        left_coord = el[0]  # Leftmost coordinate of the image.
        img_slice = img_rgb[0:image_h, left_coord:right_coord]  # Get the slice from the original image.

        if abs(right_coord - left_coord) < 200:  # If 'X' coordinate is too small...
            rightmost_column = img_rgb[0:image_h, right_coord - 1:right_coord]  # Get the rightmost column, 1px wide.
            leftmost_column = img_rgb[0:image_h, left_coord:left_coord + 1]  # Get the leftmost column, 1px wide.
            while (img_slice.shape[1]) < 200:  # While the width of the image is not 200 px...
                img_slice = np.concatenate((img_slice, rightmost_column), axis=1)  # Expand to the right by 1 px.
                img_slice = np.concatenate((leftmost_column, img_slice), axis=1)  # Expand to the left by 1 px.
            if (img_slice.shape[1]) == 201:  # If the image was expanded too much...
                img_slice = img_slice[0:image_h, 0:200]  # Crop the image to be 200 px wide.

        elif abs(right_coord - left_coord) >= 200:  # If the 'X' coordinate is too big...
            middle = int(abs(right_coord + left_coord) / 2)  # Find the middle of the current slice.
            if (middle - 100) > 0 and (middle + 100) < image_w:  # If expanding is possible...
                img_slice = img_rgb[0:image_h, middle - 100:middle + 100]  # Crop the image to be 200 px wide.

        if img_slice.shape[1] == 200:  # If the horizontal expansion was successful...

            if img_slice.shape[0] >= 200:  # If the 'Y' coordinate is too big...
                img_slice = img_slice[20:220, 0:img_slice.shape[1]]  # Crop the image to be 200 px tall.

            elif img_slice.shape[0] < 200:  # If the 'Y' coordinate is too small...
                upmost_row = img_rgb[0:1, 0:img_slice.shape[1]]  # Upmost pixel row, 1 px tall.
                downmost_row = img_rgb[image_h - 2:image_h - 1, 0:img_slice.shape[1]]  # Downmost pixel row, 1 px tall.
                while (img_slice.shape[0]) < 200:  # While the height of the image is not 200 px...
                    img_slice = np.concatenate((img_slice, upmost_row), axis=0)  # Expand up by 1 px.
                    img_slice = np.concatenate((downmost_row, img_slice), axis=0)  # Expand down by 1 px
                if (img_slice.shape[0]) == 201:  # If the image was expanded too much...
                    img_slice = img_slice[0:200, 0:img_slice.shape[1]]  # Crop the image to be 200 px tall.

        if img_slice.shape[0] == 200 and img_slice.shape[1] == 200:  # If the slice's dimensions are the right size...
            slice_name = img_name[:-4] + "_row" + row_no + "_" + "slice" + str(index)  # The first part of the name.
            if template_recognized_list[index] is True:  # If the template is recognized by name, add that name.
                slice_name = slice_name + "_" + template_names[index] + ".png"  # Second part of the name.
                # Get the path to the directory that will contain the individual elements that were recognized.
                slice_path = os.path.join(dir_name, "recognized")
                try:  # Make the directory, if it doesn't exist.
                    os.mkdir(slice_path)
                except OSError:
                    pass
                try:
                    file_name = img_name[:-4] + ".txt"  # Name for a file that will contain the recognized elements.
                    # Generate file path
                    file_path = os.path.abspath(os.path.join(str(Path(slice_path)), file_name))
                    # Reset the file if an image with the same name was already read before.
                    if "_row0_" in slice_name and file_reset is False:
                        open(file_path, 'w').close()
                        file_reset = True
                    # Add the new information into the file.
                    construct_output(indent_level=2,
                                     message="Writing the information about a recognized element ({}) into: '{}'"
                                     .format(slice_name, file_path))
                    recognized_templates_file = open(file_path, "a")
                    recognized_templates_file.write(slice_name + "\n")
                    recognized_templates_file.close()
                except OSError:
                    pass
            else:  # Otherwise, write "UNKNOWN" into the image name.
                slice_name = slice_name + "_UNKNOWN" + ".png"  # Second part of the name that were not recognized.
                slice_path = os.path.join(dir_name, "unrecognized")
                try:  # Make the directory, if it doesn't exist.
                    os.mkdir(slice_path)
                except OSError:
                    pass
                slice_path = os.path.abspath(os.path.join(slice_path, slice_name))  # Image write location.
                construct_output(indent_level=2,
                                 message="Saving element: '{}' into: '{}'".format(slice_name, slice_path))
                cv2.imwrite(slice_path, img_slice)  # Write the image.

            slices.append(img_slice)  # Append the positions.

    return slices


def check_template_type(template_name):
    """Function that checks if a template can be recognized fully only by it's name (such as clefs or modulation ... ).
    :param template_name: Name of a template.
    :return: boolean: True if a template can be recognized only by using it's name, false otherwise.
    """
    if template_name.startswith("template_clef_"):
        return True
    elif template_name.startswith("template_hollow_element"):
        return False
    elif template_name.startswith("template_modulation_"):
        return True
    elif template_name.startswith("template_note"):
        return False
    elif template_name.startswith("template_note"):
        return False
    elif template_name.startswith("template_rest"):
        return True
    elif template_name.startswith("template_time_signature"):
        return True
    elif template_name.startswith("template_Z_barline_"):
        return True
    return False
