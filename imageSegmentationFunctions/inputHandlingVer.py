import cv2
import os
from imageSegmentationFunctions.inputHandlingVerFunctions import crop_image
from imageSegmentationFunctions.inputHandlingVerFunctions import erode_dilate
from imageSegmentationFunctions.inputHandlingVerFunctions import find_histogram
from imageSegmentationFunctions.inputHandlingVerFunctions import get_element_coordinates
from imageSegmentationFunctions.inputHandlingVerFunctions import get_elements_from_image


def input_handling_ver(path):
    # This function gets slices of the original image (rows) and cuts out individual elements.
    # The input into function is the path to the image.
    # After the processing is done, original images (images from the first step = vertical segmentation) are deleted.

    images = []                                                             # get all the image names in the directory
    for r, d, f in os.walk(path):                                           # r=root, d=directories, f = files
        for file in f:                                                      # find all the files in the dir
            if '.jpg' in file:                                              # include only '.jpg' ones
                if 'el' not in file:                                        # files starting with 'el' are cut symbols
                    images.append(file)                                     # append the images into a list

    element_number = 0                                                      # indexing number for extracted elements

    for image_name in images:                                               # process every slice(image)
        img = cv2.imread(path + image_name)                                 # read the image
        img = cv2.bitwise_not(img)                                          # convert colors
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                        # transform into gray img
        th, thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # pixels have 1 of 2 values
        img = crop_image(thr)                                               # crop the left and right side of the image
        changed_img = erode_dilate(img)                                     # erode and dilate to filter out noise
        hist = find_histogram(changed_img)                                  # find the histogram of symbol occurrences
        # find the start and end coordinates of the symbols
        x_cut_start, x_cut_end = get_element_coordinates(changed_img, hist)
        # get the updated element number and cut out all the symbols into separate images
        element_number = get_elements_from_image(path, x_cut_start, x_cut_end, img, element_number)

    # for fileName in os.listdir(path):                                 # delete redundant images (from the previous step)
    #     if fileName.startswith("slice"):
    #         os.remove(os.path.join(path, fileName))
