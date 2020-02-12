import cv2
import os
from inputHandlingHorizontalFunctions import crop_image
from inputHandlingHorizontalFunctions import erode_dilate
from inputHandlingHorizontalFunctions import find_histogram
from inputHandlingHorizontalFunctions import get_element_coordinates
from inputHandlingHorizontalFunctions import get_elements_from_image
# TODO - make it into a function that has an arg = path(str)
# TODO - try to compare to min instead of mean


path = ".\\resources\\img02\\"                                         # image location

images = []                                                            # get all the image names in the directory
for r, d, f in os.walk(path):                                          # r=root, d=directories, f = files
    for file in f:
        if '.jpg' in file:
            if 'el' not in file:
                images.append(file)


elementNumber = 0                                                     # indexing number for extracted elements
for image_name in images:                                             # process every slice
    img = cv2.imread(path + image_name)                               # read the image
    img = cv2.bitwise_not(img)  # convert colors
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # transform into gray img
    th, thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # pixels have 1 of 2 values
    img = crop_image(thr)
    changed_img = erode_dilate(img)
    hist = find_histogram(changed_img)
    x_cut_start, x_cut_end = get_element_coordinates(changed_img, hist)
    elementNumber = get_elements_from_image(path, x_cut_start, x_cut_end, img, elementNumber)


# for fileName in os.listdir(path):                                   # delete redundant images from the previous step
#     if fileName.startswith("slice"):
#         os.remove(os.path.join(path, fileName))


