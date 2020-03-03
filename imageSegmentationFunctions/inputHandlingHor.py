import cv2
import numpy as np
from pathlib import Path
import os


def input_handling_hor(file_name):
    # This def gets as an input an image ('.jpg' or '.png' and produces a number of images corresponding to the
    # number of horizontal lines that contain notes in the original (input) image.

    if file_name.endswith(".png"):                                                    # if .png, convert to .jpg
        img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        b, g, r, a = cv2.split(img)
        new_img = cv2.merge((b, g, r))
        not_a = cv2.bitwise_not(a)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)
        new_img = cv2.bitwise_and(new_img, new_img, mask=a)
        new_img = cv2.add(new_img, not_a)
        img = new_img

    else:
        img = cv2.imread(file_name)                                                    # read

    file_name = file_name.split('\\')                                                  # extract only the image name
    tmp = file_name[len(file_name) - 1]
    file_name = tmp                                                                    # save it back into file_name

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                       # transform into gray img
    th, thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)   # pixels have 1 of 2 values
    non_zero_elements = cv2.findNonZero(thr)                                           # find the non zero pixels
    non_zero_min_area = cv2.minAreaRect(non_zero_elements)                             # mark the area with a rectangle

    (cx, cy), (w, h), ang = non_zero_min_area                                          # find rotated matrix
    if abs(ang) < 1:                                                                   # ignore small rot. adjustment
        ang = 0
    m = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rotated = cv2.warpAffine(thr, m, (img.shape[1], img.shape[0]))                     # do the rotation

    h, w = rotated.shape[:2]                                                           # picture dimensions
    # expand to fit A4 format, if the image is shorter than a4
    if (int(w*1.414 - h)) > 1:
        bordered = cv2.copyMakeBorder(rotated, 0, int((w*1.414 - h)), 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # make the image a standard size
    else:
        bordered = rotated

    hist = cv2.reduce(bordered, 1, cv2.REDUCE_AVG).reshape(-1)                          # reduce matrix to a vector

    th = 2                                                                             # number found experimentally
    h, w = bordered.shape[:2]                                                          # picture dimensions

    upper_bound = [y for y in range(h - 1) if hist[y] <= th < hist[y + 1]]             # upper bounds
    lower_bound = [y for y in range(h - 1) if hist[y] > th >= hist[y + 1]]             # lower bounds

    up_array = np.asarray(upper_bound)                                                 # list to array conversion
    low_array = np.asarray(lower_bound)                                                # list to array conversion

    slices = []
    for i in range(len(up_array)):                                                     # row slicing
        if(low_array[i] + 1) + int(h/350) < h and up_array[i] - int(h / 70) > 0:       # expand the slice vertically
            h_slice = bordered[up_array[i] - int(h / 70):(low_array[i] + 1) + int(h / 350), 0:w]
        else:                                                                          # don't expand on the edges
            h_slice = bordered[up_array[i]:(low_array[i] + 1), 0:w]
        slices.append(h_slice)                                                         # save all the slices in a list

    # on standard A4 paper(21 cm height), 1 note line is 1 cm -> 1/21 ~= 0.04, so ignore smaller slices
    slices[:] = [s for s in slices if not len(s) < 0.04 * h]                           # remove unwanted slices

    valid_slices_pixel_mean = []                                                       # mean value of each slice
    for s in slices:
        valid_slices_pixel_mean.append(np.mean(s))
    mean = np.mean(valid_slices_pixel_mean)                                            # mean value of all slices

    j = 0
    path = 'NaN'
    for i in range(len(slices)):                                                       # save the valid slices
        # wanted slices have approximately the same mean of pixels, ignore the unwanted lines(+- 15% of mean)
        if 1.30 * mean > valid_slices_pixel_mean[i] > 0.70 * mean:
            slice_name = "slice" + str(j) + ".jpg"                                  # slice naming
            parent = str(Path(__file__).parent.parent)                              # save into parent of parent
            path = parent + "\\resources\\" + file_name[:-4] + "\\"                 # directory for the slices
            try:                                                                    # create the dir if it doesn't exist
                os.makedirs(path)
            except FileExistsError:
                pass
            cv2.imwrite(path + slice_name, slices[i])                               # save the slices in that directory
            j = j + 1                                                               # name slices iteratively

    return path                                                                     # return path to slices
