import cv2
import numpy as np
import os
# TODO - make it into a function that has an arg = path(str)
image_filename = "img02.jpg"                                                         # image Name
img = cv2.imread(image_filename)                                                     # read

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                         # transform into gray img
th, thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)     # pixels have 1 of 2 values
nonZeroElements = cv2.findNonZero(thr)                                               # find the non zero pixels
nonZeroMinArea = cv2.minAreaRect(nonZeroElements)                                    # mark the area with a rectangle

(cx, cy), (w, h), ang = nonZeroMinArea                                               # find rotated matrix
M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
rotated = cv2.warpAffine(thr, M, (img.shape[1], img.shape[0]))                       # do the rotation

#TODO filter out noise

# cv2.imwrite("preborder.jpg",rotated)                                               # expand to fit A4 format
bordered = cv2.copyMakeBorder(rotated,0,int((W*1.414)-H),0,0,cv2.BORDER_CONSTANT,value=[0, 0, 0])
# cv2.imwrite("border.jpg",bordered)

hist = cv2.reduce(rotated, 1, cv2.REDUCE_AVG).reshape(-1)                            # reduce matrix to a vector

th = 2                                                                               # change the threshold (empirical)
H, W = img.shape[:2]                                                                 # picture dimensions

upperBound = [y for y in range(H - 1) if hist[y] <= th < hist[y + 1]]                # upper bounds
lowerBound = [y for y in range(H - 1) if hist[y] > th >= hist[y + 1]]                # lower bounds

up_array = np.asarray(upperBound)                                                    # list to array conversion
up = (H - up_array)
low_array = np.asarray(lowerBound)
low = (H - low_array)

slices = []
for i in range(len(up_array)):                                                     # row slicing
    if(low_array[i] + 1) + int(H/350) < H and up_array[i] - int(H / 70) > 0:       # expand the slice vertically
        h_slice = bordered[up_array[i] - int(H / 70):(low_array[i] + 1) + int(H / 350), 0:W]
    else:                                                                          # don't expand on the edges
        h_slice = bordered[up_array[i]:(low_array[i] + 1), 0:W]
    slices.append(h_slice)                                                         # save all the slices in a list

# on standard A4 paper(21 cm height), 1 note line is 1 cm -> 1/21 ~= 0.04, so ignore smaller slices
slices[:] = [s for s in slices if not len(s) < 0.04 * H]                           # remove unwanted slices

valid_slices_pixel_mean = []                                                       # find the mean value of each slice
for s in slices:
    valid_slices_pixel_mean.append(np.mean(s))
mean = np.mean(valid_slices_pixel_mean)                                            # find the mean value of all slices


j = 0
for i in range(len(slices)):                                                        # save the valid slices
    # wanted slices have approximately the same mean of pixels, ignore the unwanted lines(+- 15% of mean)
    if 1.30 * mean > valid_slices_pixel_mean[i] > 0.70 * mean:
        sliceName = "slice" + str(j) + ".jpg"                                       # slice naming
        path = "resources/" + image_filename[:-4] + "/"                             # directory for the slices
        try:                                                                        # create the dir if it doesn't exist
            os.makedirs(path)
        except FileExistsError:
            pass
        cv2.imwrite(path + sliceName, slices[i])                                    # save the slices in that directory
        j = j + 1                                                                   # name slices iteratively
