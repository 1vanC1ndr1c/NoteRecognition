import cv2
import numpy as np


def crop_image(img):
    # The def is used in the first pass of image cropping process.
    # It finds the mean of every column starting from the left side.
    # When iterating, if a mean of the current column is greater then the average of all the means, a symbol has been
    # found and the cropping stops at that location.
    # The procedure is then repeated from the right side.
    # To ensure that no symbols above or below the note line interfere with the process, only the means of roughly the
    # middle rows are taken into account.
    h, w = img.shape[:2]                                                # image dimensions
    img_mean = np.mean(img)                                             # find the mean of the image
    img_column_means = np.mean(img, axis=0)                             # calculate all the column means in the image

    for i in range(len(img[0])):                                        # go horizontally; len(img[0]) = no. of columns
        column_mean = 0                                                 # calculate the mean of every column
        for j in range(int(len(img) / 2), int(3 * len(img) / 4), 1):    # star at the middle of a picture
            column_mean = column_mean + np.mean(img[j][i])              # add the means of all the pixels in a column
        column_mean = column_mean / (len(img) / 2)                      # divide by the number of rows in column
        if column_mean > img_mean:                                      # cut away the spaces before score lines
            img = img[0:h, i:len(img[0])]                               # crop the image
            break                                                       # break when done

    for i in range(len(img_column_means) - 1, 0, -1):                   # go backwards (end to 0, with step being -1)
        column_mean = img_column_means[i]                               # calculate the mean of every column
        if column_mean > img_mean:                                      # cut away the spaces after score lines
            img = img[0:h, 0:i]                                         # crop the image
            break                                                       # break when done
    return img                                                          # return the cropped image


def crop_image_again(removed_noise_img, org_img):
    # The def crops the image in two different ways:
    # 1. crop the image from both sides so that everything up to the first local maximum is removed.
    # 2. crop the image only from the side which has the local maximum closer to it.
    # The 2. is needed because of the way that  connected 8th notes when have their means calculated.
    # Cropping without checking (1.) leads to the last note (depending on the vertical orientation) being cut.
    # The 2. crop insures that last note is not cut. But still, for the needs of a histogram we need the correctly cut
    # version of the image.
    # So, the histogram is calculated with the image cut from the both sides, and then the coordinates found are applied
    # to the image cut only from one side (coordinates = locations where the image will be cut).
    # This method insures that all the 8th notes are preserved and correctly saved as separate .jpg files.
    # This method is used only on the second pass - when the only images left to cut are the ones with
    # elements connected.

    h, w = removed_noise_img.shape[:2]                             # image dimensions
    img_mean = np.mean(removed_noise_img)                          # find the mean of the image
    img_column_means = np.mean(removed_noise_img, axis=0)          # calculate all the column means in the image

    # find the left local maximum
    found_left = False                                              # flag that indicates if the left max is found
    found_right = False                                             # flag that indicates if the right max is found
    max_mean_left = -1                                              # value of the left local maximum
    max_mean_right = -1                                             # value of the right local maximum
    max_mean_index_left = -1                                        # index of the left local maximum
    max_mean_index_right = -1                                       # index of the right local maximum
    it_right = len(img_column_means) - 1                            # right iterator starts from the end of the columns
    for it_left, column_mean_left in enumerate(img_column_means):   # iterate through column means

        # get the means from the left side
        if column_mean_left >= max_mean_left and found_left is False:   # find the left local max
            max_mean_left = column_mean_left                            # save left local max
            max_mean_index_left = it_left                               # save the index of said left local maximum
        else:                                                           # stop searching for the left max  when found
            found_left = True

        # get the means from the right side
        it_right = it_right - 1                                             # calculate the right side iterator
        column_mean_right = img_column_means[it_right]                      # get the right side mean

        if column_mean_right >= max_mean_right and found_right is False:    # find the right local max
            max_mean_right = column_mean_right                              # save right local max
            max_mean_index_right = it_right                                 # save the index of said right local maximum
        else:                                                               # stop searching for  right max when found
            found_right = True

        if found_left is True and found_right is True:                      # if both maximums found, stop the search
            break

    # get the distance of the right maximum from the right edge
    max_mean_index_right = len(img_column_means) - 1 - max_mean_index_right

    for i, column_mean in enumerate(img_column_means):                  # go horizontally; len(img[0]) = no. of columns
        if column_mean > img_mean:                                      # cut away the spaces before score lines
            if max_mean_index_left < max_mean_index_right:              # if max is closer on the left , crop left
                # crop the original image only if the local maximum is on the left side
                org_img = org_img[0:h, i:len(org_img[0])]
            # make a cropped version of the removed noise image
            removed_noise_img = org_img[0:h, i:len(org_img[0])]
            break                                                       # break when done

    for i in range(len(img_column_means) - 1, 0, -1):                   # go backwards (end to 0, with step being -1)
        column_mean = img_column_means[i]                               # calculate the mean of every column
        if column_mean > img_mean:                                      # cut away the spaces after score lines
            if max_mean_index_left > max_mean_index_right:              # if max is closer on the right , crop right
                # crop the original image only if the local maximum is on the right side
                org_img = org_img[0:h, 0:i]
            # make a cropped version of the removed noise image
            removed_noise_img = org_img[0:h, 0:i]
            break                                                       # break when done

    return org_img, removed_noise_img,                                  # return one-side crop and both-side crop


def erode_dilate(img):
    # The def filters out the noise from the input image.
    # Firstly, the image is eroded and thus removing all the small elements.
    # The rest of the elements are then dilated to make them bigger and then they are eroded again.
    # That usually leads to the removal of all the horizontal lines and leaves behind only the notes and wanted symbols.

    # kernel1 = np.ones((1, 1), np.uint8)  # not used right now, maybe needed in the future
    eroded_img = cv2.erode(img, None, iterations=1)                     # first erode
    dilated_img = cv2.dilate(eroded_img, None, iterations=1)            # then dilate
    eroded_img = cv2.erode(dilated_img, None, iterations=1)             # erode again
    return eroded_img


def find_histogram(dilated_img):
    # Def finds the binary histogram of the image.
    # The histogram is a binary list (only has '1's and '0's) that indicates if a pixel column of an image
    # has a mean greater of lesser( and equal) than the minimum column mean,
    # which is usually a column of all black pixels.
    # That basically indicates the positions of wanted symbols within the image (for example 011111111000).
    # Those symbols will later be cut out.
    img_column_means = np.mean(dilated_img, axis=0)                     # calculate all the column means in the image
    min_mean = min(img_column_means)                                    # find the minimum of all the column means
    hist = np.where(img_column_means > min_mean, 1, 0).tolist()         # ternary operator that produces binary list
    hist.insert(0, 0)                                                   # first value is always 0
    return hist                                                         # return the binary histogram list


def get_element_coordinates(dilated_img, hist):
    # The def receives a histogram and an image as an input.
    # The contents of the histogram are explained in 'def find_histogram(dilated_img):'
    # That histogram is being iterated through and when a change is detected, change being '1' -> '0' or '0' -> '1',
    # that index is saved in a list.
    # If the detected change is '0' -> '1', that means the start of the symbol is found.
    # If the detected change is '1' -> '0', that means the end of the symbol is found.

    x_cut_start = []                                         # coordinates for the left side of the element cut
    x_cut_end = []                                           # coordinates for the right side of the element cut
    for i in range(len(hist)):                               # find the edges (rising and falling edge) in the histogram
        if i > 0:
            if hist[i - 1] == 0 and hist[i] == 1:            # find the starting x coordinate(rising edge)
                x_cut_start.append(i - 1)                    # save the index
            elif hist[i - 1] == 1 and hist[i] == 0:          # find the starting x coordinate(falling edge)
                x_cut_end.append(i - 1)                      # save the index
    x_cut_end.append(len(dilated_img[0] - 1))                # last coordinate is the end of the picture

    return x_cut_start, x_cut_end                            # return starting and ending coordinates


def get_elements_from_image(path, x_cut_start, x_cut_end, img, element_number):
    # The def gets the coordinates of the symbols in the image. Those symbols are then cut out and saved.
    # If a part of the image that is cut out is greater than 75 pixels, that usually means that there are more
    # symbols  in that part of the image.
    # In that case, the whole procedure of detecting and cutting out symbols from an image is repeated on that part.
    # Element number is an iterating variable that is used to save the cut parts in the correct order within the dir.
    # In the case of the large elements, their element number variable is also saved so that when they are stored in
    # a directory, they are still stored in the correct location.
    # Normal images are saved as el*ELEMENT_NUMBER(5 characters wide)*.jpg
    # Large images are split and then saved as el*ELEMENT_NUMBER(5 characters wide)*part*NUMBER*.jpg

    large_elements = []                                                 # parts of the image that need to be cut further
    large_elements_index = []                                           # the corresponding indexes of the large parts
    h, w = img.shape[:2]                                                # image dimensions

    for i in range(len(x_cut_start)):                                   # iterate through the 'cut coordinates'
        # make the crop area a bit bigger if the coordinates are not at the start or the end of the image
        if x_cut_start[i] - 3 > 0 and x_cut_end[i] + 3 < w - 1:
            element = img[0:h, x_cut_start[i] - 3:x_cut_end[i] + 3]     # cut the element from the image
        # if the coordinates are the start or the end of the image, cut the minimum required area
        else:
            element = img[0:h, x_cut_start[i]:x_cut_end[i]]

        element_name = "el" + str(element_number).zfill(5) + ".jpg"     # generate the element name
        if 5 < len(element[0]) < 75:                                    # if the element is not too small or too big
            try:                                                        # if the element is not null
                cv2.imwrite(path + element_name, element)               # save the element in the directory
            except:                                                     # else, skip that element
                pass
        elif len(element[0]) >= 75:                                     # if the element is too big
            large_elements.append(element)                              # save it into large_elements list
            large_elements_index.append(element_number)                 # and save it's index
        element_number = element_number + 1                             # increase the indexing number

    large_part_index = 0                                                # additional index used in splitting large imgs
    for el in large_elements:                                           # iterate over elements in large elements
        # firstly, remove noise from the image(erode + dilate + erode)
        removed_noise_element = erode_dilate(el)
        # get both versions of the image (see def)
        one_side_crop_org_img, both_side_crop = crop_image_again(removed_noise_element, el)
        # find the histogram of remove_noise_image cropped from both sides
        hist = find_histogram(both_side_crop)
        # find the cut coordinates of remove_noise_image cropped from both sides
        x_cut_start, x_cut_end = get_element_coordinates(both_side_crop, hist)
        # replace the original element with the half - cropped one (half crop explained in def crop_image_again())
        el = one_side_crop_org_img

        h, w = el.shape[:2]                                                         # image dimensions
        for i in range(len(x_cut_start)):
            if x_cut_start[i] - 3 > 0 and x_cut_end[i] + 3 < w - 1:
                element = el[0:h, x_cut_start[i] - 3:x_cut_end[i] + 3]              # cut the element from the image
            else:
                element = el[0:h, x_cut_start[i]:x_cut_end[i]]
            element_name = "el" + str(large_elements_index[large_part_index]).zfill(5) +\
                           "part" + str(i) + ".jpg"         # generate the element name
            if 4 < len(element[0]):
                try:  # if the element is not null
                    cv2.imwrite(path + element_name, element)  # save the elements in the directory
                except:  # else, skip that element
                    pass
        large_part_index = large_part_index + 1

    return element_number
