import cv2
import numpy as np

def crop_image(img):

    h, w = img.shape[:2]  # image dimensions
    img_mean = np.mean(img)  # find the mean of the image

    for i in range(len(img[0])):  # go horizontally; len(img[0]) = no. of columns
        column_mean = 0  # calculate the mean of every column
        for j in range(int(len(img) / 2), int(3 * len(img) / 4), 1):  # star at the middle of a picture
            column_mean = column_mean + np.mean(img[j][i])  # add the means of all the pixels in a column
        column_mean = column_mean / (len(img) / 2)  # divide by the number of elements(rows) in column
        if column_mean > img_mean:  # cut away the spaces before score lines
            img = img[0:h, i:len(img[0])]  # crop the image
            break  # break when done


    for i in range(len(img[0]) - 1, 0, -1):  # go backwards (end to 0, with step being -1)
        column_mean = 0  # calculate the mean of every column
        for j in range(len(img)):
            column_mean = column_mean + np.mean(img[j][i])  # add the means of all the pixels in a column
        column_mean = column_mean / len(img)  # divide by the number of elements(rows) in column
        if column_mean > img_mean:  # cut away the spaces after score lines
            img = img[0:h, 0:i]  # crop the image
            break  # break when done
    return img



def crop_image_again(img):

    h, w = img.shape[:2]  # image dimensions
    img_mean = np.mean(img)  # find the mean of the image
    max_mean = -1
    min_mean = 10000
    max_mean_index_left = -1
    max_mean_index_right = -1

    #TODO 1.cropati sa indexima (prosiriti)
    #TODO 2. CROPATI BEZ
    #TODO 3. HISTOGRAM NA 2.
    #TODO 4. IZVUCI ELEMENTE POMOCU HISTOGRAMA SA 1.


    for i in range(len(img[0])):  # go horizontally; len(img[0]) = no. of columns
        column_mean = 0  # calculate the mean of every column
        for j in range(len(img)):  # star at the middle of a picture
            column_mean = column_mean + np.mean(img[j][i])  # add the means of all the pixels in a column
        column_mean = column_mean / (len(img))  # divide by the number of elements(rows) in column
        if column_mean < min_mean:  # find max
            min_mean = column_mean

    print(min_mean)

    # for i in range(len(img[0])):  # go horizontally; len(img[0]) = no. of columns
    #     column_mean = 0  # calculate the mean of every column
    #     for j in range(len(img)):  # star at the middle of a picture
    #         column_mean = column_mean + np.mean(img[j][i])  # add the means of all the pixels in a column
    #     column_mean = column_mean / (len(img))  # divide by the number of elements(rows) in column
    #     if column_mean >= max_mean:  # find max
    #         max_mean = column_mean
    #         max_mean_index_left = i
    #     else:                       # break after found
    #         break
    #
    # max_mean = -1
    # for i in range(len(img[0]) - 1, 0, -1):  # go backwards (end to 0, with step being -1)
    #     column_mean = 0  # calculate the mean of every column
    #     for j in range(len(img)):  # star at the middle of a picture
    #         column_mean = column_mean + np.mean(img[j][i])  # add the means of all the pixels in a column
    #     column_mean = column_mean / (len(img))  # divide by the number of elements(rows) in column
    #     if column_mean >= max_mean:  # find max
    #         max_mean = column_mean
    #         max_mean_index_right = i
    #     else:  # break after found
    #         break
    #
    # max_mean_index_right = len(img[0]) -1 - max_mean_index_right

    #
    # for i in range(len(img[0])):  # go horizontally; len(img[0]) = no. of columns
    #     column_mean = 0  # calculate the mean of every column
    #     for j in range(int(len(img) / 2), int(3 * len(img) / 4), 1):  # star at the middle of a picture
    #         column_mean = column_mean + np.mean(img[j][i])  # add the means of all the pixels in a column
    #     column_mean = column_mean / (len(img) / 2)  # divide by the number of elements(rows) in column
    #     if column_mean > img_mean:  # cut away the spaces before score lines
    #         img = img[0:h, i:len(img[0])]  # crop the image
    #         break  # break when done
    #
    # for i in range(len(img[0]) - 1, 0, -1):  # go backwards (end to 0, with step being -1)
    #     column_mean = 0  # calculate the mean of every column
    #     for j in range(len(img)):
    #         column_mean = column_mean + np.mean(img[j][i])  # add the means of all the pixels in a column
    #     column_mean = column_mean / len(img)  # divide by the number of elements(rows) in column
    #     if column_mean > img_mean:  # cut away the spaces after score lines
    #         img = img[0:h, 0:i]  # crop the image
    #         break  # break when done



    return img


def erode_dilate(img):
    kernel1 = np.ones((1, 1), np.uint8)  # kernel with dimensions 2x2, 1 if all = 1
    eroded_img = cv2.erode(img, None, iterations=1)  # first erode
    dilated_img = cv2.dilate(eroded_img, None, iterations=1)  # then dilate
    eroded_img = cv2.erode(dilated_img, None, iterations=1)  # first erode
    return eroded_img


def find_histogram(dilated_img):
    min_mean = 10000  # result of the minimum mean of a vertical line

    for i in range(len(dilated_img[0])):  # go horizontally; len(img[0]) = no. of columns
        column_mean = 0  # calculate the mean of every column
        for j in range(len(dilated_img)):
            column_mean = column_mean + np.mean(dilated_img[j][i])  # add the means of all the pixels in a column
        column_mean = column_mean / len(dilated_img)  # divide by the number of elements(rows) in column
        if column_mean < min_mean:  # if it is smaller than the minimum...
            min_mean = column_mean  # ... make it a new minimum

    print(len(dilated_img))
    print(min_mean)

    hist = [0]  # histogram of the picture
    for i in range(len(dilated_img[0])):  # go horizontally; len(img[0]) = no. of columns
        column_mean = 0  # calculate the mean of every column
        for j in range(len(dilated_img)):
            column_mean = column_mean + np.mean(dilated_img[j][i])  # add the means of all the pixels in a column
        column_mean = column_mean / len(dilated_img)  # divide by the number of elements(rows) in column

        if column_mean > min_mean:  # put 1 in a histogram if a line is not empty
            hist.append(1)
        else:  # put 0 in a histogram if a line is empty
            hist.append(0)

    return hist


def get_element_coordinates(dilated_img, hist):
    x_cut_start = []  # coordinates for the left side of the element
    x_cut_end = []  # coordinates for the right side of the element
    for i in range(len(hist)):  # find the edges (rising and falling edge)
        if i > 0:
            if hist[i - 1] == 0 and hist[i] == 1:  # find the starting x coordinate(rising edge)
                x_cut_start.append(i - 1)
            elif hist[i - 1] == 1 and hist[i] == 0:  # find the starting x coordinate(falling edge)
                x_cut_end.append(i - 1)
    x_cut_end.append(len(dilated_img[0] - 1))  # last coordinate is the end of the picture
    return x_cut_start, x_cut_end


def get_elements_from_image(path, x_cut_start, x_cut_end, img, element_number):
    large_elements = []
    large_elements_index = []
    h, w = img.shape[:2]  # image dimensions
    for i in range(len(x_cut_start)):
        if x_cut_start[i] - 3 > 0 and x_cut_end[i] + 3 < w - 1:
            element = img[0:h, x_cut_start[i] - 3:x_cut_end[i] + 3]  # cut the element from the image
        else:
            element = img[0:h, x_cut_start[i]:x_cut_end[i]]
        element_name = "el" + str(element_number).zfill(5) + ".jpg"  # generate the element name
        if 5 < len(element[0]) < 75:  # if the element is valid
            try:  # if the element is not null
                cv2.imwrite(path + element_name, element)  # save the elements in the directory
            except:  # else, skip that element
                pass
        elif len(element[0]) >= 75:
            large_elements.append(element)
            large_elements_index.append(element_number)
        element_number = element_number + 1  # increase the indexing number

    index = 0
    for el in large_elements:
        changed_el = erode_dilate(el)
        el = crop_image_again(el)
        cv2.imshow('ch', el)
        cv2.waitKey(0)
        hist = find_histogram(changed_el)
        x_cut_start, x_cut_end = get_element_coordinates(changed_el, hist)

        h, w = el.shape[:2]  # image dimensions
        for i in range(len(x_cut_start)):
            if x_cut_start[i] - 3 > 0 and x_cut_end[i] + 3 < w - 1:
                element = el[0:h, x_cut_start[i] - 3:x_cut_end[i] + 3]  # cut the element from the image
            else:
                element = el[0:h, x_cut_start[i]:x_cut_end[i]]
            element_name = "el" + str(large_elements_index[index]).zfill(5) +\
                           "part" + str(i) + ".jpg"         # generate the element name
            if 4 < len(element[0]):
                try:  # if the element is not null
                    cv2.imwrite(path + element_name, element)  # save the elements in the directory
                except:  # else, skip that element
                    pass
        index = index + 1

    return element_number
