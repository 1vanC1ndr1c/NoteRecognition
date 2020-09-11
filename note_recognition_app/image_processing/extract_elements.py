import cv2
import numpy as np

from note_recognition_app.image_processing.img_resizer import ResizeWithAspectRatio


def extract_elements(img_name):
    # (1) Crop the black areas left and right of the image.
    row_img = cv2.imread("resources/img01/row1.png")

    h, w = row_img.shape[:2]
    img_mean = np.mean(row_img)

    for i in range(len(row_img[0])):  # go from the left edge forwards
        column_mean = 0
        for j in range(len(row_img)):
            column_mean = column_mean + np.mean(row_img[j][i])
        column_mean = column_mean / len(row_img)
        if column_mean > img_mean:
            row_img = row_img[0:h, i:len(row_img[0])]
            break

    img_column_means = np.mean(row_img, axis=0)

    for i in range(len(img_column_means) - 1, 0, -1):  # go backwards (end to 0, with step being -1)
        column_mean = np.mean(img_column_means[i])
        if column_mean > img_mean:
            row_img = row_img[0:h, 0:i]
            break

    # (2) Extract elements ###########################################################################################

    img_mean = np.mean(row_img)

    line_row_img = row_img

    for i in range(len(row_img[0])):  # go from the left edge forwards
        column_mean = 0
        for j in range(len(row_img)):
            column_mean = column_mean + np.mean(row_img[j][i])

        column_mean = column_mean / len(row_img)

        if column_mean < img_mean:
            cv2.line(line_row_img, (i, 0), (i, len(row_img)), (122, 122, 122), 1)
            # row_img = row_img[0:h, i:len(row_img[0])]
            # break

    cv2.imshow('contour', ResizeWithAspectRatio(line_row_img, width=1280))
    cv2.waitKey()
