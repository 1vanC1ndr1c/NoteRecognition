import cv2
import os
from note_recognition_app.image_processing.img_resizer import ResizeWithAspectRatio

from pathlib import Path


def split_into_rows(img_name):
    """
        Code from: https://stackoverflow.com/questions/34981144/split-text-lines-in-scanned-document
    """

    try:
        # (1) read
        dir_name = "resources/" + img_name[:-4]
        try:
            os.mkdir(dir_name)
        except OSError as e:
            pass

        path = os.path.join(os.getcwd(), 'resources', img_name)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        #  (1.1) remove transparency
        trans_mask = img[:, :, 3] == 0
        img[trans_mask] = [255, 255, 255, 255]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # (2) threshold
        th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # (3) minAreaRect on the nozeros
        pts = cv2.findNonZero(threshed)
        ret = cv2.minAreaRect(pts)

        (cx, cy), (w, h), ang = ret
        if abs(ang) < 1:
            ang = 0
        # if w > h: # TODO handle images of the wrong size
        #     w, h = h, w
        #     ang += 90

        # (4) Find rotated matrix, do rotation
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))

        # (5) find and draw the upper and lower boundary of each lines
        hist = cv2.reduce(rotated, 1, cv2.REDUCE_AVG).reshape(-1)

        th = 2
        H, W = img.shape[:2]

        uppers = [y for y in range(H - 1) if hist[y] <= th < hist[y + 1]]
        lowers = [y for y in range(H - 1) if hist[y] > th >= hist[y + 1]]

        rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
        for y in uppers:
            cv2.line(rotated, (0, y), (W, y), (255, 0, 0), 1)

        for y in lowers:
            cv2.line(rotated, (0, y), (W, y), (0, 255, 0), 1)

        cv2.imwrite(dir_name + "/cutting_lines.png", rotated)

        # cut the image in the uppers and lowers range
        # TODO check if len(uppers) == len(lowers)

        for index in range(len(uppers)):
            start_cut_coord = uppers[index]
            end_cut_coord = lowers[index]

            img_slice = rotated[start_cut_coord:end_cut_coord, 0:len(rotated[0])]

            if len(img_slice) >= 10:
                img_slice_name = dir_name + "/row" + str(index) + ".png"
                cv2.imwrite(img_slice_name, img_slice)
    except:
        return False
    return False
