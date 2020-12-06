import cv2

"""
    Code found here: https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
"""


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Function that resizes the image and keeps the original aspect ration.

    :param image: Input image that needs to be resized.
    :param width: Wanted width. (If this is not None, no need to include height)
    :param height: Wanted height. (If this is not None, no need to include width)
    :param inter: Interpolation method.
    :return: cv2.image: Resized image.
    """
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)
