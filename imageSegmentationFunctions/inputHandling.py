from imageSegmentationFunctions.inputHandlingHor import input_handling_hor
from imageSegmentationFunctions.inputHandlingVer import input_handling_ver


def input_handling(path):
    new_path = input_handling_hor(path)
    input_handling_ver(new_path)
