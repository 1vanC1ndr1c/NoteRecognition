import sys

from note_recognition_app.image_processing.row_splitter import split_into_rows
from note_recognition_app.image_processing.extract_elements import extract_elements


def main():
    img_name = "img01.png"
    # is_success = split_into_rows(img_name)

    # if is_success is True:
    #     extract_elements(img_name)

    extract_elements(img_name)


if __name__ == '__main__':
    sys.exit(main())
