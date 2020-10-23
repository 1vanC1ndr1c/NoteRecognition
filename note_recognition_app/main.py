import sys

from note_recognition_app.image_processing.template_matcher import extract_elements_by_template_matching
from note_recognition_app.image_processing.row_splitter import split_into_rows

def main():
    img_name = "img00.png"
    #is_success = split_into_rows(img_name)

    # if is_success is True:
    #     extract_elements(img_name)

    # extract_elements(img_name)
    extract_elements_by_template_matching(img_name)


if __name__ == '__main__':
    sys.exit(main())
