import sys

from note_recognition_app.image_processing.row_splitter import split_into_rows


def main():
    img_name = "img01.png"
    is_sucess = split_into_rows(img_name)

    if is_sucess is True:
        print("YOLO")


if __name__ == '__main__':
    sys.exit(main())
