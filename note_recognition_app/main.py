import sys

from note_recognition_app.nerual_net.main_driver import start_neural_net


def main():
    # img_name = "img02.png"
    # is_success = split_into_rows(img_name)

    # if is_success is True:
    #     extract_elements(img_name)

    # extract_elements(img_name)
    # extract_elements_by_template_matching(img_name)
    start_neural_net()


if __name__ == '__main__':
    sys.exit(main())
