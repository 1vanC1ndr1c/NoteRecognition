import sys

from note_recognition_app.nerual_net.neural_net_main import start_the_networks


def main():
    # img_name = "img02.png"
    # is_success = split_into_rows(img_name)

    # if is_success is True:
    #     extract_elements(img_name)

    # extract_elements(img_name)
    # extract_elements_by_template_matching(img_name)
    start_the_networks()


if __name__ == '__main__':
    sys.exit(main())
