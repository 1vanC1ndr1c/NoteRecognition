import os
import sys
from pathlib import Path

# from note_recognition_app.neural_net.neural_net_main import start_the_networks
from note_recognition_app.image_processing.image_processing_main_driver import process_image


def main():
    input_image_name = "img06.png"
    input_image_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent), 'resources', 'input_images'))
    input_image_path = os.path.join(input_image_path, input_image_name)
    process_image(input_image_path)

    # # if is_success is True:
    # #     extract_elements(img_name)
    #
    # # extract_elements(img_name)
    # # extract_elements_by_template_matching(img_name)
    # start_the_networks()


if __name__ == '__main__':
    sys.exit(main())
