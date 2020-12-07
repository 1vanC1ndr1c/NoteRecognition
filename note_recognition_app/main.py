import os
import sys
from pathlib import Path

from note_recognition_app.conv_net.conv_net_main import conv_network_analysis
from note_recognition_app.image_processing.image_processing_main_driver import process_image
from note_recognition_app.info_output.output_constructor import construct_output


def main():
    input_image_name = "img01.png"
    # Picture segmentation.
    # construct_output(indent_level=-1, message="Input image: {}\n".format(input_image_name))
    # input_image_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent), 'resources', 'input_images'))
    # input_image_path = os.path.join(input_image_path, input_image_name)
    # process_image(input_image_path)

    # Convolutional network analysis.
    conv_network_analysis(input_image_name)


if __name__ == '__main__':
    sys.exit(main())
