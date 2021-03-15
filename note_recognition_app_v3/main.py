import os
import sys
from pathlib import Path

# from note_recognition_app_v2.conv_net.conv_net_main import conv_network_analysis
from note_recognition_app_v2.image_segmentation_dataset_generator import main
from note_recognition_app_v2.console_output.console_output_constructor import construct_output

# from note_recognition_app_v2.results_generator.generator import generate_results
# from note_recognition_app_v2.midi_handler.midi_constructor import construct_midi
from note_recognition_app_v2.positions_detection.element_detector import detect_elements


def main():
    # input_image_name = "img01.png"
    # # Picture segmentation.
    # construct_output(indent_level=-1, message="Input image: {}\n".format(input_image_name))
    # input_image_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent), 'resources', 'input_images'))
    # input_image_path = os.path.join(input_image_path, input_image_name)

    detect_elements('img02', retrain_flag=False)

    # # Convolutional network analysis.
    # names, durations = conv_network_analysis(input_image_name)
    #
    # # Generate the results format for the midi constructor.
    # results = generate_results(input_image_name, names, durations)
    # # Construct the midi.
    # construct_midi(results, input_image_name)


if __name__ == '__main__':
    sys.exit(main())
