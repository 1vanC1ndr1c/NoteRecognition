import os
import sys
from pathlib import Path

from note_recognition_app.conv_net.conv_net_main import conv_network_analysis
from note_recognition_app.image_processing.image_processing_main_driver import process_image
from note_recognition_app.console_output.console_output_constructor import construct_output
from note_recognition_app.results_generator.generator import generate_results
from note_recognition_app.midi_handler.midi_constructor import construct_midi


def main():
    input_image_name = "img01.png"
    # Picture segmentation.
    construct_output(indent_level=-1, message="Input image: {}\n".format(input_image_name))
    input_image_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent), 'resources', 'input_images'))
    input_image_path = os.path.join(input_image_path, input_image_name)
    process_image(input_image_path)

    # Convolutional network analysis.
    names, durations = conv_network_analysis(input_image_name)

    # Generate the results format for the midi constructor.
    results = generate_results(input_image_name, names, durations)
    # Construct the midi.
    construct_midi(results, input_image_name)


if __name__ == '__main__':
    sys.exit(main())
