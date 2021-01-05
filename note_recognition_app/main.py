import os
import sys
from pathlib import Path

from note_recognition_app.conv_net.conv_net_main import conv_network_analysis
from note_recognition_app.image_processing.image_processing_main_driver import process_image
from note_recognition_app.console_output.console_output_constructor import construct_output
from note_recognition_app.results_generator.generator import generate_results
from note_recognition_app.music_player.music_player_main_driver import play


def main():
    input_image_name = "img02.png"

    # Picture segmentation.
    construct_output(indent_level=-1, message="Input image: {}\n".format(input_image_name))
    input_image_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent), 'resources', 'input_images'))
    input_image_path = os.path.join(input_image_path, input_image_name)
    process_image(input_image_path)

    # # Convolutional network analysis.
    names, durations = conv_network_analysis(input_image_name)
    # print(names)

    # Play the midi file.
    # input_image_name = "img02.png"
    # names = ['G4', 'A4', 'B4', 'G4', 'G4', 'A4', 'B4', 'G4', 'B4', 'C5', 'D5', 'B4', 'C5', 'D5', 'D4', 'D4', 'C5',
    #          'C5', 'B4', 'G4', 'D4', 'D4', 'C5', 'C5', 'B4', 'G4', 'G4', 'D4', 'G4', 'G4', 'D4', 'G4']
    #
    # durations = ['1/4', '1/4', '1/4', '1/4', '1/4', '1/4', '1/4', '1/4', '1/4', '1/4', '1/2', '1/4', '1/4', '1/2',
    #              '1/8', '1/8', '1/8', '1/8', '1/4', '1/4', '1/8', '1/8', '1/8', '1/8', '1/4', '1/4', '1/4', '1/4',
    #              '1/2', '1/4', '1/4', '1/2']

    results = generate_results(input_image_name, names, durations)
    play(results, input_image_name)


if __name__ == '__main__':
    sys.exit(main())
