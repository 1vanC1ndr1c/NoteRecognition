import os
import sys
from pathlib import Path

import multiprocessing
from note_recognition_app.conv_net.conv_net_main import conv_network_analysis
from note_recognition_app.image_processing.image_processing_main_driver import process_image
from note_recognition_app.console_output.console_output_constructor import construct_output
from note_recognition_app.results_generator.generator import generate_results
from note_recognition_app.midi_handler.midi_constructor import construct_midi
from note_recognition_app.gui.main_window import run_gui


def main():
    queue_foreground_to_background = multiprocessing.Queue()
    queue_background_to_foreground = multiprocessing.Queue()
    foreground_process = multiprocessing.Process(target=foreground, args=(queue_foreground_to_background,
                                                                          queue_background_to_foreground))

    background_process = multiprocessing.Process(target=background, args=(queue_foreground_to_background,
                                                                          queue_background_to_foreground))

    foreground_process.start()
    background_process.start()

    foreground_process.join()
    background_process.join()


def foreground(queue_foreground_to_background, queue_background_to_foreground):
    queue_foreground_to_background.put('TEST1')
    run_gui(queue_foreground_to_background, queue_background_to_foreground)


def background(queue_foreground_to_background, queue_background_to_foreground):
    print('TODO2')
    print(queue_foreground_to_background.get())

    # # todo path
    # input_image_name = "img01.png"
    # # Picture segmentation.
    # construct_output(indent_level=-1, message="Input image: {}\n".format(input_image_name))
    # input_image_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent), 'resources', 'input_images'))
    # input_image_path = os.path.join(input_image_path, input_image_name)
    # process_image(input_image_path)
    #
    # # Convolutional network analysis.
    # names, durations = conv_network_analysis(input_image_name, retrain_flag=False)
    #
    # # Generate the results format for the midi constructor.
    # results = generate_results(input_image_name, names, durations)
    # # Construct the midi.
    # construct_midi(results, input_image_name)


if __name__ == '__main__':
    sys.exit(main())
