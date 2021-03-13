import multiprocessing
import os
import sys
from pathlib import Path

from note_recognition_app.console_output.console_output_constructor import construct_output
from note_recognition_app.console_output.stdout_redirect import StreamRedirect
from note_recognition_app.conv_net.conv_net_main import conv_network_analysis
from note_recognition_app.gui.main_window import run_gui
from note_recognition_app.image_processing.image_processing_main_driver import process_image
from note_recognition_app.midi_handler.midi_constructor import construct_midi
from note_recognition_app.results_generator.generator import generate_results


def main():
    # Queue where all the stdout messages will be redirected.
    queue_stdout = multiprocessing.Queue()
    # Queue for GUI -> backend communication.
    queue_foreground_to_background = multiprocessing.Queue()
    # Queue for backend -> GUI communication.
    queue_background_to_foreground = multiprocessing.Queue()

    # Define the GUI process.
    foreground_process = multiprocessing.Process(target=foreground, args=(queue_foreground_to_background,
                                                                          queue_background_to_foreground,
                                                                          queue_stdout))

    # Define the backend process.
    background_process = multiprocessing.Process(target=background, args=(queue_foreground_to_background,
                                                                          queue_background_to_foreground,
                                                                          queue_stdout))

    # Start the processes.
    foreground_process.start()
    background_process.start()

    foreground_process.join()
    background_process.join()


def foreground(queue_foreground_to_background, queue_background_to_foreground, queue_stdout):
    # Call on 'run_gui' function that starts the GUI.
    run_gui(queue_foreground_to_background, queue_background_to_foreground, queue_stdout)


def background(queue_foreground_to_background, queue_background_to_foreground, queue_stdout):
    # Redirect the stdout into own class that sends it into queue given in the constructor.
    sys.stdout = StreamRedirect(queue_stdout)

    while True:  # Endless loop.
        # Get the message from GUI with the name to the image (or a message indicating the end of the program).
        gui_input_image_path, retrain_flag = queue_foreground_to_background.get()
        if gui_input_image_path != 'End.':  # 'End.' is sent when the GUI is closed.
            try:
                # Get the image name from the given path.
                input_image_name = gui_input_image_path.split('/')[-1]

                # Picture segmentation.
                construct_output(indent_level=-1, message="Input image: {}\n".format(input_image_name))

                # Transform it into OS depended path.
                input_image_path = os.path.abspath(
                    os.path.join(str(Path(__file__).parent.parent), 'resources', 'input_images'))
                input_image_path = os.path.join(input_image_path, input_image_name)

                # Process the images to get the individual elements.
                process_image(input_image_path)

                # Convolutional network analysis.
                names, durations = conv_network_analysis(input_image_name, retrain_flag=retrain_flag)

                # Generate the results format for the midi constructor.
                results = generate_results(input_image_name, names, durations)
                # Construct the midi.
                construct_midi(results, input_image_name)

                # Send the information about the successful operation.
                queue_background_to_foreground.put('Success.')
            except:
                # Catch errors.
                queue_background_to_foreground.put('ERROR! Check logs file.')
        else:
            break


if __name__ == '__main__':
    sys.exit(main())
