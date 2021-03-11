from note_recognition_app.console_output.console_output_constructor import construct_output
from note_recognition_app.conv_net import duration_processing_conv_net
from note_recognition_app.conv_net import value_processing_conv_net
from note_recognition_app.conv_net.input_data_processing import prepare_new_data

import tensorflow as tf


def conv_network_analysis(input_image_name, retrain_flag=False):
    """
    Main function for convolutional network analysis.
    Calls on the network for value recognizing.
    Calls on the network for duration analyzing.
    :param input_image_name: Name of the image that is being analyzed.
    :param retrain_flag: Flag that indicates if the retraining is needed.
    """
    construct_output(indent_level="block",
                     message="Analyzing the elements of the image ({}) with a convolutional network."
                     .format(input_image_name))

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    if retrain_flag is True:
        # Import the dataset and split it to training and testing.
        (test_arr, test_label), (train_arr, train__label) = prepare_new_data(test_data_percentage=0)
        value_processing_conv_net.train_note_values_conv_net(test_arr, test_label, train_arr, train__label)
        duration_processing_conv_net.train_note_duration_conv_net(test_arr, test_label, train_arr, train__label)

    # Load the trained data from the disk.
    value_names = value_processing_conv_net.analyze_using_saved_data(input_image_name)
    # Load the trained data from the disk.
    durations = duration_processing_conv_net.analyze_using_saved_data(input_image_name)

    construct_output(indent_level="block",
                     message="Done analyzing the elements of the image ({}) with a convolutional network."
                     .format(input_image_name))

    return value_names, durations
