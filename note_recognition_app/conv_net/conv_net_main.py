from note_recognition_app.console_output.console_output_constructor import construct_output
from note_recognition_app.conv_net import duration_processing_conv_net
from note_recognition_app.conv_net import value_processing_conv_net
from note_recognition_app.conv_net.input_data_processing import prepare_new_data


def conv_network_analysis(input_image_name):
    """
    Main function for convolutional network analysis.
    Calls on the network for value recognizing.
    Calls on the network for duration analyzing.
    :param input_image_name: Name of the image that is being analyzed.
    """
    construct_output(indent_level="block",
                     message="Analyzing the elements of the image ({}) with a convolutional network."
                     .format(input_image_name))

    # Import the dataset and split it to training and testing.
    (test_arr, test_label), (train_arr, train__label) = prepare_new_data(test_data_percentage=0.2)
    value_processing_conv_net.train_note_values_conv_net(test_arr, test_label, train_arr, train__label)
    duration_processing_conv_net.train_note_duration_conv_net(test_arr, test_label, train_arr, train__label)

    # Load the trained data from the disk.
    value_names = value_processing_conv_net.analyze_using_saved_data(input_image_name)
    # Load the trained data from the disk.
    durations = duration_processing_conv_net.analyze_using_saved_data(input_image_name)

    for i in range(len(value_names)):
        print(value_names[i], durations[i])

    construct_output(indent_level="block",
                     message="Done analyzing the elements of the image ({}) with a convolutional network."
                     .format(input_image_name))

    return value_names, durations
