from note_recognition_app.conv_net import value_processing_conv_net
from note_recognition_app.info_output.output_constructor import construct_output


def conv_network_analysis(input_image_name):
    """
    Main function for convolutional network analysis.
    Calls on the network for value recognizing.
    TODO Calls on the network for duration analyzing.
    :param input_image_name: Name of the image that is being analyzed.
    """
    construct_output(indent_level="block",
                     message="Analyzing the elements of the image ({}) with a convolutional network."
                     .format(input_image_name))

    # If retraining is needed, uncomment this.
    #  value_processing_conv_net.train_note_values_conv_net(test_data_percentage=0.2) #UNCOMMENT FOR RETRAINING

    # Load the trained data from the disk.
    value_names = value_processing_conv_net.analyze_using_saved_data(input_image_name)
    print(value_names)

    # TODO similar for note durations
    #   If retraining is needed, uncomment this.
    #   duration_processing_conv_net.train_note_durations_conv_net(test_data_percentage=0.2) #UNCOMMENT FOR RETRAINING
    #   Load the trained data from the disk.
    #   durations = duration_processing_conv_net.analyze_using_saved_data(input_image_name)

    construct_output(indent_level="block",
                     message="Done analyzing the elements of the image ({}) with a convolutional network."
                     .format(input_image_name))
