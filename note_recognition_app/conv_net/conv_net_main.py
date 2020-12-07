from note_recognition_app.conv_net import value_processing_conv_net


def conv_network_analysis(input_image_name):
    # If retraining is needed, uncomment this and comment loading the trained data from the disk.
    # value_processing_conv_net.train_note_values_conv_net(test_data_percentage=0.2) #UNCOMMENT FOR RETRAINING

    value_names = value_processing_conv_net.analyze_using_saved_data(input_image_name)
    print(value_names)

