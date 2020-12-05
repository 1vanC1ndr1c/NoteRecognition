from note_recognition_app.nerual_net.value_processing_neural_net import train_note_values_neural_net


def start_the_networks():
    # If retraining is needed, uncomment this and comment loading the trained data from the disk.
    train_note_values_neural_net(0.2)
