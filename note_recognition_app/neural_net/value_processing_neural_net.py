import os
from pathlib import Path

import numpy as np
#import tensorflow as tf

from note_recognition_app.neural_net.input_data_processing import prepare_new_data


def train_note_values_neural_net(test_data_percentage):
    """
    This function trains the neural network for recognizing note values based on input data.
    Tutorial for this code found here:
    https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb
    The results are saved on a disk so that they can be used without retraining the network.
    :param test_data_percentage: A number ([0-1]) indicating how much of the input data will be used for testing.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Alleged fix for some tensorflow bugs.

    # Import the dataset and split it to training and testing.
    (test_data_arr, test_data_label), (train_data_arr, train_data_label) = prepare_new_data(test_data_percentage)

    # Scale these values to a range of 0 to 1 before feeding them to the neural network model
    print("Scaling test values to [0-1] range.")
    test_data_arr = test_data_arr / 255.0
    print("Scaling train values to [0-1] range (this will take a while).")
    train_data_arr = train_data_arr / 255.0

    # Construct the path for saving the results of training.
    saved_model_values_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent.parent), 'resources'))
    saved_model_values_path = os.path.join(saved_model_values_path, 'saved_models')
    saved_model_name = "value_processing_net_saved.ckpt"
    saved_model_values_path = os.path.join(saved_model_values_path, saved_model_name)
    values_model_cb = tf.keras.callbacks.ModelCheckpoint(filepath=saved_model_values_path,
                                                         save_weights_only=True,
                                                         verbose=1)

    # First network only recognizes the values. No need to feed it unrecognized elements (elements with no value).
    value_network_train_data_arr = np.array(
        [x for i, x in enumerate(train_data_arr) if train_data_label[i][0] != "Uncategorized"])
    value_network_train_data_label = np.array([x for x in train_data_label if x[0] != "Uncategorized"])

    value_network_test_data_arr = np.array(
        [x for i, x in enumerate(test_data_arr) if test_data_label[i][0] != "Uncategorized"])
    value_network_test_data_label = np.array([x for x in test_data_label if x[0] != "Uncategorized"])

    class_names = ["A3", "A4", "A5",  # class_names contains possible results
                   "B3", "B4", "B5",
                   "C3", "C4", "C5",
                   "D3", "D4", "D5",
                   "E3", "E4", "E5",
                   "F3", "F4", "F5",
                   "G3", "G4", "G5"]

    # Fetch only the labels (note values) from the data.
    value_network_test_data_label = [item[0] for item in value_network_test_data_label]
    value_network_train_data_label = [item[0] for item in value_network_train_data_label]
    # Assign the corresponding numerical values to labels.
    value_network_test_data_label_values_numerical = values_to_numerical(value_network_test_data_label, class_names)
    value_network_train_data_label_values_numerical = values_to_numerical(value_network_train_data_label, class_names)

    with tf.device('/GPU:1'):  # Specify using nvidia discrete GPU instead of Intel integrated graphics.

        # Set up the layers.
        # The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images
        # from a 2D array(200x200px) to 1D array(of 200x200 = 40000 pixels)
        # After  the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers.
        # These are densely connected, or fully connected, neural layers.
        # The first Dense layer has 128 nodes( or neurons).
        # The second( and last) layer returns an array with length of 22.
        # Each node contains a score that indicates the current image belongs to one of the 22 classes.

        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(200, 200)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(22)
        ])

        # Before the model is ready for training, it needs a few more settings.
        # These are added during the model's compile step:
        # Loss function —This measures how accurate the model is during training.
        # You want to minimize this function to "steer" the model in the right direction.
        # Optimizer —This is how the model is updated based on the data it sees and its loss function.
        # Metrics —Used to monitor the training and testing steps.
        # The following example uses accuracy, the fraction of the images that are correctly classified.

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # Training the neural network model requires the following steps:
        # Feed the training data to the model.
        # In this example, the training data is in the train_images and train_labels arrays.
        # The model learns to associate images and labels.
        # You ask the model to make predictions about a test set—in this example, the test_images array.
        # Verify that the predictions match the labels from the test_labels array.
        model.fit(
            value_network_train_data_arr,
            value_network_train_data_label_values_numerical,
            epochs=2,
            callbacks=[values_model_cb])

        # Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

        # Compare how the model performs on the test dataset.
        # test_loss, test_acc = model.evaluate(value_network_test_data_arr,
        #                                      value_network_test_data_label_values_numerical,
        #                                      verbose=2
        #                                      )
        # print('\nTest accuracy:', test_acc)
        predictions = probability_model.predict(value_network_test_data_arr)
        # print(predictions[0])
        # print("max= ", np.argmax(predictions[0]))
        # import cv2
        # cv2.imshow("sa", value_network_test_data_arr[0])
        # cv2.waitKey()


def values_to_numerical(data_label_values, class_names):
    """
    This function assigns a numerical value (number) based on a classification and returns a collection containing
    numbers instead of string inputs.
    :param data_label_values: Collection containing string classifiers.
    :param class_names:  Collection containing unique string classifiers. Their index within the list will be
        used to represent the numerical values in the return value.
    :return: np.array : Array containing numbers correlated to the classifications from the original collection.
    """
    data_label_values_numerical = []  # List that will contain aforementioned numbers.

    for label_value in data_label_values:  # Iterate through string classifications.
        # If a classification is contained within unique string classifications (class_names)....
        if label_value in class_names:
            # Assign a numerical value to it (in the return value).
            data_label_values_numerical.append(class_names.index(label_value))
        else:
            # If an unrecognized classification is found, that is considered a severe error.
            print("values_to_numerical ERROR! Unclassified items! Check input stream!", label_value)
            exit(-1)  # Stop the program

    # Convert it to numpy array.
    data_label_values_numerical = np.asarray(data_label_values_numerical)
    return data_label_values_numerical
