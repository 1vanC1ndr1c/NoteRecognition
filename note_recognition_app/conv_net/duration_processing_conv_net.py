import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

from note_recognition_app.console_output.console_output_constructor import construct_output


def train_note_duration_conv_net(test_data_arr, test_data_label, train_data_arr, train_data_label):
    """
    This function trains the convolutional network for recognizing note durations based on input data.
    Tutorial for this code found here:
    https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb
    The results are saved on a disk so that they can be used without retraining the network.
    :param train_data_label: Labels with names and durations for the train data images.
    :param train_data_arr: Array containing the train images.
    :param test_data_label: Labels with names and durations for the test data images.
    :param test_data_arr: Array containing the test images.
    """

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Alleged fix for some tensorflow bugs.

    construct_output(indent_level=0,
                     message="Convolutional Network 1 (Note duration determining).")

    # Scale these values to a range of 0 to 1 before feeding them to the convolutional network model
    print("Scaling test values to [0-1] range.")
    test_data_arr = test_data_arr / 255.0
    print("Scaling train values to [0-1] range (this will take a while).")
    train_data_arr = train_data_arr / 255.0
    #
    # Construct the path for saving the results of training.
    saved_model_duration_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent.parent), 'resources'))
    saved_model_duration_path = os.path.join(saved_model_duration_path, 'saved_models')
    saved_model_name = "duration_processing_net_saved.ckpt"
    saved_model_duration_path = os.path.join(saved_model_duration_path, saved_model_name)
    duration_model_cb = tf.keras.callbacks.ModelCheckpoint(filepath=saved_model_duration_path,
                                                           save_weights_only=True,
                                                           verbose=1)

    # Second network recognizes the duration.
    duration_network_train_data_arr = np.array(
        [x for i, x in enumerate(train_data_arr) if train_data_label[i][0][1] != "Uncategorized"])
    duration_network_train_data_label = np.array([(x[0][1], x[1]) for x in train_data_label
                                                  if x[0][1] != "Uncategorized"])

    duration_network_test_data_arr = np.array(
        [x for i, x in enumerate(test_data_arr) if test_data_label[i][0][1] != "Uncategorized"])
    duration_network_test_data_label = np.array([(x[0][1], x[1]) for x in test_data_label
                                                 if x[0][1] != "Uncategorized"])

    # class_names contains possible results
    class_names = ["1/1", "1/2", "1/4", "1/8", "1/16"]

    # Fetch only the labels (note durations) from the data.
    duration_network_train_data_label = [item[0] for item in duration_network_train_data_label]
    # Assign the corresponding numerical values to labels.
    duration_train_label_values_numerical = values_to_numerical(duration_network_train_data_label, class_names)

    with tf.device('/GPU:1'):  # Specify using nvidia discrete GPU instead of Intel integrated graphics.
        construct_output(indent_level=0, message="Start training.")
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
            tf.keras.layers.Dense(5)
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

        # Training the convolutional network model requires the following steps:
        # Feed the training data to the model.
        # In this example, the training data is in the train_images and train_labels arrays.
        # The model learns to associate images and labels.
        # You ask the model to make predictions about a test set—in this example, the test_images array.
        # Verify that the predictions match the labels from the test_labels array.
        model.fit(
            duration_network_train_data_arr,
            duration_train_label_values_numerical,
            epochs=3,
            callbacks=[duration_model_cb]
        )
        construct_output(indent_level=0, message="Save the network weights to avoid retraining on every run.")

        # Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

        # TESTING THE NETWORK. =======================================================================================
        # Compare how the model performs on the test dataset.
        # duration_network_test_data_label = [item[0] for item in duration_network_test_data_label]
        # duration_network_test_data_label_values_numerical = values_to_numerical(
        #     duration_network_test_data_label,
        #     class_names)
        # test_loss, test_acc = model.evaluate(duration_network_test_data_arr,
        #                                      duration_network_test_data_label_values_numerical,
        #                                      verbose=2
        #                                      )
        # print('\nTest accuracy:', test_acc)
        # predictions = probability_model.predict(duration_network_test_data_arr)
        # print(predictions[0])
        # print("max= ", np.argmax(predictions[0]))
        # import cv2
        # cv2.imshow("img", duration_network_test_data_arr[0])
        # cv2.waitKey()

        construct_output(indent_level=0, message="End training.")
        construct_output(indent_level=0, message="Convolutional Network 1 (Note duration determining) Done.")


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


def analyze_using_saved_data(img_name):
    """
    This functions uses saved weights of an already trained network to recognize the notes from the input image.
    :param img_name: Name of the input image.
    :return: list: Sequence of notes writen as (for example): ['D4', 'D4', 'A4', 'A4', 'B4']
    """

    # Construct a path to the individual note elements of the image.
    dir_name = os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'input_images')
    dir_name = os.path.join(dir_name, 'input_images_individual_elements', img_name[:-4], 'unrecognized')

    # Create a convolutional network.
    model_note_duration = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(200, 200)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5)
    ])
    # Load saved values into the created network.
    saved_model_duration_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent.parent), 'resources'))
    saved_model_duration_path = os.path.join(saved_model_duration_path, 'saved_models')
    saved_model_name = "duration_processing_net_saved.ckpt"
    saved_model_duration_path = os.path.join(saved_model_duration_path, saved_model_name)
    model_note_duration.load_weights(saved_model_duration_path)
    model_note_duration = tf.keras.Sequential([model_note_duration, tf.keras.layers.Softmax()])

    # Sort the elements by their position in the image.
    elements = [el for el in os.listdir(dir_name) if el.endswith(".png")]
    elements_sorted = []
    for el in elements:
        el_row = el[el.index("_row") + 4: el.index("_slice")]
        el_row_position = el[el.index("_slice") + 6: el.index("_UNKNOWN")]
        elements_sorted.append((el_row, el_row_position, el))
    elements_sorted = sorted(elements_sorted, key=lambda e: (int(e[0]), int(e[1])))

    # class_names contains all the possible result names.
    class_names = ["1/1", "1/2", "1/4", "1/8", "1/16"]

    predictions_note_duration_names = []  # List that will contain note duration of the original image (ordered).
    for element in elements_sorted:  # Iterate through all the elements.
        element = element[2]  # Get the element name.
        img_path = os.path.join(dir_name, element)  # Construct the path to the element.
        img = cv2.imread(img_path)  # Read the image.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the color space.
        img = np.array([img]) / 255.0  # Scale to [0-1]
        predictions = model_note_duration.predict(img)  # Use the network to get the element duration (numerical)
        # Turn the numerical duration into a string.
        predictions_note_duration_names.append(class_names[int(np.argmax(predictions[0]))])

    return predictions_note_duration_names
