from note_recognition_app.nerual_net.input_data_processing import prepare_new_data

import matplotlib.pyplot as plt

import tensorflow as tf

import numpy as np


def start_neural_net():
    # Import the dataset_OLD and split it to training and testing.
    (test_data_arr, test_data_label), (train_data_arr, train_data_label) = prepare_new_data(test_data_percentage=0.2)

    # Scale these values to a range of 0 to 1 before feeding them to the neural network model
    print("Scaling test values to [0-1] range.")
    test_data_arr = test_data_arr / 255.0
    print("Scaling train values to [0-1] range (this will take a while).")
    train_data_arr = train_data_arr / 255.0

    class_names = ["A3", "A4", "A5",  # class_names contains possible results
                   "B3", "B4", "B5",
                   "C3", "C4", "C5",
                   "D3", "D4", "D5",
                   "E3", "E4", "E5",
                   "F3", "F4", "F5",
                   "G3", "G4", "G5",
                   "Uncategorized"]

    # Fetch only the labels (note values) from the data.
    test_data_label_values = [item[0] for item in test_data_label]
    train_data_label_values = [item[0] for item in train_data_label]
    # Assign the corresponding numerical values to labels.
    test_data_label_values_numerical = values_to_numerical(test_data_label_values, class_names)
    train_data_label_values_numerical = values_to_numerical(train_data_label_values, class_names)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(200, 200)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_data_arr, train_data_label_values_numerical, epochs=2)

    test_loss, test_acc = model.evaluate(test_data_arr, test_data_label_values_numerical, verbose=2)

    print('\nTest accuracy:', test_acc)


def values_to_numerical(data_label_values, class_names):
    data_label_values_numerical = []
    for label_value in data_label_values:
        if label_value in class_names:
            data_label_values_numerical.append(class_names.index(label_value))
        else:
            print(label_value)
    data_label_values_numerical = np.asarray(data_label_values_numerical)
    return data_label_values_numerical
