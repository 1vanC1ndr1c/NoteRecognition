import os
from os import listdir
from os.path import isfile, join
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def start_the_networks():
    # If retraining is needed, uncomment this and comment loading the trained data from the disk.
    # train_note_values_neural_net(0.2)

    # Otherwise, use the saved data.
    model_note_values = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(200, 200)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(22)
    ])
    saved_model_values_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent.parent), 'resources'))
    saved_model_values_path = os.path.join(saved_model_values_path, 'saved_models')
    saved_model_name = "value_processing_net_saved.ckpt"
    saved_model_values_path = os.path.join(saved_model_values_path, saved_model_name)
    model_note_values.load_weights(saved_model_values_path)
    model_note_values = tf.keras.Sequential([model_note_values, tf.keras.layers.Softmax()])

    # test area
    img_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'TEST_value'))
    img_name = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    img_name = img_name[1]
    img_path = os.path.join(img_path, img_name)
    img = cv2.imread(img_path)  # Read the image.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array([img]) / 255.0

    predictions = model_note_values.predict(img)

    class_names = ["A3", "A4", "A5",  # class_names contains possible results
                   "B3", "B4", "B5",
                   "C3", "C4", "C5",
                   "D3", "D4", "D5",
                   "E3", "E4", "E5",
                   "F3", "F4", "F5",
                   "G3", "G4", "G5"]

    print(predictions[0])
    print("max index = ", np.argmax(predictions[0]), ", value = ", class_names[int(np.argmax(predictions[0]))])
