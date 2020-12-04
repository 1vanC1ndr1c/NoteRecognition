import os
from os import listdir
from pathlib import Path
import re

import cv2

# import tensorflow as tf
import numpy as np
import random
import concurrent.futures

from io import StringIO
import sys


def prepare_the_data(directory):
    # Generate the path to the images.
    dataset_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent), 'resources', 'dataset'))
    # Load all the image files in the directory.
    dataset_images_names = [im for im in listdir(dataset_path) if im.endswith(".png")]

    # Dictionary that will hold classifications of the images (note values).
    names_and_note_values = {}
    # Iterate through all the image names to classify them.
    for index, img_name in enumerate(dataset_images_names):
        if img_name.startswith("note"):  # If it is classified as a note.
            # Find the note value from the name ( for example '..._C1_...')
            value = re.findall(r'_[A|B|C|D|E|F|G|a|b|c|d|e|f|g][0-9]_', img_name)
            if len(value) > 0:  # If a note has value, save the found value.
                names_and_note_values[img_name] = value[0][1:-1]
            else:  # Otherwise, classify the note as uncategorized.
                names_and_note_values[img_name] = "Uncategorized"

    # Split the data into train and test components.
    no_of_items = int(0.2 * len(names_and_note_values))  # Take 20% of items for testing.
    # Get a random 20% of items from the original list.
    # test_data_dictionary contains an image name and its classification.
    test_data_dictionary = dict(random.sample(names_and_note_values.items(), no_of_items))
    # Use the rest of the items for training.
    # train contains an image name and its classification.
    train_data_dictionary = {k: v for k, v in names_and_note_values.items() if k not in test_data_dictionary}

    class_names = ["A3", "A4", "A5",
                   "B3", "B4",
                   "C3", "C4",
                   "D3", "D4",
                   "E3", "E4",
                   "F3", "F4",
                   "G3", "G4", ]

    output_value = "Loading test data > Image(0 of {})".format(len(test_data_dictionary))
    output_value = output_value + " | Loading train data > Image(0 of {})".format(len(train_data_dictionary))
    output_value = [output_value]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        test_future = executor.submit(load_images, test_data_dictionary, dataset_path, "test", output_value)
        train_future = executor.submit(load_images, train_data_dictionary, dataset_path, "train", output_value)

        test_data = test_future.result()
        train_data = train_future.result()


def load_images(image_dictionary, dataset_path, type_of_data, output_value):
    counter = 0
    data_len = len(image_dictionary)
    data = []  # List containing(test_images, test_classification, image_name)
    for img_name, img_value in image_dictionary.items():
        counter = counter + 1
        out_str = output_value[0]
        if type_of_data == "train":
            out_str = out_str[0: out_str.find("|") + 1]
            out_str = out_str + "Loading {} data > Image({} of {})".format(type_of_data, counter, data_len)
            print("\r{}".format(out_str), end='')
        elif type_of_data == "test":
            out_str = out_str[out_str.find("|") - 1:]
            out_str = "Loading {} data > Image({} of {})".format(type_of_data, counter, data_len) + out_str
            print("\r{}".format(out_str), end='')
        output_value[0] = out_str

        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        data.append((img, img_value, img_name))
    return data
