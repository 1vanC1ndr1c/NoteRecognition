import concurrent.futures
import os
import random
import re
from os import listdir
from pathlib import Path

import cv2
import numpy as np


def prepare_new_data(test_data_percentage):
    """
    Main function that calls all the other functions listed below.
    It classifies all the images found in the dataset into a dictionary containing the image name as a key and a
    note value as a value.
    It splits the data into a training set and testing set.
    It then loads the image, saves it with its value and a name into .dat files (separate files for train and test).
    :param test_data_percentage: A number ([0-1]) indicating how much of the resources data will be used for testing.
    :return: list, list: A list containing all the images for testing, their values and names.
                        A list containing all the images for training, their values and names.
    """
    # Generate the path to the images.
    dataset_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'dataset'))

    # Load all the image files in the directory. TEST, LIMITED DATASET, comment the other one.
    # dataset_images_names = [im for i, im in enumerate(listdir(dataset_path)) if im.endswith(".png") and i < 200]

    # # Load all the image files in the directory. REAL DATASET, comment the other one.
    dataset_images_names = [im for im in listdir(dataset_path) if im.endswith(".png")]

    # Classify the resources images based on their values. (Connect an image with what it is shown on it.)
    names_and_note_values = classify_the_images(dataset_images_names)

    # Split the names_and_note_values into two dictionaries (TEST and TRAIN dictionaries).
    test_data_dictionary, train_data_dictionary = split_the_data(names_and_note_values, test_data_percentage)

    # Get the images, their values, and their names for both test and train dictionaries.
    test_data, train_data = load_dataset(test_data_dictionary, train_data_dictionary, dataset_path)

    return test_data, train_data


def classify_the_images(dataset_images_names):
    """
    This function associates an image name with the note value shown on the image.
    The values can be read from image names.
    The image name format is: ...._[LETTER][NUMBER]...(for example ...._C1....).
    The function extracts that part of the name (if it exists) and adds it into a dictionary where the key
    is the image name and the value is the note value.
    :param dataset_images_names: list: contains the image names.
    :return: dict: containing the image name and corresponding value.
    """
    # Dictionary that will hold classifications of the images (note values).
    names_and_classification = {}
    # Iterate through all the image names to classify them.
    for index, img_name in enumerate(dataset_images_names):
        if img_name.startswith("note"):  # If it is classified as a note.
            duration = "Uncategorized"
            if "16TH" in img_name:
                duration = "1/16"
            elif "EIGHTH" in img_name:
                duration = "1/8"
            elif "HALF" in img_name:
                duration = "1/2"
            elif "QUARTER" in img_name:
                duration = "1/4"
            elif "WHOLE" in img_name:
                duration = "1/1"

            # Find the note value from the name ( for example '..._C1_...')
            value = re.findall(r'_[A|B|C|D|E|F|G|a|b|c|d|e|f|g][0-9]_', img_name)
            if len(value) > 0:  # If a note has value, save the found value.
                value = value[0][1:-1]
            else:  # Otherwise, classify the note as uncategorized.
                value = "Uncategorized"

            names_and_classification[img_name] = (value, duration)

    return names_and_classification

def split_the_data(names_and_note_values, test_data_percentage):
    """
    Splits the data into train and test components.
    :param names_and_note_values: Collection that needs to be split.
    :param test_data_percentage: Number [0-1] indicating how much of the 'names_and_note_values' is used for testing.
    :return: dict, dict: Original data split into test and training dictionaries.
    """
    no_of_items = int(test_data_percentage * len(names_and_note_values))  # Take a percentage of items for testing.
    # Get a random requested percent of items from the original list.
    # test_data_dictionary contains an image name and its classification.
    test_data_dictionary = dict(random.sample(names_and_note_values.items(), no_of_items))
    # Use the rest of the items for training.
    # train contains an image name and its classification.
    train_data_dictionary = {k: v for k, v in names_and_note_values.items() if k not in test_data_dictionary}
    return test_data_dictionary, train_data_dictionary


def load_dataset(test_data_dictionary, train_data_dictionary, dataset_path):
    """
    This function loads the dataset.
    :param test_data_dictionary: Dictionary containing image name and labels for testing part of dataset.
    :param train_data_dictionary: Dictionary containing image name and labels for training part of dataset.
    :param dataset_path: Path to the images location.
    :return: (np.array, np.array), (np.array, np.array): Arrays containing images and their labels (in tuples).
    """

    output_value = "Loading test data > Image(0 of {})".format(len(test_data_dictionary))
    output_value = output_value + " | Loading train data > Image(0 of {})".format(len(train_data_dictionary))
    # Turn it into a list so it can be passed by reference to both threads as a common variable.
    output_value = [output_value]

    # Load the test and train set parallel to one another using threads.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Test images loading thread.
        test_future = executor.submit(load_images, test_data_dictionary, dataset_path, "test", output_value)
        # Training images loading thread.
        train_future = executor.submit(load_images, train_data_dictionary, dataset_path, "train", output_value)
        # Get the results.
        test_data_array, test_data_label = test_future.result()
        train_data_array, train_data_label = train_future.result()

    print()
    return (test_data_array, test_data_label), (train_data_array, train_data_label)


def load_images(image_dictionary, dataset_path, type_of_data, output_value):
    """
    Function that reads the images specified by name and path.
    :param image_dictionary: Contains image name and its corresponding classification.
    :param dataset_path: Path to said image.
    :param type_of_data: Specifies if it is a 'train' or 'test' set.
    :param output_value: String used to report the progress using stdout.
    :return: np.array, np.array: containing a tuple(image(array), (image classification, image name))
    """
    counter = 0  # Counter used to report the progress( 'counter value' out of 'maximum number')
    data_len = len(image_dictionary)  # Get the length of dictionary.
    data_array = []  # List containing numpy array images
    data_label = []  # List containing labels that define images.
    for img_name, img_value in image_dictionary.items():  # Iterate over the dictionary.
        counter = counter + 1
        out_str = output_value[0]  # Read the string used to report the progress.

        if type_of_data == "train":  # If the training set is being read...
            out_str = out_str[0: out_str.find("|") + 2]  # Change only the counter for that part of the string.
            out_str = out_str + "Loading {} data > Image({} of {})".format(type_of_data, counter, data_len)
            print("\r{}".format(out_str), end='')  # Print the current progress.

        elif type_of_data == "test":  # If the testing set is being read...
            out_str = out_str[out_str.find("|") - 2:]  # Change only the counter for that part of the string.
            out_str = "Loading {} data > Image({} of {})".format(type_of_data, counter, data_len) + out_str
            print("\r{}".format(out_str), end='')  # Print the current progress.

        output_value[0] = out_str  # Update the values shown in output.

        img_path = os.path.join(dataset_path, img_name)  # Construct the full path to the image.
        img = cv2.imread(img_path)  # Read the image.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        data_array.append(img)  # Save the image.
        data_label.append((img_value, img_name))  # Save image classification and name into a list.

    data_array = np.array(data_array)
    return data_array, data_label
