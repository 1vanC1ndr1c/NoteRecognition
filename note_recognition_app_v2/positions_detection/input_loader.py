import os
import sys

import numpy as np
import skimage
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm


def create_input_data(train_path, test_path, img_h, img_w, img_channels, seed, batch_size):
    x_train, y_train, x_test, sizes_test = load_images_into_training_set(
        train_path, test_path, img_h, img_w, img_channels)
    train_generator, val_generator = create_generators(x_train, y_train, seed, batch_size)

    return x_train, y_train, x_test, sizes_test, train_generator, val_generator


def load_images_into_training_set(train_path, test_path, img_h, img_w, img_channels):
    # Get train and test image names.
    train_image_names = next(os.walk(train_path))[1]
    test_image_names = next(os.walk(test_path))[1]

    np.random.seed(10)

    # Holder for the training set.
    x_train = np.zeros((len(train_image_names), img_h, img_w, img_channels), dtype=np.uint8)
    y_train = np.zeros((len(train_image_names), img_h, img_w, 1), dtype=np.bool)

    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()

    for index, img_name in tqdm(enumerate(train_image_names), total=len(train_image_names)):
        print('\nWorking on {}'.format(img_name))

        path = train_path + '/' + img_name  # Get the path to the image.
        img = imread(path + '/images/' + img_name + '.png')  # Read the image.
        if len(img.shape) == 2:  # Convert it to rgb, if needed.
            img = skimage.color.gray2rgb(img)

        img = img[:, :, :img_channels]  # Read the image channels.
        img = resize(img, (img_h, img_w), mode='constant', preserve_range=True)  # Resize the image.
        x_train[index] = img  # Add it to the training set.

        mask = np.zeros((img_h, img_w, 1), dtype=np.bool)
        total_masks = len(next(os.walk(path + '/masks/'))[2])
        for mask_number, mask_file in enumerate(next(os.walk(path + '/masks/'))[2]):
            print('\r{} Working on {} ({} out of {})'.format(10 * " ", mask_file, mask_number, total_masks), end="")
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (img_h, img_w), mode='constant',
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        y_train[index] = mask

    # Get and resize test images
    x_test = np.zeros((len(test_image_names), img_h, img_w, img_channels), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for index, img_name in tqdm(enumerate(test_image_names), total=len(test_image_names)):
        print('\nWorking on {}'.format(img_name))
        path = test_path + '/' + img_name
        img = imread(path + '/images/' + img_name + '.png')
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        img = img[:, :, :img_channels]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (img_h, img_w), mode='constant', preserve_range=True)
        x_test[index] = img
    print('Done!')

    return x_train, y_train, x_test, sizes_test


def create_generators(x_train, y_train, seed, batch_size):
    from keras.preprocessing import image

    # Creating the training Image and Mask generator.
    image_data_generator = image.ImageDataGenerator(shear_range=0.5,
                                                    rotation_range=50,
                                                    zoom_range=0.2,
                                                    width_shift_range=0.2,
                                                    height_shift_range=0.2, fill_mode='reflect')
    mask_data_generator = image.ImageDataGenerator(shear_range=0.5,
                                                   rotation_range=50,
                                                   zoom_range=0.2,
                                                   width_shift_range=0.2,
                                                   height_shift_range=0.2, fill_mode='reflect')

    # Keep the same seed for image and mask generators so they fit together.

    image_data_generator.fit(x_train[:int(x_train.shape[0] * 0.9)], augment=True, seed=seed)
    mask_data_generator.fit(y_train[:int(y_train.shape[0] * 0.9)], augment=True, seed=seed)

    x = image_data_generator.flow(x_train[:int(x_train.shape[0] * 0.9)],
                                  batch_size=batch_size,
                                  shuffle=True, seed=seed)
    y = mask_data_generator.flow(y_train[:int(y_train.shape[0] * 0.9)],
                                 batch_size=batch_size,
                                 shuffle=True, seed=seed)

    # Creating the validation Image and Mask generator.
    image_data_generator_val = image.ImageDataGenerator()
    mask_data_generator_val = image.ImageDataGenerator()

    image_data_generator_val.fit(x_train[int(x_train.shape[0] * 0.9):], augment=True, seed=seed)
    mask_data_generator_val.fit(y_train[int(y_train.shape[0] * 0.9):], augment=True, seed=seed)

    x_val = image_data_generator_val.flow(x_train[int(x_train.shape[0] * 0.9):],
                                          batch_size=batch_size,
                                          shuffle=True, seed=seed)
    y_val = mask_data_generator_val.flow(y_train[int(y_train.shape[0] * 0.9):],
                                         batch_size=batch_size,
                                         shuffle=True, seed=seed)

    # Creating a training and validation generator that generate masks and images.
    train_generator = zip(x, y)
    val_generator = zip(x_val, y_val)

    return train_generator, val_generator
