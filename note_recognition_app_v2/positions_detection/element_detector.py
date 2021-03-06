import os
import warnings
from os import listdir
from os.path import isfile, join
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage
from PIL import Image, ImageChops
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from skimage.io import imread
from skimage.morphology import label
from skimage.transform import resize

from note_recognition_app_v2.positions_detection import unet_model_builder
from note_recognition_app_v2.positions_detection.input_loader import create_input_data
from note_recognition_app_v2.positions_detection.unet_model_builder import build_unet_model

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# Set some parameters.
BATCH_SIZE = 1
EPOCHS = 20
STEPS_PER_EPOCH = 60
VALIDATION_STEPS = 30
THRESHOLD = 0.96
SEED = 42

# IMG_CHANNELS = 3
# TRAIN_PATH = os.path.abspath(os.path.join(str(Path(__file__).parent), 'resources', 'stage1_train'))
# TEST_PATH = os.path.abspath(os.path.join(str(Path(__file__).parent), 'resources', 'stage1_test'))
# IMG_WIDTH = 128  # For faster computing on kaggle.
# IMG_HEIGHT = 128  # For faster computing on kaggle.


IMG_CHANNELS = 3
IMG_WIDTH = 512  # for faster computing on kaggle
IMG_HEIGHT = 512  # for faster computing on kaggle
TRAIN_PATH = os.path.abspath(os.path.join(str(Path(__file__).parent), 'resources', 'train'))
TEST_PATH = os.path.abspath(os.path.join(str(Path(__file__).parent), 'resources', 'test'))


def detect_elements(input_img, retrain_flag=False):
    if retrain_flag is True:
        x_train, y_train, x_test, sizes_test, train_generator, val_generator = create_input_data(
            TRAIN_PATH, TEST_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, SEED, BATCH_SIZE)
        print(sizes_test)
        model = build_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

        model_fit(
            model, train_generator, val_generator,
            unet_model_builder.mean_iou, x_train, x_test, sizes_test, y_train)

    predict_without_retraining(input_img)


def predict_without_retraining(img_name):
    path = TRAIN_PATH + '/' + img_name  # Get the path to the image.
    img = imread(path + '/images/' + img_name + '.png')  # Read the image.

    if TRAIN_PATH.endswith('stage1_train'):
        model_checkpoint_path = os.path.join(str(Path(__file__).parent),
                                             'resources', 'position_detector_model', 'medicine', 'checkpoint.h5')
        masks_path = os.path.join(str(Path(__file__).parent),
                                  'resources', 'stage1_train', img_name, 'masks')
    else:
        model_checkpoint_path = os.path.join(str(Path(__file__).parent),
                                             'resources', 'position_detector_model', 'notes', 'checkpoint.h5')
        masks_path = os.path.join(str(Path(__file__).parent),
                                  'resources', 'train', img_name, 'masks')

    masks_names = [f for f in listdir(masks_path) if isfile(join(masks_path, f))]
    masks_img = Image.open(os.path.join(masks_path, masks_names[0]))
    for index, mask_name in enumerate(masks_names):
        if index == 0:
            continue
        mask_img = Image.open(os.path.join(masks_path, mask_name))
        diff = ImageChops.difference(masks_img, mask_img)
        if diff.getbbox():
            masks_img = diff

    model = load_model(model_checkpoint_path, custom_objects={'mean_iou': unet_model_builder.mean_iou})

    if len(img.shape) == 2:  # Convert it to rgb, if needed.
        img = skimage.color.gray2rgb(img)
    img = img[:, :, :IMG_CHANNELS]  # Read the image channels.
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img_data = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    img_data[0] = img

    predictions = model.predict([img_data], verbose=1)
    prediction = resize(np.squeeze(predictions[0]), (3700, 3700), mode='constant', preserve_range=True)
    prediction_thresh = (prediction > THRESHOLD).astype(np.uint8)

    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(img.astype('uint8'))
    axarr[1].imshow(np.array(masks_img).astype('uint8'))
    axarr[2].imshow(prediction_thresh)
    plt.show()


def model_fit(model, train_generator, val_generator, mean_iou, x_train, x_test, sizes_test, y_train):
    earlystopper = EarlyStopping(patience=3, verbose=1)

    if TRAIN_PATH.endswith('stage1_train'):
        model_checkpoint_path = os.path.join(str(Path(__file__).parent),
                                             'resources', 'position_detector_model', 'medicine', 'checkpoint.h5')
    else:
        model_checkpoint_path = os.path.join(str(Path(__file__).parent),
                                             'resources', 'position_detector_model', 'notes', 'checkpoint.h5')
    checkpointer = ModelCheckpoint(model_checkpoint_path, verbose=1, save_best_only=True)
    model.fit_generator(train_generator,
                        validation_data=val_generator,
                        validation_steps=VALIDATION_STEPS,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=EPOCHS,
                        callbacks=[earlystopper, checkpointer])

    # # Predict on train, val and test
    # model = load_model(model_checkpoint_path, custom_objects={'mean_iou': mean_iou})
    # preds_train = model.predict(x_train[:int(x_train.shape[0] * 0.9)], verbose=1)
    # preds_val = model.predict(x_train[int(x_train.shape[0] * 0.9):], verbose=1)
    # preds_test = model.predict(x_test, verbose=1)
    #
    # # Threshold predictions
    # preds_train_t = (preds_train > THRESHOLD).astype(np.uint8)
    # preds_val_t = (preds_val > THRESHOLD).astype(np.uint8)
    # preds_test_t = (preds_test > THRESHOLD).astype(np.uint8)
    #
    # # Create list of upsampled test masks
    # preds_test_upsampled = []
    # for i in range(len(preds_test)):
    #     preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
    #                                        (sizes_test[i][0], sizes_test[i][1]),
    #                                        mode='constant', preserve_range=True))
    #
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    #
    # # Perform a sanity check on some random training samples
    # ix = random.randint(0, len(preds_train_t))
    # imshow(x_train[ix])
    # plt.show()
    # imshow(np.squeeze(y_train[ix]))
    # plt.show()
    # imshow(np.squeeze(preds_train_t[ix]))
    # plt.show()
    #
    # # Perform a sanity check on some random validation samples
    # ix = random.randint(0, len(preds_val_t))
    # imshow(x_train[int(x_train.shape[0] * 0.9):][ix])
    # plt.show()
    # imshow(np.squeeze(y_train[int(y_train.shape[0] * 0.9):][ix]))
    # plt.show()
    # imshow(np.squeeze(preds_val_t[ix]))
    # plt.show()
    #
    # return preds_test_upsampled


# Run-length encoding inspired by from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
