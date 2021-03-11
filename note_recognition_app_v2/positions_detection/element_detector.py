import os
import warnings
from pathlib import Path

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from skimage.morphology import label
from skimage.transform import resize

from note_recognition_app_v2.positions_detection import unet_model_builder
from note_recognition_app_v2.positions_detection.retrainer import retrain
from note_recognition_app_v2.positions_detection.unet_model_builder import build_unet_model

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# Set some parameters.
BATCH_SIZE = 1
EPOCHS = 5
STEPS_PER_EPOCH = 20
VALIDATION_STEPS = 10
THRESHOLD = 0.3
SEED = 42
IMG_CHANNELS = 3

TRAIN_PATH = os.path.abspath(os.path.join(str(Path(__file__).parent), 'resources', 'stage1_train'))
TEST_PATH = os.path.abspath(os.path.join(str(Path(__file__).parent), 'resources', 'stage1_test'))
IMG_WIDTH = 128  # For faster computing on kaggle.
IMG_HEIGHT = 128  # For faster computing on kaggle.


# IMG_WIDTH = 512  # for faster computing on kaggle
# IMG_HEIGHT = 512  # for faster computing on kaggle
# TRAIN_PATH = os.path.abspath(os.path.join(str(Path(__file__).parent), 'resources', 'train'))
# TEST_PATH = os.path.abspath(os.path.join(str(Path(__file__).parent), 'resources', 'test'))


def detect_elements(input_img):
    x_train, y_train, x_test, sizes_test, train_generator, val_generator = retrain(
        TRAIN_PATH, TEST_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, SEED, BATCH_SIZE)
    model = build_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    preds_test_upsampled = model_fit(
        model, train_generator, val_generator, unet_model_builder.mean_iou, x_train, x_test, sizes_test, y_train)

    new_test_ids = []
    rles = []
    test_ids = next(os.walk(TEST_PATH))[1]
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))


def model_fit(model, train_generator, val_generator, mean_iou, x_train, x_test, sizes_test, y_train):
    # Fit model
    earlystopper = EarlyStopping(patience=3, verbose=1)
    model_checkpoint_path = os.path.join(str(Path(__file__).parent),
                                         'resources', 'position_detector_model', 'checkpoint.h5')
    checkpointer = ModelCheckpoint(model_checkpoint_path, verbose=1, save_best_only=True)
    results = model.fit_generator(train_generator,
                                  validation_data=val_generator,
                                  validation_steps=VALIDATION_STEPS,
                                  steps_per_epoch=STEPS_PER_EPOCH,
                                  epochs=EPOCHS,
                                  callbacks=[earlystopper, checkpointer])

    # Predict on train, val and test
    model = load_model(model_checkpoint_path, custom_objects={'mean_iou': mean_iou})
    preds_train = model.predict(x_train[:int(x_train.shape[0] * 0.9)], verbose=1)
    preds_val = model.predict(x_train[int(x_train.shape[0] * 0.9):], verbose=1)
    preds_test = model.predict(x_test, verbose=1)

    # Threshold predictions
    preds_train_t = (preds_train > THRESHOLD).astype(np.uint8)
    preds_val_t = (preds_val > THRESHOLD).astype(np.uint8)
    preds_test_t = (preds_test > THRESHOLD).astype(np.uint8)

    # Create list of upsampled test masks
    preds_test_upsampled = []
    for i in range(len(preds_test)):
        preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                           (sizes_test[i][0], sizes_test[i][1]),
                                           mode='constant', preserve_range=True))

    # Perform a sanity check on some random training samples
    # ix = random.randint(0, len(preds_train_t))
    # TODO Replace with cv2.imshow
    # imshow(X_train[ix])
    # plt.show()
    # imshow(np.squeeze(Y_train[ix]))
    # plt.show()
    # imshow(np.squeeze(preds_train_t[ix]))
    # plt.show()

    # Perform a sanity check on some random validation samples
    # ix = random.randint(0, len(preds_val_t))
    # TODO Replace with cv2.imshow
    # imshow(X_train[int(X_train.shape[0] * 0.9):][ix])
    # plt.show()
    # imshow(np.squeeze(Y_train[int(Y_train.shape[0] * 0.9):][ix]))
    # plt.show()
    # imshow(np.squeeze(preds_val_t[ix]))
    # plt.show()

    return preds_test_upsampled


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
