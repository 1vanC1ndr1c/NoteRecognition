from PySide2 import QtCore
from PySide2.QtGui import QFont, QIcon, Qt, QCloseEvent, QImage, QPixmap
from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, \
    QHBoxLayout, QSizePolicy, QFrame, QMenu, QMainWindow, QAction, QGroupBox, QLabel, QFileDialog, QTextEdit, QCheckBox
from note_recognition_app.image_processing.img_resizer import ResizeWithAspectRatio
import os
import sys
from pathlib import Path

import cv2

from PySide2 import QtCore


class LeftSide:
    def __init__(self, font, queue_foreground_to_background, queue_background_to_foreground):

        self.layout = QVBoxLayout()

        self._queue_foreground_to_background = queue_foreground_to_background
        self._queue_background_to_foreground = queue_background_to_foreground
        self._font = font

        self._default_img_location = os.path.abspath(
            os.path.join(
                str(Path(__file__).parent.parent.parent),
                'resources', 'input_images'))

        self._img_path = None

        self._img_label = QTextEdit("PLEASE SELECT AN IMAGE.")
        self._img_label.setFont(QFont("Arial", 36))
        self._img_label.setFixedWidth(400)
        self._img_label.setFixedHeight(570)
        self._img_label.setEnabled(False)

        self._retrain_flag = False

        self._left_side()

    def _left_side(self):
        self._img_select = QPushButton("Select an image.")
        self._img_select.setFixedWidth(200)
        self._img_select.setStyleSheet("background-color: green")
        self._img_select.setFont(self._font)
        self._img_select.sizeHint()
        self._img_select.clicked.connect(lambda c: self._img_selection())

        self._retrain_checkbox = QCheckBox("Retrain?")
        self._retrain_checkbox.setChecked(False)
        self._retrain_checkbox.stateChanged.connect(lambda: self._toggle_retrain())

        self._img_predict = QPushButton("Start.")
        self._img_predict.setFixedWidth(200)
        self._img_predict.setStyleSheet("background-color: gray")
        self._img_predict.setFont(self._font)
        self._img_predict.sizeHint()
        self._img_predict.clicked.connect(lambda c: self._img_prediction())
        self._img_predict.setEnabled(False)

        self.layout.addWidget(self._img_label)
        self.layout.addWidget(self._img_select)
        self.layout.addWidget(self._retrain_checkbox)
        self.layout.addWidget(self._img_predict)

    def _img_selection(self, ):
        file_name = QFileDialog.getOpenFileName(dir=self._default_img_location)
        file_name = file_name[0]
        if len(file_name) > 0:
            print(file_name)
            self._reset_layout(self.layout)
            self._img_path = file_name
            self._img_label = self._load_img_into_label()
            self._left_side()
            self._img_predict.setStyleSheet("background-color: green")
            self._img_predict.setEnabled(True)

    def _img_prediction(self):
        self._queue_foreground_to_background.put((self._img_path, self._retrain_flag))

    def _toggle_retrain(self):
        self._retrain_flag = not self._retrain_flag
        print(self._retrain_flag)

    def _reset_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                self._reset_layout(child.layout())

    def _load_img_into_label(self):
        img = cv2.imread(self._img_path, cv2.IMREAD_UNCHANGED)
        img = ResizeWithAspectRatio(img, width=400)

        try:
            trans_mask = img[:, :, 3] == 0
            img[trans_mask] = [255, 255, 255, 255]
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except IndexError as e:
            pass

        img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888).rgbSwapped()
        img_label = QLabel()
        img_label.setPixmap(QPixmap.fromImage(img))
        return img_label
