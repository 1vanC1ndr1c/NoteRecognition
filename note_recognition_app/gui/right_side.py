import multiprocessing
import os
import sys
from pathlib import Path

from PyQt5.QtCore import QThread
from PySide2 import QtGui
from PySide2.QtGui import QFont, QIcon, Qt, QCloseEvent
from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy, QFrame, QMainWindow, \
    QTextEdit, QPushButton

from note_recognition_app.console_output.stdout_redirect import StreamRedirect
from note_recognition_app.gui.left_side import LeftSide
from note_recognition_app.gui.mediaplayer import MidiPlayer


class RightSide:
    def __init__(self, font):
        self._font = font

        self.layout = QVBoxLayout()

        self.stdout_info = QTextEdit()
        self.stdout_info.setFixedWidth(800)
        self.stdout_info.moveCursor(QtGui.QTextCursor.Start)
        self.stdout_info.ensureCursorVisible()
        self.stdout_info.setLineWrapColumnOrWidth(500)
        self.stdout_info.setLineWrapMode(QTextEdit.FixedPixelWidth)

        self.layout.addWidget(self.stdout_info)

        self._set_media_player()

        self._midi_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent.parent), 'results'))
        self._midi_file = None
        self._midi_player = MidiPlayer()

    def update_stdout_text_window(self, msg):
        cursor = self.stdout_info.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(msg)
        self.stdout_info.setTextCursor(cursor)
        self.stdout_info.ensureCursorVisible()

    def _set_media_player(self):
        self._resources_path = os.path.abspath(
            os.path.join(
                str(Path(__file__).parent.parent.parent),
                'resources', 'gui'))

        self._media_box = QHBoxLayout()

        self._play_button = QPushButton('PLAY')
        self._play_icon = QtGui.QPixmap(os.path.join(self._resources_path, 'play.png'))
        self._play_button.setIcon(self._play_icon)
        self._play_button.setEnabled(False)

        self._pause_button = QPushButton('PAUSE')
        self._pause_icon = QtGui.QPixmap(os.path.join(self._resources_path, 'pause.png'))
        self._pause_button.setIcon(self._pause_icon)
        self._pause_button.setEnabled(False)

        self._stop_button = QPushButton('STOP')
        self._stop_icon = QtGui.QPixmap(os.path.join(self._resources_path, 'pause.png'))
        self._stop_button.setIcon(self._stop_icon)
        self._stop_button.setEnabled(False)

        self._media_box.addWidget(self._play_button)
        self._media_box.addWidget(self._pause_button)
        self._media_box.addWidget(self._stop_button)
        self.layout.addLayout(self._media_box)

    def update_media_player(self, msg):
        self._midi_file = os.path.join(self._midi_path, msg[:-4] + '.mid')

        self._play_button.setEnabled(True)
        self._pause_button.setEnabled(True)
        self._stop_button.setEnabled(True)

    def _play_midi(self):
        self._midi_player.play()
