import math
import os
import threading
from pathlib import Path

from PySide2 import QtGui
from PySide2.QtWidgets import QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLineEdit

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
        self.midi_player = MidiPlayer()
        self._toggle_pause = True
        self._already_playing = False

        self._play_default_stylesheet = None
        self._pause_default_stylesheet = None
        self._stop_default_stylesheet = None

        self.end_signal = 'NO END.'
        self.player_timer_thread = threading.Thread(target=self._update_media_player_timer)
        self.player_timer_thread.start()

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
        self._play_default_stylesheet = self._play_button.styleSheet()
        self._play_button.setEnabled(False)
        self._play_button.clicked.connect(lambda c: self._play_midi())

        self._pause_button = QPushButton('PAUSE')
        self._pause_icon = QtGui.QPixmap(os.path.join(self._resources_path, 'pause.png'))
        self._pause_button.setIcon(self._pause_icon)
        self._pause_default_stylesheet = self._pause_button.styleSheet()
        self._pause_button.setEnabled(False)
        self._pause_button.clicked.connect(lambda c: self._pause())

        self._stop_button = QPushButton('STOP')
        self._stop_icon = QtGui.QPixmap(os.path.join(self._resources_path, 'stop.png'))
        self._stop_button.setIcon(self._stop_icon)
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(lambda c: self._stop())
        self._stop_button.setStyleSheet("QPushButton:pressed { background-color: red }")

        self._player_timer = QLineEdit('--/-- sec.')
        self._player_timer.setFixedWidth(200)
        self._player_timer.setEnabled(False)

        self._media_box.addWidget(self._play_button)
        self._media_box.addWidget(self._pause_button)
        self._media_box.addWidget(self._stop_button)
        self._media_box.addWidget(self._player_timer)
        self.layout.addLayout(self._media_box)

    def update_media_player(self, msg):
        self._midi_file = os.path.join(self._midi_path, msg[:-4] + '.mid')
        self.midi_player.load(self._midi_file)
        self._play_button.setEnabled(True)
        self._pause_button.setEnabled(True)
        self._stop_button.setEnabled(True)

    def _play_midi(self):
        self._play_button.setStyleSheet("background-color: green")
        self._pause_button.setStyleSheet(self._pause_default_stylesheet)

        self._toggle_pause = True
        if self._already_playing is False:
            self.midi_player.play()
            self._already_playing = True

    def _pause(self):
        self._play_button.setStyleSheet(self._play_default_stylesheet)
        self._pause_button.setStyleSheet("background-color: green")

        if self._toggle_pause is True:
            self.midi_player.pause()
            self._already_playing = False
        else:
            self.midi_player.unpause()
            self._already_playing = True

        self._toggle_pause = not self._toggle_pause

    def _stop(self):
        self._play_button.setStyleSheet(self._play_default_stylesheet)
        self._pause_button.setStyleSheet(self._pause_default_stylesheet)
        self._toggle_pause = True
        self.midi_player.stop()
        self._already_playing = False

    def _update_media_player_timer(self):
        while True:
            if self.end_signal == 'End.':
                break

            total = self.midi_player.song_duration
            if total != 0:
                current = self.midi_player.current_runtime
                if int(current) > math.floor(total):
                    current = total
                output = f'{current}/{total} sec.'
            else:
                output = '--/-- sec.'
            self._player_timer.setText(output)
