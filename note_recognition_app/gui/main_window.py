import multiprocessing
import sys

import threading

from PyQt5.QtCore import QThread
from PySide2 import QtGui
from PySide2.QtGui import QFont, QIcon, Qt, QCloseEvent
from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy, QFrame, QMainWindow, \
    QTextEdit

from note_recognition_app.console_output.stdout_redirect import StreamRedirect
from note_recognition_app.gui.left_side import LeftSide
from note_recognition_app.gui.queue_workers import ReaderWorker
from note_recognition_app.gui.right_side import RightSide


def run_gui(queue_foreground_to_background, queue_background_to_foreground, queue_stdout):
    app = init_q_application()
    gui = Gui(queue_foreground_to_background, queue_background_to_foreground, queue_stdout)
    gui.setGeometry(100, 100, 900, 400)
    gui.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
    # gui.showMaximized()
    gui.show()
    app.exec_()


def _create_vertical_line():
    vertical_line = QFrame()
    vertical_line.setFixedWidth(20)
    vertical_line.setFrameShape(QFrame.VLine)
    vertical_line.setFrameShadow(QFrame.Sunken)
    vertical_line.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
    vertical_line.setMinimumHeight(300)
    return vertical_line


class Gui(QMainWindow):

    def __init__(self, queue_foreground_to_background, queue_background_to_foreground, queue_stdout):
        super().__init__()

        self._queue_stdout = queue_stdout
        self._queue_foreground_to_background = queue_foreground_to_background
        self._queue_background_to_foreground = queue_background_to_foreground

        self._main_widget = QWidget()
        self._parent_layout = QHBoxLayout()
        self._font = QFont("Arial", 12)
        self._main_widget.setFont(self._font)

        self._left_widget = LeftSide(
            self._font,
            self._queue_foreground_to_background,
            self._queue_background_to_foreground).main_widget

        self._middle_line = _create_vertical_line()

        self._right_side = RightSide(self.font)
        self._right_layout = self._right_side.layout
        self._start_std_reader_thread()

        self._background_reader_thread = threading.Thread(target=self._bg_reading_function)
        self._background_reader_thread.start()

        self._parent_layout.addWidget(self._left_widget)
        self._parent_layout.addWidget(self._middle_line)
        self._parent_layout.addLayout(self._right_layout)
        self._parent_layout.addStretch()

        self._main_widget.setLayout(self._parent_layout)
        self.setCentralWidget(self._main_widget)

    def closeEvent(self, event: QCloseEvent):
        self.std_reader_worker.read_queue = False
        self._queue_background_to_foreground.put(("End.", False))
        self._background_reader_thread.join()
        self._queue_foreground_to_background.put(("End.", False))
        event.accept()

    def _start_std_reader_thread(self):
        self.std_thread = QThread()
        self.std_reader_worker = ReaderWorker(self._queue_stdout, read_queue=True)
        self.std_reader_worker.moveToThread(self.std_thread)
        self.std_thread.started.connect(self.std_reader_worker.run)
        self.std_reader_worker.finished.connect(self.std_thread.quit)
        self.std_reader_worker.finished.connect(self.std_reader_worker.deleteLater)
        self.std_thread.finished.connect(self.std_thread.deleteLater)
        self.std_reader_worker.new_msg.connect(self._right_side.update_stdout_text_window)
        self.std_thread.start()

    def _bg_reading_function(self):
        while True:
            msg = self._queue_background_to_foreground.get()
            if msg[0] == 'Success.':
                self._right_side.update_media_player(msg[1])
            if msg[0] == 'End.':
                break


def init_q_application():
    app = QApplication(sys.argv)
    app_icon = QIcon()
    app.setWindowIcon(app_icon)
    app.setStyle("fusion")
    app.setApplicationName("Note Recognition.")
    return app


if __name__ == '__main__':
    # Queue where all the stdout messages will be redirected.
    _queue_stdout = multiprocessing.Queue()
    sys.stdout = StreamRedirect(_queue_stdout)
    run_gui('q1', 'q2', _queue_stdout)
