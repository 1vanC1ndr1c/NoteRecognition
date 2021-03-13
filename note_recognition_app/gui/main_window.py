import multiprocessing
import sys

from PyQt5.QtCore import QThread
from PySide2 import QtGui
from PySide2.QtGui import QFont, QIcon, Qt, QCloseEvent
from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy, QFrame, QMainWindow, \
    QTextEdit

from note_recognition_app.console_output.stdout_redirect import StreamRedirect
from note_recognition_app.gui.left_side import LeftSide
from note_recognition_app.gui.std_reader import StdReaderWorker


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

        self._start_std_reader_thread()

        self._main_widget = QWidget()
        self._parent_layout = QHBoxLayout()
        self._font = QFont("Arial", 12)
        self._main_widget.setFont(self._font)

        self._left_layout = LeftSide(
            self._font,
            self._queue_foreground_to_background,
            self._queue_background_to_foreground).layout

        self._middle_line = _create_vertical_line()
        self._right_layout = QVBoxLayout()
        self.stdout_info = QTextEdit()
        self.stdout_info.setFixedWidth(800)
        self.stdout_info.moveCursor(QtGui.QTextCursor.Start)
        self.stdout_info.ensureCursorVisible()
        self.stdout_info.setLineWrapColumnOrWidth(500)
        self.stdout_info.setLineWrapMode(QTextEdit.FixedPixelWidth)
        self._right_layout.addWidget(self.stdout_info)

        self._left_widget = QWidget()
        self._left_widget.setFixedWidth(500)
        self._left_widget.setLayout(self._left_layout)
        self._parent_layout.addWidget(self._left_widget)
        self._parent_layout.addWidget(self._middle_line)
        self._parent_layout.addLayout(self._right_layout)
        self._parent_layout.addStretch()

        self._main_widget.setLayout(self._parent_layout)
        self.setCentralWidget(self._main_widget)

    def closeEvent(self, event: QCloseEvent):
        self.std_reader_worker.read_queue = False
        self._queue_foreground_to_background.put(("End.", False))
        event.accept()

    def _start_std_reader_thread(self):
        self.thread = QThread()
        self.std_reader_worker = StdReaderWorker(self._queue_stdout, read_queue=True)
        self.std_reader_worker.moveToThread(self.thread)
        self.thread.started.connect(self.std_reader_worker.run)
        self.std_reader_worker.finished.connect(self.thread.quit)
        self.std_reader_worker.finished.connect(self.std_reader_worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.std_reader_worker.new_msg.connect(self._update_stdout_text_window)
        self.thread.start()

    def _update_stdout_text_window(self, msg):
        cursor = self.stdout_info.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(msg)
        self.stdout_info.setTextCursor(cursor)
        self.stdout_info.ensureCursorVisible()


def init_q_application():
    app = QApplication(sys.argv)
    app_icon = QIcon()
    app.setWindowIcon(app_icon)
    app.setStyle("fusion")
    app.setApplicationName("Note Recognition.")
    return app


if __name__ == '__main__':
    # Queue where all the stdout messages will be redirected.
    queue_stdout = multiprocessing.Queue()
    sys.stdout = StreamRedirect(queue_stdout)
    run_gui('q1', 'q2', queue_stdout)
