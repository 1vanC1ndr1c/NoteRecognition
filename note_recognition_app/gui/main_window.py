# import sys
#
# from PySide2 import QtCore
# from PySide2.QtGui import QFont, QIcon, Qt, QCloseEvent
# from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, \
#     QHBoxLayout, QSizePolicy, QFrame, QMenu, QMainWindow, QAction, QGroupBox
#
#
# def run_gui(queue_foreground_to_background, queue_background_to_foreground):
#     app = init_q_application()
#     gui = Gui(queue_foreground_to_background, queue_background_to_foreground)
#     gui.setGeometry(100, 100, 900, 400)
#     gui.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
#     # gui.showMaximized()
#     gui.show()
#     app.exec_()
#
#
# def _create_vertical_line():
#     vertical_line = QFrame()
#     vertical_line.setFixedWidth(20)
#     vertical_line.setFrameShape(QFrame.VLine)
#     vertical_line.setFrameShadow(QFrame.Sunken)
#     vertical_line.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
#     vertical_line.setMinimumHeight(300)
#     return vertical_line
#
#
# class Gui(QMainWindow):
#
#     def __init__(self, queue_foreground_to_background, queue_background_to_foreground):
#         super().__init__()
#         self._queue_foreground_to_background = queue_foreground_to_background
#         self._queue_background_to_foreground = queue_background_to_foreground
#
#         self._main_widget = QWidget()
#         self._parent_layout = QHBoxLayout()
#         self._font = QFont("Arial", 12)
#         self._main_widget.setFont(self._font)
#
#         self._left_layout = QVBoxLayout()
#         self._middle_line = _create_vertical_line()
#         self._right_layout = QVBoxLayout
#
#         self._parent_layout.addWidget(self._left_layout)
#         self._parent_layout.addWidget(self._middle_line)
#         self._parent_layout.addWidget(self._right_layout)
#
#         self._main_widget.setLayout(self._parent_layout)
#         self.setCentralWidget(self._main_widget)
#
#
# def init_q_application():
#     app = QApplication(sys.argv)
#     app_icon = QIcon()
#     app.setWindowIcon(app_icon)
#     app.setStyle("fusion")
#     app.setApplicationName("Note Recognition.")
#     return app
#
# if __name__ == '__main__':
#     run_gui('q1', 'q2')