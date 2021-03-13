from PyQt5.QtCore import QObject, pyqtSignal


class StdReaderWorker(QObject):
    """
    Worker class called from GUI that continuously reads the stdout queue and sends a signal for the GUI update.
    """
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    new_msg = pyqtSignal(str)

    def __init__(self, queue, read_queue=True):
        super().__init__()
        self._queue = queue
        self.read_queue = read_queue

    def run(self):
        while self.read_queue is True:
            msg = self._queue.get()
            self.new_msg.emit(msg)
