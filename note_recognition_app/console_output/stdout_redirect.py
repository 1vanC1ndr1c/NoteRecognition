class StreamRedirect:
    """
    A class that is used as a fake sys.stdout and redirects all the messages into a queue.
    That queue is read by GUI.
    """

    def __init__(self, queue):
        self._queue = queue

    def write(self, text):
        self._queue.put(text)

    def flush(self):
        pass
