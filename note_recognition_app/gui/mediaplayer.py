import math
import threading
from datetime import datetime

import pygame
from mutagen.smf import SMF


class MidiPlayer:
    """
    Class that uses pygame to play midi files in the background.
    Methods of the class are called from GUI with corresponding buttons.
    """

    def __init__(self):
        # Mixer config.
        self.freq = 44100  # Audio CD quality.
        self.bitsize = -16  # Unsigned 16 bit.
        self.channels = 2  # 1 is mono, 2 is stereo.
        self.buffer = 1024  # Number of samples.

        self._mixer = pygame.mixer
        self._mixer.init(self.freq, self.bitsize, self.channels, self.buffer)
        self._mixer.music.set_volume(0.8)  # Optional volume 0 to 1.0

        # Variable that will save the midi file path that is being loaded.
        self._midi_file_path = None
        # Duration of the song (used for information on gui.)
        self.song_duration = 0
        # Flag that indicates if the song is playing.
        self._play = False
        # Flag that indicates if a new song is being loaded.
        self._new_song = False
        # Start time of the song.
        self._timer_start = 0
        # Used for calculating for how many seconds has the song been playing.
        self._seconds_in_day = 24 * 60 * 60
        # Current runtime indicates for how long the song has been playing.
        self.current_runtime = 0

        # A variable that ends the playing loop and shuts down the thread.
        self.end_signal = 'NO END.'
        # Start playing loop thread.
        self._loop_thread = threading.Thread(target=self._loop_thread_function)
        self._loop_thread.start()

    def load(self, midi_file_path):
        """ Song loader."""
        # Save the file path.
        self._midi_file_path = midi_file_path
        # Load the song.
        self._mixer.music.load(self._midi_file_path)
        # Get the song duration.
        self.song_duration = SMF(midi_file_path).info.length
        # Indicate that a new song was loaded.
        self._new_song = True
        # Set the flag that indicates that the song can be played.
        self._play = False

    def play(self):
        """ Play function is synonymous with un-pausing the song."""
        self.unpause()

    def pause(self):
        """Method that pauses the song."""
        # Set the flag that indicates that song should not be currently playing.
        self._play = False
        # Call the corresponding pygame.mixer function.
        self._mixer.music.pause()

    def unpause(self):
        """Method that un-pauses the song."""
        # Set the flag that indicates that song should be currently playing.
        self._play = True
        # Call the corresponding pygame.mixer function.
        self._mixer.music.unpause()

    def stop(self):
        """Method that stops the song."""
        # Set the flag that indicates that song should not be currently playing.
        self._play = False
        # Set that flag that indicates that a new song is being loaded
        # (effectively rewinding the song to the start position)
        self._new_song = True
        # Call the corresponding pygame.mixer function.
        self._mixer.music.stop()

    def _loop_thread_function(self):
        """
        Pygame must be used within a loop.
        This function provides the needed loop and manages the song and needed functionalities
        (stop,pause,unpause, play).
        """
        pause_time = 0  # Used to adjust current running time when the song is unpaused.
        current_time = 0  # Current time (datetime.now).
        self.current_runtime = 0  # Current runtime of the song.

        while True:  # Iterate forever.
            if self.end_signal == 'End.':  # Stop when given the end signal.
                return

            # If a new song is being loaded.
            if self._play is True and self._new_song is True:
                self._mixer.music.play()  # Play the song.
                self._timer_start = datetime.now()  # Mark the starting time.
                self._new_song = False  # Reset the 'new song' flag.
                pause_time = 0  # Reset the pause time.
                current_time = 0  # Reset current time.
                self.current_runtime = 0  # Reset current runtime.

            # If the song is paused and the song has not already started..
            if self._play is False and self._new_song is False and current_time != 0:
                # Mark the pause time and transform it into seconds.
                pause_time = datetime.now()
                pause_time = pause_time - current_time
                pause_time = divmod(pause_time.days * self._seconds_in_day + pause_time.seconds, 60)
                pause_time = pause_time[0] * 60 + pause_time[1]

            # If the song is being played, keep it in a play loop.
            while self._play is True and self._new_song is False:
                if self.end_signal == 'End.':  # Stop when given the end signal.
                    return
                pygame.time.wait(1)
                current_time = datetime.now()  # Get the current time.
                difference = current_time - self._timer_start  # Calculate the time since the start of the song.
                self.current_runtime = divmod(difference.days * self._seconds_in_day + difference.seconds, 60)
                # Transform it into seconds, adjust for pause time.
                self.current_runtime = self.current_runtime[0] * 60 + self.current_runtime[1] - pause_time
                if int(self.current_runtime) > math.floor(self.song_duration):
                    self._play = False  # Stop the song if the current runtime exceeds the total duration.
