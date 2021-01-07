import os
from pathlib import Path

from midiutil.MidiFile import MIDIFile

from note_recognition_app.console_output.console_output_constructor import construct_output


def construct_midi(results, img_name):
    """
    This function constructs and saves the midi file.
    :param results: Individual element names and their classifications.
    :param img_name: Input image name.
    """

    construct_output(indent_level="block",
                     message="Creating the '.midi' file."
                     .format(img_name))

    # Create  MIDI object
    midi_file = MIDIFile(1)  # Only 1 track.
    track = 0  # Track number.
    time = 0  # Start at the beginning.

    # Add  properties to the midi file.
    midi_file.addTrackName(track, time, img_name)
    midi_file.addTempo(track, time, 120)

    # Convert note names to their numerical values (i.e. C4 -> 60)
    results = match_notes_to_midi_values(results)

    time_signature = "4/4"  # Default time signature

    max_bar_length = 4  # Length of one bar.
    current_bar_length = 0  # Length of the current bar.
    channel = 0  # Channel number (only one channel used).
    volume = 100  # Volume will be set to 100.
    time = 0  # Start on beat 0.

    for result in results:  # Iterate through all the classified images.
        result_name = result[0]  # Get the image name.
        pitch = -1  # Set the default pitch to be negative (not allowed, error check).
        note_duration = -1  # Set the default note value to be negative (not allowed, error check).

        if "clef" in result_name:  # Check the clef type (not important for now).
            pass  # TODO
        # Check the time signature (not important right now).
        elif "time_signature_4-4" in result_name:
            time_signature = "4/4"
            max_bar_length = 4  # Change the maximum bar length accordingly.
        elif "time_signature_2-4" in result_name:
            time_signature = "2/4"
            max_bar_length = 2  # Change the maximum bar length accordingly.
        elif "time_signature_3-4" in result_name:
            time_signature = "3/4"
            max_bar_length = 3  # Change the maximum bar length accordingly.
        elif "UNKNOWN" in result_name:
            # Notes that were classified are marked as "UNKNOWN".
            is_note = False  # Check is it a note (not used right now).

            # Set the real duration of the current note.
            if result[2] == "1/16":
                note_duration = 1 / 4
            elif result[2] == "1/8":
                note_duration = 1 / 2
            elif result[2] == "1/4":
                note_duration = 1
            elif result[2] == "1/2":
                note_duration = 2
            elif result[2] == "1/1":
                note_duration = 4

            note_pitch = result[1]  # Get the note pitch.
            if note_duration != -1 and note_pitch in range(47, 82):  # Check if the data is correct.
                current_bar_length = current_bar_length + note_duration  # Update current bar length.
                # Add the note to the midi file.
                midi_file.addNote(track, channel, note_pitch, time, note_duration, volume)
                time = time + note_duration  # Update the timer.

        # Check if there are enough notes in the bar.
        elif "barline" in result_name:
            if current_bar_length < max_bar_length:
                # If notes are missing, add a rest to the bar, so that the rest if correctly aligned.
                duration = max_bar_length - current_bar_length
                if duration > 0:  # Check if the current bar duration is greater then maximum bar duration.
                    time = time + duration  # If it isn't, add a rest to the bar.
            current_bar_length = 0  # If a barline is reached, reset the current bar length.

    # Construct the path the the location where the generated midi files will be saved.
    midi_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent.parent), 'results'))
    midi_path = os.path.join(midi_path, img_name[:-4] + ".mid")
    with open(midi_path, 'wb') as out_file:
        # Save the midi file.
        midi_file.writeFile(out_file)

    construct_output(indent_level="block",
                     message="Creating the '.midi' file done."
                     .format(img_name))


def match_notes_to_midi_values(results):
    """
    This function replaces the note names with their values in a midi file.
    For example. "C4" becomes 60 (number).
    :param results: Images names and their classifications. One of the classifications is the note name that
        needs to be replaced with it's corresponding numerical value.
    :return: list: input where the note names are replaced by numerical values.
    """

    modulation = "None."  # Modulation modifier.
    for index, result in enumerate(results):  # Iterate through results.

        result = list(result)  # Convert triple into a list with 3 elements (easier handling).
        results[index] = result  # Save the new list in the place of the triple.
        note_numerical_value = -1  # Value that will replace the note name.

        if result[1] == "modulation_d_maj":  # Check if the image specifies any modulations.
            modulation = "D_MAJ"
        # TODO other modulations.

        # Get the numerical value for a specific note.
        if result[1] == "C3":
            note_numerical_value = 48
        elif result[1] == "D3":
            note_numerical_value = 50
        elif result[1] == "E3":
            note_numerical_value = 52
        elif result[1] == "F3":
            note_numerical_value = 53
        elif result[1] == "G3":
            note_numerical_value = 55
        elif result[1] == "A3":
            note_numerical_value = 57
        elif result[1] == "B3":
            note_numerical_value = 59
        elif result[1] == "C4":
            note_numerical_value = 60
        elif result[1] == "D4":
            note_numerical_value = 62
        elif result[1] == "E4":
            note_numerical_value = 64
        elif result[1] == "F4":
            note_numerical_value = 65
        elif result[1] == "G4":
            note_numerical_value = 67
        elif result[1] == "A4":
            note_numerical_value = 69
        elif result[1] == "B4":
            note_numerical_value = 71
        elif result[1] == "C5":
            note_numerical_value = 72
        elif result[1] == "D5":
            note_numerical_value = 74
        elif result[1] == "E5":
            note_numerical_value = 76
        elif result[1] == "F5":
            note_numerical_value = 77
        elif result[1] == "G5":
            note_numerical_value = 79
        elif result[1] == "A5":
            note_numerical_value = 81

        if modulation == "D_MAJ":  # Adjust numerical values according to the modulation.
            if "F" in result[1] or "C" in result[1]:
                note_numerical_value = note_numerical_value + 1
        # TODO add other modulations.

        if note_numerical_value != -1:  # If a note has a correct numerical value...
            # Replace the note name with the numerical value.
            results[index][1] = note_numerical_value

    return results  # Return the changed list.
