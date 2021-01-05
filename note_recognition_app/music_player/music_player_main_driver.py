from midiutil.MidiFile import MIDIFile
import re


def play(results, img_name):
    # Create  MIDI object
    midi_file = MIDIFile(1)  # Only 1 track.
    track = 0  # The track.
    time = 0  # Start at the beginning.
    midi_file.addTrackName(track, time, img_name)
    midi_file.addTempo(track, time, 120)
    results = match_notes_to_midi_values(results)
    time_signature = "4/4"
    barline_start = 0
    max_bar_length = 4
    bar_length = 0
    channel = 0
    volume = 100
    pitch = 60  # C4 (middle C)
    time = 0  # start on beat 0
    duration = 1  # 1 beat long
    for result in results:
        result_name = result[0]
        pitch = -1
        note_duration = -1

        if "clef" in result_name:
            pass  # TODO
        elif "time_signature_4-4" in result_name:
            time_signature = "4/4"
            max_bar_length = 4
        elif "time_signature_2-4" in result_name:
            time_signature = "2/4"
            max_bar_length = 2
        elif "time_signature_3-4" in result_name:
            time_signature = "3/4"
            max_bar_length = 3
        elif "UNKNOWN" in result_name:
            is_note = False
            note_pitch = -1
            try:
                int(result[1])
                is_note = True
                note_pitch = int(result[1])
            except ValueError:
                is_note = False

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

            if note_duration != -1 and note_pitch != -1:
                bar_length = bar_length + note_duration
                midi_file.addNote(track, channel, note_pitch, time, note_duration, volume)
                time = time + note_duration

        elif "barline" in result_name:
            if bar_length < max_bar_length:
                duration = max_bar_length - bar_length
                if duration > 0:
                    pitch = 0
                    midi_file.addNote(track, channel, pitch, time, duration, volume)
                    time = time + note_duration
            bar_length = 0

    with open(img_name[:-4] + ".mid", 'wb') as out_file:
        midi_file.writeFile(out_file)

    return


def match_notes_to_midi_values(results):
    for index, result in enumerate(results):
        result = list(result)
        results[index] = result
        number = -1
        if result[1] == "C3":
            number = 48
        elif result[1] == "D3":
            number = 50
        elif result[1] == "E3":
            number = 52
        elif result[1] == "F3":
            number = 53
        elif result[1] == "G3":
            number = 55
        elif result[1] == "A3":
            number = 57
        elif result[1] == "B3":
            number = 59
        elif result[1] == "C4":
            number = 60
        elif result[1] == "D4":
            number = 62
        elif result[1] == "E4":
            number = 64
        elif result[1] == "F4":
            number = 65
        elif result[1] == "G4":
            number = 67
        elif result[1] == "A4":
            number = 69
        elif result[1] == "B4":
            number = 71
        elif result[1] == "C5":
            number = 72
        elif result[1] == "D5":
            number = 74
        elif result[1] == "E5":
            number = 76
        elif result[1] == "F5":
            number = 77
        elif result[1] == "G5":
            number = 79
        elif result[1] == "A5":
            number = 81

        if number != -1:
            results[index][1] = number
    return results
