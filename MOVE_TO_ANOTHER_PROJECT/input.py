import mido

mid = mido.MidiFile('resources/test3.mid')


def getMidiNotes():
    for i, track in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, track.name))
        for msg in track:  # for every line in a current track
            if str(msg).startswith("note_on"):  # get the information about the notes
                print(msg)
                vel = str(msg).split("velocity=", 1)[1]  # check when the note has started, instead of ended
                vel = vel.split(" ", 1)[0]
                if (vel != "0"):
                    note = str(msg).split("note=", 1)[1]  # get the substring with the note value
                    note = note.split(" ", 1)[0]  # final note value (numerical)
                    noteAsChar = "C"
                    # TODO replace with dictionary
                    if note == '60':
                        noteAsChar = "C"
                    elif note == '62':
                        noteAsChar = "D"
                    elif note == '64':
                        noteAsChar = "E"
                    elif note == '65':
                        noteAsChar = "F"
                    elif note == '67':
                        noteAsChar = "G"
                    elif note == '69':
                        noteAsChar = "A"
                    elif note == '71':
                        noteAsChar = "B"
                    elif note == '72':
                        noteAsChar = "C2"
                    print(noteAsChar)

    return track
