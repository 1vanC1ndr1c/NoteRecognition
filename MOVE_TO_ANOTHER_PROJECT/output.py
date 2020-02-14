import mido


def createMidi(track):
    mid = mido.MidiFile()
    track = track
    mid.tracks.append(track)

    track.append(mido.Message('note_on', note=64, velocity=64, time=32))
    track.append(mido.Message('note_off', note=64, velocity=127, time=32))

    mid.save('new_song.mid')

    for i, trck in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, trck.name))
        for msg in trck:  # for every line in a current track
            if str(msg).startswith("note_on"):  # get the information about the notes
                print(msg)
                print(msg)
