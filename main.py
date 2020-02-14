from imageSegmentationFunctions.inputHandling import input_handling
from pathlib import Path


def main():
    img_name = "img05.png"
    parent = str(Path(__file__).parent)
    path = parent + "\\resources-input-images\\" + img_name
    input_handling(path)


if __name__ == "__main__":
    main()

# track = input.getMidiNotes()
# print("==============")
# output.createMidi(track)
