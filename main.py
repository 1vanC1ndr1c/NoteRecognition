from imageSegmentationFunctions.inputHandling import input_handling
from pathlib import Path


def main():

    img_name = "img04.jpg"
    parent = str(Path(__file__).parent)
    path = parent + ".\\resources_input_images\\" + img_name
    input_handling(path)


if __name__ == "__main__":
    main()

# track = input.getMidiNotes()
# print("==============")
# output.createMidi(track)

