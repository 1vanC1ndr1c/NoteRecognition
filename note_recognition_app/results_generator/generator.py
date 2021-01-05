import os
from pathlib import Path


def generate_results(img_name, note_names, note_durations):
    """
    This function matches the results with file names and sorts them so that it matches the original
    order (from the original image).
    :param img_name: Name of the input image.
    :param note_names: Values of note.
    :param note_durations: Note durations.
    :return: list: List containing the order and description of the elements on the image.
    """
    results = []  # A list that will be returned.

    # Path that contains the information about the original image.
    image_path = os.path.abspath(os.path.join(str(Path(__file__).parent.parent.parent), 'resources', 'input_images'))
    image_path = os.path.join(image_path, 'input_images_individual_elements')
    image_path = os.path.join(image_path, img_name[:-4])

    # Find all the unrecognized elements and their names.
    unrecognized_elements_path = os.path.join(image_path, "unrecognized")
    unrecognized_elements = os.listdir(unrecognized_elements_path)

    results_and_positions = []
    for el in unrecognized_elements:
        row_number = el[(el.index("_row") + len("_row")):(el.index("_slice"))]
        el_number = el[(el.index("slice") + len("slice")):]
        el_number = el_number[:el_number.index("_")]
        results_and_positions.append((el, row_number, el_number))
    results_and_positions.sort(key=lambda k: (int(k[1]), int(k[2])))
    if len(note_names) != len(note_durations) != len(unrecognized_elements):
        print("generator.py = Not enough elements were recognized!")

    for index, el in enumerate(results_and_positions):
        el = el[0]
        if el.endswith(".png"):
            results.append((el[len("img02_"):-len(".png")], note_names[index], note_durations[index]))

    # Find all the recognized elements and their names.
    recognized_elements_path = os.path.join(image_path, "recognized", img_name[:-4] + ".txt")

    with open(recognized_elements_path, 'r') as file:
        recognized_elements = file.readlines()
    if len(recognized_elements) == 0:
        print("generator.py = Error in reading image file info!")
        exit(-1)
    for index, content in enumerate(recognized_elements):
        if content.endswith("\n"):
            recognized_elements[index] = content[:-1]
    for index, el in enumerate(recognized_elements):
        el_name = el[(el.find("template_") + len("template_")):el.find(".png")]
        if el_name.startswith("Z_barline"):
            el_name = "barline"
        results.append((el[len("img02_"):-len(".png")], el_name, "-"))

    # Join the individual elements with their descriptions.
    results_and_positions = []
    for result in results:
        x = result[0]
        row_number = x[(x.index("row") + len("row")):x.index("_slice")]
        el_number = x[(x.index("slice") + len("slice")):]
        el_number = el_number[:el_number.index("_")]
        results_and_positions.append((result, row_number, el_number))

    # Sort the elements with the order found in the original image.
    results_and_positions.sort(key=lambda k: (int(k[1]), int(k[2])))

    results = []
    for x in results_and_positions:
        results.append(x[0])

    return results
