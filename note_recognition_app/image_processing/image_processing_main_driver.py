from note_recognition_app.image_processing.row_splitter import split_into_rows
from note_recognition_app.image_processing.single_element_template_matcher import extract_elements_by_template_matching
from note_recognition_app.info_output.output_constructor import construct_output


def process_image(input_image_path):
    """
    Main function for image processing.
    Calls on module for row splitting (first), and then module for individual elements extraction(second).
    :param input_image_path: Path to the image that needs to be processed.
    """
    img_name = input_image_path[input_image_path.rfind('\\') + 1:]  # Extract image name from the given path.
    construct_output(indent_level="block", message="Processing the input image ({}).".format(img_name))

    split_into_rows(input_image_path)  # Firstly, extract rows.
    extract_elements_by_template_matching(img_name)  # Then, extract elements from those rows.

    construct_output(indent_level="block", message="Input image processing done.")
