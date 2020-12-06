from note_recognition_app.image_processing.row_splitter import split_into_rows


def process_image(input_image_path):
    is_success = split_into_rows(input_image_path)

    if is_success is False:
        print("Row splitting not successful! Ending the program.")
        exit(-1)

