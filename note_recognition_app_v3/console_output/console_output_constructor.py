def construct_output(indent_level, message: str):
    output_message = ""

    if indent_level == "block":
        print("\n*{}*\n".format(message))
        return
    if indent_level == -1:
        print(message)
        return
    if indent_level == 0:
        print(message + " " + (100 - len(message)) * "=")
        return
    if indent_level > 0:
        print(indent_level * 5 * " " + message)
        return
    pass
