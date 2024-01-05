from os import path


def abspath(file, *args):
    return path.abspath(path.join(path.dirname(file), *args))
