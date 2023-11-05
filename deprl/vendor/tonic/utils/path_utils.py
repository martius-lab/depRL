import os


def normalize_path_decorator(func):
    """
    Adapts the output of a function in an OS appropriate way.
    """

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, str):
            normalized_path = os.path.normpath(result)
            return normalized_path
        else:
            raise ValueError("Function does not return a string path.")

    return wrapper
