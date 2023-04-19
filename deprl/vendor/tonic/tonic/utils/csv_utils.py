import csv
import os


def check_if_csv_has_updated(csv_path, last_update):
    """
    Checks if the csv file has been updated since the last update.
    """
    current_update = os.path.getmtime(csv_path)
    if current_update > last_update:
        return current_update, True
    return current_update, False


def load_csv(csv_path):
    """
    Loads the csv file and returns the data.
    """
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


def load_csv_to_dict(csv_path):
    """
    Converts the csv file to a dictionary.
    """
    data = load_csv(csv_path)
    keys = data[0]
    data = data[1:]
    data = [list(map(float, x)) for x in data if x[0] != "None"]
    data = {k: [x[idx] for x in data] for idx, k in enumerate(keys)}
    return data
