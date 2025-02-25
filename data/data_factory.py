import numpy as np
import pandas as pd
import os


def extract_short_name(path):
    """
    Extract the short name from the path. The short name is the content inside the square brackets.
    :param path: File path
    :return: Extracted short name
    """
    try:
        return path.split('[')[1].split(']')[0]
    except IndexError:
        print(f"Invalid path format: {path}. Unable to extract short name.")
        return None


def data_provider(args):
    """
    Data provider function that loads and processes data according to the passed parameters.
    :param args: Object containing parameters such as data path, sequence length, and prediction length
    :return: Input of training data, output of training data, list of training data, list of test data
    """
    root_path = args.root_path
    data_path = args.data_path

    all_paths = [
        'battery_data_frames[Cell_1].csv',
        'battery_data_frames[Cell_2].csv',
        'battery_data_frames[Cell_3].csv',
        'battery_data_frames[Cell_4].csv'
    ]

    if args.data == 'CALCE':
        pass

    try:
        data_sequence = pd.read_csv(os.path.join(root_path, data_path))
        data_sequence = data_sequence[['capacity']]
        data_sequence = data_sequence.to_numpy()
    except FileNotFoundError:
        print(f"File not found: {os.path.join(root_path, data_path)}")
        return None, None, None, None

