import numpy as np
import os

ROOT_DATA_PATH = "Datasets"

DATA_FILE_EXTENSION = "{}s-img-dataset.npy"


def full_path_for_name(name):
    return os.path.join(ROOT_DATA_PATH,DATA_FILE_EXTENSION.format(name)) 

def composite_datasets(set_a, set_b):
    return np.maximum(set_a, set_b)

def data_set_with_name(name, size = None):
    full_data_set = np.load(full_path_for_name(name))
    return full_data_set[:size]

