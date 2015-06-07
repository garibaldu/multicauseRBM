import numpy as np
import os

ROOT_DATA_PATH = "datasets"

DATA_FILE_EXTENSION = "{}s-img-dataset.npy"


def full_path_for_name(name):
    return os.path.join(ROOT_DATA_PATH,DATA_FILE_EXTENSION.format(name)) 

def composite_datasets(set_a, set_b):
    return np.maximum(set_a, set_b)

def data_set_with_name(name, size = None):
    full_data_set = np.load(full_path_for_name(name))
    return flatten_data_set(full_data_set[:size])

def flatten_data_set(imgs):
    squashed = np.array(imgs)
    old_shape = squashed.shape
    squashed = squashed.reshape(old_shape[0], old_shape[1] * old_shape[2])
    return squashed