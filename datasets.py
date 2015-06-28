import numpy as np
import os, pickle, sampler

ROOT_DATA_PATH = "datasets"

DATA_FILE_EXTENSION = "{}s-img-dataset.npy"


def full_path_for_name(name):
    return os.path.join(ROOT_DATA_PATH,DATA_FILE_EXTENSION.format(name)) 

def composite_datasets(set_a, set_b):
    return np.maximum(set_a, set_b)

def join_datasets(set_a, set_b):
    return np.concatenate((set_a, set_b) ,axis = 0)

def data_set_with_name(name, size = None):
    full_data_set = np.load(full_path_for_name(name))
    return flatten_data_set(full_data_set[:size])

def labels_for_data_set(label, dataset):
	labels = np.empty(dataset.shape[0])
	labels.fill(label)
	return labels



def flatten_data_set(imgs):
    squashed = np.array(imgs)
    old_shape = squashed.shape
    squashed = squashed.reshape(old_shape[0], old_shape[1] * old_shape[2])
    return squashed


def model_file_name(base_name):
    return os.path.join('.', 'models','{}_model'.format(base_name))

def load_models(names):
    models = {}
    for i in range(len(names)):
        with open(model_file_name(names[i]), 'rb') as f:
            current_model = pickle.load(f)
        models[names[i]] = current_model
    return models

def train_test_sets(full_data_set, train_size=300, test_size=300):
    training = full_data_set[:train_size]
    test = full_data_set[train_size:test_size]
    return (training, test)

