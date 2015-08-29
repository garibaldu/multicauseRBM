import numpy as np
import os, pickle, rbmpy.sampler, math, logging

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

def squash_images(imgs):
    squashed = np.array(imgs)
    old_shape = squashed.shape
    squashed = squashed.reshape(old_shape[0], old_shape[1] * old_shape[2])
    return squashed

def inflate_images(imgs):
    inflated = np.array(imgs)
    old_shape = inflated.shape
    size= math.sqrt(old_shape[1])
    inflated = inflated.reshape(old_shape[0], size, size)
    return inflated

class SquareToyData(object):


    def gen_square(self,xy,sq_shape, img_size):
        """Square image starting at i, of sq_size within img_size. i must be < (sq_size + img_size)"""
        img = np.zeros(img_size)
        x = xy[0]
        y = xy[1]
        x2 = x + sq_shape[0]
        y2 = y + sq_shape[1]
        img[x:x2,y:y2] = 1
        return img

    def gen_training(self,sq_shape, img_size):
        if img_size[0] != img_size[1]:
            logging.warn("Unsquashing will not work with none squares yet!")
        training = []
        for x in range(img_size[0]-(sq_shape[0]-1)):
            for y in range(img_size[1] - (sq_shape[1]-1)):
                training.append(self.gen_square((x,y), sq_shape, img_size))
        return np.array(training)
