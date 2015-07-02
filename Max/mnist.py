

import os, struct
from array import array
import numpy as np
import random, math

TRAIN_FILE_NAME = 'train-images.idx3-ubyte'


def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
        print(fname_img)
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')

    with open(fname_lbl, 'rb') as flbl:
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        lbl = array("b", flbl.read())

    with open(fname_img, 'rb') as fimg:
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = array("B", fimg.read())

    imgs = np.array(img).reshape(size,rows,cols)
    imgs = np.where(imgs > 0, 1,0) # normalise the dataset
    labels = np.array(lbl)
    
    return __raw_to_dict(imgs, labels)


def __raw_to_dict(imgs,labels):
    digits_to_imgs = {}


    # TODO: List comprehension-fiy
    for labelidx in range(len(labels)):
        label = labels[labelidx]
        img = imgs[labelidx]

        if digits_to_imgs.get(label) is None:
            digits_to_imgs[label] = []
        digits_to_imgs[label].append(img)
        digits_to_imgs[label] = digits_to_imgs[label]
    
    return digits_to_imgs

def filename_for_label(digit_label , path = "."):
    return os.path.join(path,"{}s-img-dataset.npy".format(digit_label)) 




def main():
    for digit in digits_to_imgs:
        np.save(filename_for_label(digit,path = 'MNIST_Digits'), digits_to_imgs[digit])

if __name__ == '__main__':
    main()




