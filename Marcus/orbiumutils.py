import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rng
from scipy.special import expit as sigmoid
np.set_printoptions(precision=2)
import time

def load_mnist_digits(digits, dataset_size):
    vis_train_pats = flatten_dataset(load_mnist_digit(digits[0],dataset_size))
    for i in digits[1:]:
        vis_train_pats = np.vstack((vis_train_pats, flatten_dataset(load_mnist_digit(i,dataset_size))))
    # Now scramble the order.
    num_pats = vis_train_pats.shape[0]
    rand_order = rng.permutation(np.arange(num_pats))
    vis_train_pats = vis_train_pats[rand_order]
    # THE FOLLOWING WRITES LIST OF DIGIT IMAGES AS A CSV TO A PLAIN TXT FILE
    # np.savetxt(fname='mnist_digits.txt', X=vis_train_pats, fmt='%.2f', delimiter=',')
    return vis_train_pats


def load_mnist_digit(digit, dataset_size):
    assert(digit >= 0 and digit < 10)
    with open("datasets/{}.npy".format(digit),'rb') as f:
        return np.load(f)[:dataset_size]
    
def flatten_dataset(images):
    smushed = images.copy()
    return smushed.reshape((smushed.shape[0], -1))

def show_example_images(pats):
    i=0
    plt.clf()
    for r in range(6):
        for c in range(6):
            plt.subplot(6,6,i+1)
            plt.imshow(pats[i].reshape(28,28), cmap='Greys', interpolation='nearest')
            plt.axis('off')
            i += 1
    filename = 'examples.png'
    plt.savefig(filename)
    print('Saved figure named %s' % (filename))

