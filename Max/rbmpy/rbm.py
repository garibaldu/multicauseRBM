import math

import numpy as np

from sklearn.neural_network import BernoulliRBM


class RBM(object):
    """A model class to represent a Restricted Boltzmann Machine (RBM), captures state and can be transformed from the Scikit learn RBM.


   Args:
        num_hid (int): The number of hidden units for the RBM
        num_vis (int): The number of visible units for the RBM. For a `x` by `y` image this is equivalent to `x * y`
        num_items (int): Number of items in the set the RBM will fit to.

    Attributes:
        weights (numpy array): The weight matrix that captures the model this RBM has been fitted to. Shape is (num_hid, num_vis)
        visible_bias (numpy array): The bias into the visible layer of the RBM, or the background noise
        hidden_bias (numpy array): The bias into the hidden layer of the RBM, or the background noise
    """


    def __init__(self, num_hid, num_vis, num_items):

        self.hidden = np.zeros((num_items, num_hid))
        self.visible = np.zeros((num_items, num_vis))

        self.weights = np.asarray( np.random.normal(size = (num_hid, num_vis)),order= 'fortran' )

        self.visible_bias = np.random.normal(size = num_vis)
        self.hidden_bias = np.random.normal(size = num_hid)

        self.num_items = num_items

    def num_hid(self):
        return self.hidden.shape[1]

    def num_vis(self):
        return self.visible.shape[1]


def create_from_sklearn_rbm(sklearnRBM, num_vis, num_items):
    rbm = RBM(sklearnRBM.n_components, num_vis, num_items)
    rbm.weights = np.array(sklearnRBM.components_)
    rbm.visible_bias = sklearnRBM.intercept_visible_
    rbm.hidden_bias = sklearnRBM.intercept_hidden_

def random_visibles_for_rbm(rbm):
    return np.random.randint(0,2,(rbm.num_items, rbm.num_vis()))

def random_hiddens_for_rbm(rbm):
    return np.random.randint(0,2,(rbm.num_items, rbm.num_hid()))

def random_hidden_for_rbm(rbm):
    return np.random.randint(0,2,rbm.num_hid())

def weights_into_hiddens(weights):
    num_vis = math.sqrt(weights.shape[1])
    return weights.reshape(weights.shape[0],num_vis,num_vis)
