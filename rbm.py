import numpy as np
from sklearn.neural_network import BernoulliRBM

class RBM(object):

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

def random_hiddens_for_rbm(rbm):
    return np.zeros((rbm.num_items, rbm.num_hid()))
