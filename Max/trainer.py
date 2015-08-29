import numpy as np
import rbm, logging
from sampler import VanillaSampler
from progress import Progress
from scipy.special import expit


class VanillaTrainier(object):
    """Trainer that can knows how to update an RBM weights and hidden/visible states, requires a `Sampler`.

       Args:
            rbm (rbm.RBM): The RBM we are training.
            sampler (sampler.Sampler): The sampler used to generate the reconstructions for the RBM's training.


        Attributes:
            rbm (rbm.RBM): The rbm this instance is training.
            sampler (sampler.Sampler): The sampler for generating reconstructions for the RBM's training.
    """

    def __init__(self, rbm, sampler):
        self.rbm = rbm
        self.sampler = sampler

    def train(self, epochs, training ,learning_rate = 0.002):
        """
        Train the rbm provided in the init to fit the given data.

        Args:
            epochs (int): The number of times to go over the training set, assumes this number is at least equal to the training set size.
            training (numpy.array): The training set. The shape should match the RBM that the trainer was supplied.
            learning_rate (Optional(float)): RBM's learning_rate, used in hebbian learning.

        """
        self.rbm.visible = training
        wake_vis = training
        wake_hid = rbm.random_hiddens_for_rbm(self.rbm)

        sleep_vis = wake_vis
        sleep_hid = self.sampler.visible_to_hidden(sleep_vis)

        for epoch in range(0, epochs):

            wake_hid = self.sampler.visible_to_hidden(wake_vis)
            sleep_vis = self.sampler.hidden_to_visible(sleep_hid) # reconstruction based on training item
            sleep_hid = self.sampler.visible_to_hidden(sleep_vis) # hidden based on reconstruction


            hebbian_pos = self.__hebbian__(wake_vis, wake_hid)
            hebbian_neg = self.__hebbian__(sleep_vis, sleep_hid)

            # weight update
            # TODO: make sure the hids are all different and check mean(1)?????
            self.rbm.weights += learning_rate * (hebbian_pos - hebbian_neg).sum(0).transpose()

            # bias updates
            self.rbm.visible_bias = self.rbm.visible_bias + learning_rate * (wake_vis - sleep_vis).sum(0)
            self.rbm.visible_bias = np.mean(self.rbm.visible_bias) * np.ones(self.rbm.visible_bias.shape)

            self.rbm.hidden_bias =  self.rbm.hidden_bias + learning_rate * (wake_hid - sleep_hid).sum(0)
            self.rbm.hidden_bias = np.mean(self.rbm.hidden_bias) * np.ones(self.rbm.hidden_bias.shape)

        self.rbm.hidden = wake_hid


    def __hebbian__(self, visible, hidden):
        return visible[:,:,np.newaxis] * hidden[:, np.newaxis,:]



class ORBMTrainer(object):

    def __init__(self, rbm_a, rbm_b,sampler):
        self._check_shape(rbm_a, rbm_b)

        self.rbm_a = rbm_a
        self.rbm_b = rbm_b
        self.sampler = sampler
        self.progess_logger = None

    def _check_shape(self,a,b):
        if not a.num_vis() == b.num_vis():
            raise ValueError("RBMs must have equal/matching number of visible units!")

    def set_logging(self, epoch_freq):
        self.progess_logger = Progress(__name__)


    def train(self, epochs, training, learning_rate = 0.002, num_gibbs = 500,logging_freq = None):
        sleep_a_sampler = VanillaSampler(self.rbm_a)
        sleep_b_sampler = VanillaSampler(self.rbm_b)

        if logging_freq:
            self.progess_logger = Progress(__name__, epochs)
            self.progess_logger.set_percentage_update_frequency(logging_freq)

        # generate a random hidden pattern to start from
        rand_h_a = rbm.random_hiddens_for_rbm(self.rbm_a)
        rand_h_b = rbm.random_hiddens_for_rbm(self.rbm_b)
        # logging.warn("Do persisent contrastive divergence")
        # logging.warn("Ensure to deal with the hidden bias")

        h_a, h_b = self.sampler.v_to_h(rand_h_a, rand_h_b, training, num_gibbs = num_gibbs)
        v_a, v_b = self.sampler.h_to_v(h_a, h_b)
        sleep_h_a = np.copy(h_a)
        sleep_h_b = np.copy(h_b)
        sleep_v_a = np.copy(v_a)
        sleep_v_b = np.copy(v_b)

        for epoch in range(epochs):
            # wake phase
            h_a, h_b = self.sampler.v_to_h(h_a, h_b, training, num_gibbs = num_gibbs)
            v_a, v_b = self.sampler.h_to_v(h_a, h_b)

            phi_a = self.sampler.phi_vis(h_a, self.rbm_a.weights)
            phi_b = self.sampler.phi_vis(h_b, self.rbm_b.weights)

            d_w_a = np.dot(expit(phi_a).T, h_a).T
            d_w_b = np.dot(expit(phi_b).T, h_b).T

            # logging.warn("Fix the d_w to do the dot")

             # to apply the perceptron lr part of the lr we need to find phi_a_b
            phi_ab = expit(phi_a + phi_b)
            # logging.warn("Fix the phi_ab see page 4")
            d_w_a += np.dot((training - phi_ab).T, h_a).T
            d_w_b += np.dot((training - phi_ab).T, h_b).T


             # now sleep phase
            rand_v = rbm.random_visibles_for_rbm(self.rbm_a)
            # logging.warn("Ensure I sleep for the same lenght as the training ")
            # logging.warn("A and B are indepednat in the prior, so I should usea  VanillaSampler here!")
            sleep_h_a = sleep_a_sampler.visible_to_hidden(rand_v)
            sleep_h_b = sleep_b_sampler.visible_to_hidden(rand_v)

            sleep_v_a = sleep_a_sampler.hidden_to_visible(sleep_h_a)
            sleep_v_b = sleep_b_sampler.hidden_to_visible(sleep_h_b)

            d_w_a -= (sleep_v_a[:,np.newaxis,:] * sleep_h_a[:,:,np.newaxis]).sum(0)
            d_w_b -= (sleep_v_b[:,np.newaxis,:] * sleep_h_b[:,:,np.newaxis]).sum(0)

            self.rbm_a.weights += learning_rate * d_w_a
            self.rbm_b.weights += learning_rate * d_w_b

            self.rbm_a.hidden_bias += np.mean(learning_rate * (h_a - sleep_h_a).sum(0))
            self.rbm_a.hidden_bias += np.mean(learning_rate * (h_b - sleep_h_b).sum(0))

            if self.progess_logger:
                self.progess_logger.set_completed_units(epoch)

        self.progess_logger.finished()
