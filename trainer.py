import numpy as np
import rbm


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

