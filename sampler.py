from scipy.special import expit
import numpy as np
from numpy import newaxis
from performance import plot_correction_decorator

class VanillaSampler(object):

    def __init__(self, rbm):
        self.rbm = rbm

    def bernouli_flip(self, weighted_sum):
        p = expit(weighted_sum) > np.random.rand(*weighted_sum.shape)
        return np.where(p, 1, 0)
    
    def visible_to_hidden(self, visible):
        return self.bernouli_flip(np.dot(visible, self.rbm.weights.transpose()) + self.rbm.hidden_bias)

    def hidden_to_visible(self, hidden):
        return expit(np.dot(hidden, self.rbm.weights) + self.rbm.visible_bias)

    def reconstruction_given_visible(self, visible):
        hid_given_vis = self.visible_to_hidden(visible)
        vis_given_hid = self.hidden_to_visible(hid_given_vis)
        return vis_given_hid 


class PartitionedSampler(VanillaSampler):

    def __init__(self, rbm_a, rbm_b, num_items = None):
        self.rbm_a = rbm_a
        self.rbm_b = rbm_b
        # TODO: Move the num items outside, it doesn't belong in the sampler!!!
        # It will just lead to issues where external data isn't cut the same way!
        if num_items == None:
            self.size = self.rbm_a.num_items
        else:
            self.size = num_items


    def visible_to_hidden(self, visible, num_samples, hidden_a = None, hidden_b = None ):
        # grab a slice of the hiddens and visible that are the correct size
        if(hidden_a == None):
            hidden_a = np.zeros((self.size, self.rbm_a.hidden.shape[1]))#self.rbm_a.hidden[:(self.size)]
        if(hidden_b == None):
            hidden_b = np.zeros((self.size, self.rbm_b.hidden.shape[1]))#self.rbm_b.hidden[:(self.size)]
        visible = visible[:self.size]

        vis_bias_a = self.rbm_a.visible_bias
        vis_bias_b = self.rbm_b.visible_bias
        hid_bias_a = self.rbm_a.hidden_bias
        hid_bias_b = self.rbm_b.hidden_bias


        weights_a = self.rbm_a.weights
        weights_b = self.rbm_b.weights
        weights_a_T = weights_a.transpose()
        weights_b_T = weights_b.transpose()

        for epoch in range(num_samples):

            phi_a = np.dot(hidden_a, weights_a) + vis_bias_a
            phi_b = np.dot(hidden_b, weights_b) + vis_bias_b

            if (np.mod(epoch,2) == 0): 
                print("{}% complete".format(epoch/num_samples * 100))

            correction_a, correction_b = calc_correction(hidden_a, hidden_b, weights_a, weights_b)
            """
            Apply the correction to the weighted sum into the hiddens
            """
            psi_a = np.dot(visible ,weights_a_T) + correction_a.sum(2) + hid_bias_a
            psi_b = np.dot(visible ,weights_b_T) + correction_b.sum(2) + hid_bias_b

            # now, do we turn on he hiddens? Bernoulli sample to decide
            hidden_a = self.bernouli_flip(psi_a)
            hidden_b = self.bernouli_flip(psi_b)

        return hidden_a, hidden_b

    def hidden_to_sample(self, hidden, rbm):
        return expit(np.dot(hidden,rbm.weights) + rbm.visible_bias)

    def reconstructions_given_visible(self, visible, num_samples):
        hid_a, hid_b = self.visible_to_hidden(visible, num_samples)
        vis_a = self.hidden_to_sample(hid_a, self.rbm_a)
        vis_b = self.hidden_to_sample(hid_b, self.rbm_b)
        return vis_a, vis_b



def build_hinge_func(x):
    return np.log(expit(x))
hinge = np.vectorize(build_hinge_func)
 
def calc_correction(hidden_a, hidden_b, weights_a, weights_b):        
    phi_a = np.dot(hidden_a, weights_a)[:,newaxis,:]
    phi_b = np.dot(hidden_b, weights_b)[:,newaxis,:]

    on_weights_a = (hidden_a[:,:,newaxis] * weights_a[newaxis,:,:])
    off_weights_a = (1 - hidden_a[:,:,newaxis]) * weights_a[newaxis,:,:]

    on_weights_b =  (hidden_b[:,:,newaxis] * weights_b[newaxis,:,:])
    off_weights_b = (1 - hidden_b[:,:,newaxis]) * weights_b[newaxis,:,:]
    
    j_off_a = phi_a - on_weights_a
    j_off_b = phi_b - on_weights_b
    j_on_a = phi_a + off_weights_a
    j_on_b = phi_b + off_weights_b
    
    correction_a = np.log(expit(j_off_a))  - np.log(expit(j_off_a + phi_b)) + np.log(expit(j_on_a + phi_b)) - np.log(expit(j_on_a))
    correction_b = np.log(expit(j_off_b))  - np.log(expit(j_off_b + phi_a)) + np.log(expit(j_on_b + phi_a)) - np.log(expit(j_on_b))
    
    return correction_a, correction_b
        



    


