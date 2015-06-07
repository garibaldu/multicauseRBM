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
        return self.bernouli_flip(np.dot(hidden, self.rbm.weights) + self.rbm.visible_bias)

    def reconstruction_given_visible(self, visible):
        hid_given_vis = self.visible_to_hidden(visible)
        vis_given_hid = expit(np.dot(hid_given_vis, self.rbm.weights) + self.rbm.visible_bias)
        return vis_given_hid 


class PartitionedSampler(VanillaSampler):

    def __init__(self, rbm_a, rbm_b, num_items = None):
        self.rbm_a = rbm_a
        self.rbm_b = rbm_b

        if num_items == None:
            self.size = self.rbm_a.num_items
        else:
            self.size = num_items


    def visible_to_hidden(self, visible, num_samples):
        # grab a slice of the hiddens and visible that are the correct size
        hidden_a = self.rbm_a.hidden[:(self.size)]
        hidden_b = self.rbm_b.hidden[:(self.size)]
        visible = visible[:self.size]

        vis_bias_a = self.rbm_a.visible_bias
        vis_bias_b = self.rbm_b.visible_bias
        hid_bias_a = self.rbm_a.hidden_bias
        hid_bias_b = self.rbm_b.hidden_bias


        for epoch in range(num_samples):

            phi_a = np.dot(hidden_a, self.rbm_a.weights) + vis_bias_a
            phi_b = np.dot(hidden_b, self.rbm_b.weights) + vis_bias_b

            if (np.mod(epoch,2) == 0): 
                print("{}% complete".format(epoch/num_samples * 100))

            correction_a, correction_b = calc_correction(hidden_a, hidden_b, self.rbm_a.weights, self.rbm_b.weights)
  
            """
            Apply the correction to the weighted sum into the hiddens
            """
            psi_a = np.dot(visible ,self.rbm_a.weights.transpose()) + correction_a.sum(2) + hid_bias_a
            psi_b = np.dot(visible ,self.rbm_b.weights.transpose()) + correction_b.sum(2) + hid_bias_b

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
    
    correction_a = hinge(j_off_a)  - hinge(j_off_a + phi_b) + hinge(j_on_a + phi_b) - hinge(j_on_a)
    correction_b = hinge(j_off_b)  - hinge(j_off_b + phi_a) + hinge(j_on_b + phi_a) - hinge(j_on_b)
    
    return correction_a, correction_b
        



    


