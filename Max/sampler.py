from scipy.special import expit
import numpy as np
from numpy import newaxis
from performance import plot_correction_decorator
import logging

class VanillaSampler(object):
    """
    Sampler, allows you to draw gibbs samples from a supplied rbm

    Args:
        rbm (rbm.RBM): The rbm to draw samples from.
    """

    def __init__(self, rbm):
        self.rbm = rbm

    def __bernouli_flip__(self, weighted_sum):
        p = expit(weighted_sum) > np.random.rand(*weighted_sum.shape)
        return np.where(p, 1, 0)
    
    def visible_to_hidden(self, visible, num_samples = 1):
        """
        Generate a hidden pattern given a visible one.
        """
        return self.__bernouli_flip__(np.dot(visible, self.rbm.weights.transpose()) + self.rbm.hidden_bias)

    def hidden_to_visible(self, hidden):
        """
        Generate a Visible pattern given a hidden one.
        """
        return expit(np.dot(hidden, self.rbm.weights) + self.rbm.visible_bias)

    def reconstruction_given_visible(self, visible):
        """
        Perform one gibbs alternation, taking a visible pattern and returning the reconstruction given that visible pattern. 
        """
        hid_given_vis = self.visible_to_hidden(visible)
        vis_given_hid = self.hidden_to_visible(hid_given_vis)
        return vis_given_hid 


class PartitionedSampler(VanillaSampler):
    """
    PartitionedSampler, uses new technique of applying correction to hidden update with the visibles clamped (as we aren't training).
    """

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
        if(hidden_a is None):
            hidden_a = np.zeros((self.size, self.rbm_a.hidden.shape[1]))#self.rbm_a.hidden[:(self.size)]
        if(hidden_b is None):
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
                logging.debug("{}% complete".format(epoch/num_samples * 100))

            correction_a, correction_b = calc_correction(hidden_a, hidden_b, weights_a, weights_b)
            """
            Apply the correction to the weighted sum into the hiddens
            """
            psi_a = np.dot(visible ,weights_a_T) + correction_a.sum(2) + hid_bias_a
            psi_b = np.dot(visible ,weights_b_T) + correction_b.sum(2) + hid_bias_b

            # now, do we turn on he hiddens? Bernoulli sample to decide
            hidden_a = self.__bernouli_flip__(psi_a)
            hidden_b = self.__bernouli_flip__(psi_b)

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
        


class ApproximatedSampler(object):

    def __init__(self, w_a, w_b, v_bias_a, v_bias_b):
        self.w_a = w_a
        self.w_b = w_b
        self.v_bias_a = v_bias_a
        self.v_bias_b = v_bias_b

    def __bernoulli_trial__(self,weighted_sum):
        p = weighted_sum > np.random.rand(*weighted_sum.shape)
        return np.where(p, 1,0)

    def v_to_v(self, h_a, h_b, v , num_gibbs = 100):
        generated_h_a, generated_h_b = self.v_to_h(h_a,h_b, v,num_gibbs = num_gibbs)
        v_a, v_b = self.h_to_v(generated_h_a, generated_h_b)
        return v_a, v_b

    def v_to_h(self, h_a, h_b, v , num_gibbs = 100):
        """return the hidden representations for the supplied visible pattern"""
        hid_a = h_a
        hid_b = h_b

        for epoch in range(num_gibbs):
            # get the bentness of the coin used for the bernoulli trial
            psi_a, psi_b = self.p_hid(hid_a, hid_b, self.w_a, self.w_b,v)
            hid_a = self.__bernoulli_trial__(psi_a)
            hid_b = self.__bernoulli_trial__(psi_b) 
        return hid_a, hid_b

    def h_to_v(self, h_a, h_b):
        phi_a, phi_b = self.p_vis(h_a, h_b, self.w_a, self.w_b)
        v_a = self.__bernoulli_trial__(phi_a)
        v_b = self.__bernoulli_trial__(phi_b)
        return v_a, v_b

    def p_vis(self, h_a, h_b, w_a, w_b):
        phi_a = (w_a.T * h_a).sum(1)
        phi_b = (w_b.T * h_b).sum(1)
        return expit(phi_a), expit(phi_b)

    def p_hid(self,h_a, h_b, w_a, w_b, v):
        """calculate the probability that for the supplied hiddens, they will activate vector-wise"""
        c_a, c_b = self.approx_correction(h_a, h_b, w_a, w_b,v)
        psi_a = (w_a * (v + c_a)).sum(1) + self.v_bias_a # of course this isn't really the correction it's more of an ammendent (? word)
        psi_b = (w_b * (v + c_b)).sum(1) + self.v_bias_b
        return expit(psi_a),expit(psi_b)  

    def approx_correction(self, h_a, h_b, w_a, w_b,v):
        col_hid_a = h_a.reshape(2,1) # we reshape the hiddens to be a column vector
        col_hid_b = h_b.reshape(2,1)
        phi_a = np.dot(w_a, h_a) - (w_a * col_hid_a) # effective phi, we subtract activations for that h_j
        phi_b = np.dot(w_b, h_b) - (w_b * col_hid_b)
        sig_A = phi_a + w_a/2
        sig_B = phi_b + w_b/2
        epsilon_a = np.dot(w_b,h_b)
        epsilon_b = np.dot(w_a,h_a)
        sig_AB = sig_A + epsilon_a
        sig_BA = sig_B + epsilon_b
        c_a = expit(sig_A) - expit(sig_AB)
        c_b = expit(sig_B) - expit(sig_BA)
        return c_a, c_b

    


