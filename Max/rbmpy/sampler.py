from scipy.special import expit
import numpy as np
from numpy import newaxis
from rbmpy.performance import plot_correction_decorator
import logging
from rbmpy.progress import Progress
from collections import Counter

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

    def hidden_to_visible(self, hidden, return_sigmoid = False):
        """
        Generate a Visible pattern given a hidden one.
        """
        if not return_sigmoid:
            return self.__bernouli_flip__(np.dot(hidden, self.rbm.weights) + self.rbm.visible_bias)
        else:
            return expit(np.dot(hidden, self.rbm.weights) + self.rbm.visible_bias)

    def reconstruction_given_visible(self, visible, return_sigmoid = False):
        """
        Perform one gibbs alternation, taking a visible pattern and returning the reconstruction given that visible pattern.
        """
        hid_given_vis = self.visible_to_hidden(visible)
        if not return_sigmoid:
            return  self.hidden_to_visible(hid_given_vis)
        else:
            return expit(np.dot(hid_given_vis, self.rbm.weights) + self.rbm.visible_bias)


    # now I should look at the dreams and see if they are kosher
    def dream(self, model, num_gibbs = 1000, return_sigmoid = False):
        if return_sigmoid:
            current_v = np.random.randn(model.visible.shape[1])
            dream_hid = np.random.rand(model.hidden.shape[1])
        else:
            current_v = np.random.randint(2, size= model.visible.shape[1])
            dream_hid = np.random.randint(2, size= model.hidden.shape[1])

        for i in range(num_gibbs):
            dream_hid = self.visible_to_hidden(current_v)
            current_v = self.hidden_to_visible(dream_hid, return_sigmoid = return_sigmoid)

        return current_v


def goodnight(model, sampler, hours_of_sleep, num_gibbs_per_hour):
    """Generate a dictionary of reconstructions to the number of times they occurred"""
    result_dict = Counter()
    v_prime = sampler.dream(model, num_gibbs_per_hour)
    reconstruction_dict = {} # the actual reconstructions that occurred
    for i in range(hours_of_sleep):
        v_prime = sampler.dream(model, num_gibbs_per_hour)
        result_dict[tuple(v_prime)] += 1
        reconstruction_dict[tuple(v_prime)] = v_prime
    return result_dict, reconstruction_dict


def orbm_goodnight(model_a, model_b, sampler, hours_of_sleep, num_gibbs_per_hour):
    """Generate a dictionary of reconstructions to the number of times they occurred"""
    result_dict = Counter()
    v_prime = sampler.dream(model_a, model_b, num_gibbs_per_hour)
    reconstruction_dict = {} # the actual reconstructions that occurred
    for i in range(hours_of_sleep):
        v_prime = sampler.dream(model_a, model_b, num_gibbs_per_hour)
        result_dict[tuple(v_prime)] += 1
        reconstruction_dict[tuple(v_prime)] = v_prime
    return result_dict, reconstruction_dict

class ContinuousSampler(VanillaSampler):
    """
    A continous flavour of the vanilla sampler, allows non-binary visible representation. Hiddens are still binary.
    """

    def hidden_to_visible(self, hidden, return_sigmoid = True):
        """I match the interface of VanillaSampler here, return_sigmoid has no effect,  as it will always return the sigmoid in the contrinous case."""
        return expit(np.dot(hidden, self.rbm.weights) + self.rbm.visible_bias)

class PartitionedSampler(VanillaSampler):
    """
    PartitionedSampler
    This is an older implementation of the full blown correction calculation. Works for 2 dimensional data.
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

            correction_a, correction_b = self.calc_correction(hidden_a, hidden_b, weights_a, weights_b)
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

    def calc_correction(self, hidden_a, hidden_b, weights_a, weights_b):
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

def build_hinge_func(x):
    return np.log(expit(x))
hinge = np.vectorize(build_hinge_func)

class ApproximatedSampler(object):
    """
    Sampler that operates on a single instance at a time, as opposed to over the whole
    dataset
    """


    def __init__(self, w_a, w_b, h_bias_a, h_bias_b):
        self.w_a = w_a
        self.w_b = w_b
        self.h_bias_a = h_bias_a
        self.h_bias_b = h_bias_b

    def __bernoulli_trial__(self,weighted_sum):
        p = weighted_sum > np.random.rand(*weighted_sum.shape)
        return np.where(p, 1,0)

    def dream(self, model_a, model_b, num_gibbs = 1000):
        a_vanilla = VanillaSampler(model_a)
        b_vanilla = VanillaSampler(model_b)

        a_dream_v = a_vanilla.dream(model_a, num_gibbs, return_sigmoid = False)
        b_dream_v = b_vanilla.dream(model_b, num_gibbs, return_sigmoid = False)


        a_dream_h = a_vanilla.visible_to_hidden(a_dream_v)
        b_dream_h = b_vanilla.visible_to_hidden(b_dream_v)

        phi_a =  np.dot(a_dream_h, model_a.weights)
        phi_b =  np.dot(b_dream_h, model_b.weights)
        sig_ab = expit(phi_a + phi_b)
        # print("phi_a {}\tphi_b {}\t\tdream_h_a {}\tdream_h_b {}\tSig_ab {}".format(phi_a, phi_b, a_dream_h, b_dream_h, sig_ab))
        return self.__bernoulli_trial__(sig_ab)

    def v_to_v(self, h_a, h_b, v , num_gibbs = 100, logging_freq = None):
        generated_h_a, generated_h_b = self.v_to_h(h_a,h_b, v,num_gibbs = num_gibbs, logging_freq = logging_freq)
        v_a, v_b = self.h_to_v(generated_h_a, generated_h_b)
        return v_a, v_b

    def v_to_h(self, h_a, h_b, v , num_gibbs = 100, logging_freq = 10):
        """return the hidden representations for the supplied visible pattern"""
        hid_a = h_a
        hid_b = h_b

        if logging_freq:
            progess_logger = Progress(self.__class__.__name__, num_gibbs)
            progess_logger.set_percentage_update_frequency(logging_freq)

        for epoch in range(num_gibbs):
            # get the bentness of the coin used for the bernoulli trial
            psi_a, psi_b = self.p_hid(hid_a, hid_b, self.w_a, self.w_b, v)
            hid_a = self.__bernoulli_trial__(psi_a)
            hid_b = self.__bernoulli_trial__(psi_b)
            if logging_freq:
                progess_logger.set_completed_units(epoch)
        return hid_a, hid_b

    def h_to_v(self, h_a, h_b):
        phi_a, phi_b = self.p_vis(h_a, h_b, self.w_a, self.w_b)
        v_a = self.__bernoulli_trial__(phi_a)
        v_b = self.__bernoulli_trial__(phi_b)
        return v_a, v_b

    def phi(self,w,h):
        return (w_a.T * h_a).sum(1)

    def p_vis(self, h_a, h_b, w_a, w_b):
        phi_a = (w_a.T * h_a).sum(1)
        phi_b = (w_b.T * h_b).sum(1)
        return expit(phi_a), expit(phi_b)

    def p_hid(self,h_a, h_b, w_a, w_b, v):
        """calculate the probability that for the supplied hiddens, they will activate vector-wise"""
        c_a, c_b = self.correction(h_a, h_b, w_a, w_b, v)
        psi_a = self.psi(w_a,v,c_a, self.h_bias_a)# of course this isn't really the correction it's more of an ammendent (? word)
        psi_b = self.psi(w_b,v,c_b, self.h_bias_b)
        return expit(psi_a),expit(psi_b)

    def psi(self,w,v, c, h_bias):
        """Calculate the raw (non-expit'd) value the supplied w,v,c, and bias."""
        return (w * (v + c)).sum(1) + h_bias

    def correction(self, h_a, h_b, w_a, w_b,v):
        col_hid_a = h_a.reshape(h_a.shape[0],1) # we reshape the hiddens to be a column vector
        col_hid_b = h_b.reshape(h_b.shape[0],1)
        phi_a = np.dot(h_a, w_a) - (w_a * col_hid_a) # effective phi, we subtract activations for that h_j
        phi_b = np.dot(h_b, w_b) - (w_b * col_hid_b)
        sig_A = phi_a + w_a/2
        sig_B = phi_b + w_b/2
        epsilon_a = np.dot(h_b,w_b)
        epsilon_b = np.dot(h_a,w_a)
        sig_AB = sig_A + epsilon_a
        sig_BA = sig_B + epsilon_b
        c_a = expit(sig_A) - expit(sig_AB)
        c_b = expit(sig_B) - expit(sig_BA)
        return c_a, c_b

class FullCorrection(ApproximatedSampler):

    """
    Single training sampler that works on an instance at time, but applies the
    full correction.
    """

    def correction(self, h_a, h_b, w_a, w_b,v):
        col_hid_a = h_a.reshape(h_a.shape[0],1) # we reshape the hiddens to be a column vector
        col_hid_b = h_b.reshape(h_b.shape[0],1)
        phi_a = np.dot(h_a, w_a)
        phi_b = np.dot(h_b, w_b)

        on_weights_a = (w_a * col_hid_a)
        off_weights_a = 1 - on_weights_a
        on_weights_b = (w_b * col_hid_b)
        off_weights_b= 1 - on_weights_b

        j_off_a = phi_a - on_weights_a
        j_off_b = phi_b - on_weights_b
        j_on_a = phi_a + off_weights_a
        j_on_b = phi_b + off_weights_b

        correction_a = np.log(expit(j_off_a))  - np.log(expit(j_off_a + phi_b)) + np.log(expit(j_on_a + phi_b)) - np.log(expit(j_on_a))
        correction_b = np.log(expit(j_off_b))  - np.log(expit(j_off_b + phi_a)) + np.log(expit(j_on_b + phi_a)) - np.log(expit(j_on_b))

        return correction_a, correction_b

class DirtySampler(ApproximatedSampler):

    def correction(self, h_a, h_b, w_a, w_b,v):

        phi_a = np.dot(h_a, w_a)
        phi_b = np.dot(h_b, w_b)
        sig_A = phi_a
        sig_B = phi_b
        epsilon_a = np.dot(h_b,w_b)
        epsilon_b = np.dot(h_a,w_a)

        sig_AB = sig_A + epsilon_a
        sig_BA = sig_B + epsilon_b
        c_a = expit(sig_A) - expit(sig_AB)
        c_b = expit(sig_B) - expit(sig_BA)
        return c_a, c_b

class LayerWiseApproxSampler(ApproximatedSampler):

    def p_hid(self,h_a, h_b,w_a, w_b,v):
        psi_a, psi_b = super().p_hid(h_a, h_b, w_a, w_b,v)
        updated_h_a = self.__bernoulli_trial__(psi_a)
        psi_a, psi_b = super().p_hid(updated_h_a, h_b, w_a, w_b,v)
        updated_h_b = self.__bernoulli_trial__(psi_b)
        psi_a, psi_b = super().p_hid(updated_h_a,updated_h_b, w_a, w_b,v)
        return psi_a, psi_b

class ContinuousApproxSampler(ApproximatedSampler):

    """
    A continous flavour of our ApproximatedSampler, allows nonbinary visible units.
    """

    def h_to_v(self, h_a, h_b):
        return super().p_vis(h_a, h_b, self.w_a, self.w_b)




# what a name! :(
class ApproximatedMulDimSampler(ApproximatedSampler):
    """
    The ApproximatedSampler that works over mutliple dimensions, useful for
    offline learning where we need to push the whole training set through at
    once.
    """

    def phi_vis(self, h, w):
        return np.dot(h, w)

    def p_vis(self, h_a, h_b, w_a, w_b):
        phi_a = self.phi_vis(h_a,w_a)
        phi_b = self.phi_vis(h_b,w_b)
        return expit(phi_a), expit(phi_b)

    def psi(self,w,v, c, h_bias):
        return np.dot((v * c.sum(1)), w.T) + h_bias

    def correction(self,h_a, h_b, w_a, w_b,v):
        phi_a = self.phi_i(h_a ,w_a)
        phi_b = self.phi_i(h_b ,w_b)
        sig_A = phi_a + w_a/2
        sig_B = phi_b + w_b/2
        epsilon_a = np.dot(h_b,w_b)[:,newaxis,:]
        epsilon_b = np.dot(h_a,w_a)[:,newaxis,:]
        sig_AB = sig_A + epsilon_a
        sig_BA = sig_B + epsilon_b
        c_a = expit(sig_A) - expit(sig_AB)
        c_b = expit(sig_B) - expit(sig_BA)
        return c_a, c_b

    def phi_i(self, h,w):
        col_hid = h.reshape(h.shape[0], h.shape[1], 1)# we reshape the hiddens to be a column vector
        # TODO: DOES THIS MAKE SENSE? IS THIS HOW I REMOVE THE THINGS@!!!!@@!
        phi_i = np.dot(h, w)[:,newaxis,:] - (w * col_hid) # effective phi, we subtract activations for that h_j
        return phi_i


class DirtyCorrectionMulDimSampler(ApproximatedMulDimSampler):

    def correction(self, h_a, h_b, w_a, w_b,v):

        phi_a = np.dot(h_a, w_a)
        phi_b = np.dot(h_b, w_b)
        sig_A = phi_a
        sig_B = phi_b
        epsilon_a = np.dot(h_b,w_b)[:,newaxis,:]
        epsilon_b = np.dot(h_a,w_a)[:,newaxis,:]

        sig_AB = sig_A + epsilon_a
        sig_BA = sig_B + epsilon_b
        c_a = expit(sig_A) - expit(sig_AB)
        c_b = expit(sig_B) - expit(sig_BA)
        return c_a, c_b

class ContinuousApproxMulDimSampler(ContinuousApproxSampler, ApproximatedMulDimSampler ):
    pass

class FullCorrectionMulDimSampler(ApproximatedMulDimSampler):

    def correction(self, h_a, h_b, w_a, w_b, v):
        phi_a = np.dot(h_a, w_a)[:,newaxis,:]
        phi_b = np.dot(h_b, w_b)[:,newaxis,:]

        on_w_a = (h_a[:,:,newaxis] * w_a[newaxis,:,:])
        off_w_a = (1 - h_a[:,:,newaxis]) * w_a[newaxis,:,:]

        on_w_b =  (h_b[:,:,newaxis] * w_b[newaxis,:,:])
        off_w_b = (1 - h_b[:,:,newaxis]) * w_b[newaxis,:,:]

        j_off_a = phi_a - on_w_a
        j_off_b = phi_b - on_w_b
        j_on_a = phi_a + off_w_a
        j_on_b = phi_b + off_w_b

        correction_a = np.log(expit(j_off_a))  - np.log(expit(j_off_a + phi_b)) + np.log(expit(j_on_a + phi_b)) - np.log(expit(j_on_a))
        correction_b = np.log(expit(j_off_b))  - np.log(expit(j_off_b + phi_a)) + np.log(expit(j_on_b + phi_a)) - np.log(expit(j_on_b))

        return correction_a, correction_b
