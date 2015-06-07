import numpy as np
import plotter


def log_likelyhood_score(sample, target):
    """Lets find the log likelyhood"""
    # we need the actual visible pattern that we want to compute the score for
    # if vi|ha is 400 | 786
    # and actual vi is 400 | 786 we are in business
    score = (target * np.log(sample)) + ((1 - target) * np.log((1 - sample)))
    score_sum = score.sum()
    return score_sum



class plot_correction_decorator(object):

    def __init__(self, f):
        self.f = f

    def __call__(self, *args):
        """
        The __call__ method is not called until the
        decorated function is called.
        """
        
        result = self.f(*args)
        plotter.plot_weights(result[0].sum(1))
        return result
        



