import numpy as np
import plotter, datasets, sampler, math
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.extmath import log_logistic


def log_likelyhood_score(sample, target):
    """Lets find the log likelyhood"""
    # we need the actual visible pattern that we want to compute the score for
    # if vi|ha is 400 | 786
    # and actual vi is 400 | 786 we are in business

    score = (target * np.log(sample)) + ((1 - target) * np.log((1 - sample)))
    return score


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
        

class Average_Decorator(object):

    def __init__(self,run_times = 3):
        
        self.run_times = run_times

    def __call__(self,f):

        def wrapped_f(*args, **kwargs):
            print("Inside wrapped_f()")
            results = []
            for i in range(self.run_times):
                results.append(f(*args, **kwargs))
            return results
            print("After f(*args)")
        return wrapped_f

        


class Result:

    def __init__(self, num_items, num_samples, rbm_a, rbm_b, data_a, data_b):

        self.rbm_a = rbm_a
        self.rbm_b = rbm_b
        self.num_items = num_items
        self.num_samples = num_samples
        self.part_sampler = sampler.PartitionedSampler(rbm_a, rbm_b, num_items= self.num_items)
        self.van_data_a_sampler = sampler.VanillaSampler(rbm_a)
        self.van_data_b_sampler = sampler.VanillaSampler(rbm_b)

        self.vis_target_a = self.van_data_a_sampler.reconstruction_given_visible(data_a)
        self.vis_target_b = self.van_data_b_sampler.reconstruction_given_visible(data_b)

        print("Constructing Composite Dataset")
        self.composite = datasets.composite_datasets(data_a, data_b)

    def calculate_result(self):
        self.run_vanilla()
        self.run_partitioned()
        self.imagewise_score()

    def run_vanilla(self):
        print("Generating Vanilla Samples")
        self.vis_van_a = self.van_data_a_sampler.reconstruction_given_visible(self.composite)
        self.vis_van_b = self.van_data_b_sampler.reconstruction_given_visible(self.composite)

    def run_partitioned(self, stored_hidden_interval = 10):
        self.stored_hidden_interval = stored_hidden_interval # number of samples between stores of the hidden layer 
        mini_batches =  math.floor(self.num_samples / self.stored_hidden_interval)                
        print("Generating Partitioned Reconstructions (This may take a while)")

        stored_hiddens = {}
        hid_a = None
        hid_b = None
        for batch in range(mini_batches):
            print("Running batch {} of {}".format(batch, mini_batches))
            hid_a, hid_b = self.part_sampler.visible_to_hidden(self.composite, num_samples = self.stored_hidden_interval,hidden_a = hid_a,hidden_b = hid_b)
            stored_hiddens[batch] = (hid_a, hid_b) 
            
        self.stored_hiddens = stored_hiddens

    def visibles_for_stored_hidden(self, iteration):
        a=  self.part_sampler.hidden_to_sample(self.stored_hiddens[iteration][0],self.rbm_a)
        b=  self.part_sampler.hidden_to_sample(self.stored_hiddens[iteration][1],self.rbm_b)
        return a,b

    def visibles_for_partitioned(self):
        return self.visibles_for_stored_hidden(len(self.stored_hiddens) - 1)

    def imagewise_score(self):
        part_vis_a, part_vis_b = self.visibles_for_partitioned()
        part_vis_a_score = log_likelyhood_score(part_vis_a, self.vis_target_a)
        part_vis_b_score = log_likelyhood_score(part_vis_b, self.vis_target_b)
        van_vis_a_score = log_likelyhood_score(self.vis_van_a, self.vis_target_a)
        van_vis_b_score = log_likelyhood_score(self.vis_van_b, self.vis_target_b)
        self.score_a = {"PART" : part_vis_a_score.sum(1), "VAN" : van_vis_a_score.sum(1)}
        self.score_b = {"PART" : part_vis_b_score.sum(1), "VAN" : van_vis_b_score.sum(1)}


    def win_images(self, score_a, score_b):
        part_a = score_a["PART"]
        part_b = score_b["PART"]
        van_a = score_a["VAN"]
        van_b = score_b["VAN"]

        win_a = np.compress((part_a >= van_a), self.composite, axis = 0)
        win_b = np.compress((part_b >= van_b), self.composite, axis = 0)
        return (win_a, win_b)


    def equal_images(self,score_a, score_b):
        part_a = score_a["PART"]
        part_b = score_b["PART"]
        van_a = score_a["VAN"]
        van_b = score_b["VAN"]

        win_a = np.compress((part_a == van_a), self.composite, axis = 0)
        win_b = np.compress((part_b == van_b), self.composite, axis = 0)
        return (win_a, win_b)

    def lose_images(self,score_a, score_b):
        part_a = score_a["PART"]
        part_b = score_b["PART"]
        van_a = score_a["VAN"]
        van_b = score_b["VAN"]

        win_a = np.compress((part_a < van_a), self.composite, axis = 0)
        win_b = np.compress((part_b < van_b), self.composite, axis = 0)
        return (win_a, win_b)

    def plot_various_images(self):

        win_a, win_b = self.win_images(self.score_a, self.score_b)
        lose_a, lose_b = self.lose_images(self.score_a, self.score_b)
        equal_a, equal_b = self.equal_images(self.score_a, self.score_b)
        plotter.plot(win_a)
        plotter.plot(lose_a)
        plotter.plot(equal_a)

