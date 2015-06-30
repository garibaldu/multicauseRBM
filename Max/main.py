from scipy.special import expit
from rbm import RBM
from sampler import VanillaSampler, PartitionedSampler
from trainer import VanillaTrainier
import numpy as np
import datasets, performance, plotter, mnist, pickle

import argparse

parser = argparse.ArgumentParser(description='Run some Restricted Bolzmann and Partitioned Restricted Boltzmann Sampling')
parser.add_argument('RBM File Names', metavar='N', type=argparse.FileType('rb'), nargs='+',
                   help='2 model filenames to load')
parser.add_argument('--NumSamples', metavar='S', type=int,
                   help='Number of Alternating Gibbs Samples to perform in Partitioned sampling technique')
parser.add_argument('--NumItems', metavar='I', type=int,
                   help='Size of the set we are sampling')

args = parser.parse_args()
dict_repr = vars(args)

def evaluate_models(models, num_samples, num_items):
	model_a = pickle.load(models[0])
	model_b = pickle.load(models[1])

	if num_items == None:
		num_items = 100
	if num_samples == None:
		num_samples = 100


	b = model_b.visible[:num_items]
	a = model_a.visible[:num_items]
	composite = datasets.composite_datasets(b, a)

	part_sampler = PartitionedSampler(model_a, model_b, num_items= num_items)
	van_a_sampler = VanillaSampler(model_a)
	van_b_sampler = VanillaSampler(model_b)

	vis_part_a, vis_part_b = part_sampler.reconstructions_given_visible(composite, num_samples = num_samples)
	vis_van_a = van_a_sampler.reconstruction_given_visible(composite)
	vis_van_b = van_b_sampler.reconstruction_given_visible(composite)

	vis_target_a = van_a_sampler.reconstruction_given_visible(a)
	vis_target_b = van_b_sampler.reconstruction_given_visible(b)
	 
	  
	score_part_a = performance.log_likelyhood_score(vis_part_a, vis_target_a)
	score_van_a = performance.log_likelyhood_score(vis_van_a, vis_target_a)

	score_part_b = performance.log_likelyhood_score(vis_part_b, vis_target_b)
	score_van_b = performance.log_likelyhood_score(vis_van_b, vis_target_b)

	print("Scores:\n\tPart_a{}\tVan_a{}\n\tPart_b{}\tVan_b{}\n\tA Win?{}\tB Win?{}".format(score_part_a, score_van_a, score_part_b, score_van_b, score_part_a > score_van_a, score_part_b > score_van_b))


evaluate_models(dict_repr["RBM File Names"], dict_repr["NumSamples"], dict_repr["NumItems"])





