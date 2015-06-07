import numpy as np

class RBM(object):


	def __init__(self, num_hid, num_vis, num_items):

		self.hidden = np.zeros(num_items, num_hid)
		seff.visible = np.zeros(num_items, num_vis)

		self.weights = np.random.normal(size = (num_hid, num_vis))

		self.visible_bias = np.random.normal(size = num_vis)
		self.hidden_bias = np.random.normal(size = num_hid)

		self.num_items = num_items