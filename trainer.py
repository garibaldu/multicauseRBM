import numpy as np


class VanillaTrainier(object):

	
	def __init__(self, rbm, sampler):
		self.rbm = rbm
		self.sampler = sampler

	def train(self, epochs):
		wake_vis = self.visible
		wake_hid = self.hidden

		sleep_vis = wake_vis
		sleep_hid = self.sampler.visible_to_hidden(sleep_vis)

		for epoch in range(0, epochs):

			wake_hid = self.sampler.visible_to_hidden(wake_vis)
			sleep_vis = self.sampler.visible_given_hidden(sleepH, sleepV.shape)
			sleepH = self.hidden_given_visible(sleepV, sleepH.shape)

			# over the whole training set
			hebbian_pos = self.hebbian(wakeV, wakeH)
			hebbian_neg = self.hebbian(sleepV, sleepH)

			# weight update
			# TODO: make sure the hids are all different and check mean(1)?????
			self.weights =  self.weights + self.learning_rate * (hebbian_pos - hebbian_neg).sum(1).transpose()
			# bias updates 
			self.visible_bias = self.visible_bias + self.learning_rate * (wakeV - sleepV).sum(0)
			self.visible_bias = np.mean(self.visible_bias) * np.ones(self.visible_bias.shape)

			self.hidden_bias =  self.hidden_bias + self.learning_rate * (wakeH - sleepH).sum(0)
			self.hidden_bias = np.mean(self.hidden_bias) * np.ones(self.hidden_bias.shape)

		self.hidden = wakeH
		print('finished')



	def hebbian(self, visible, hidden):
		return visible[:,:,np.newaxis] * hidden.[np.newaxis, :,:]