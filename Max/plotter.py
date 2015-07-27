# You're a wizard Harry plotter
import numpy as np
import matplotlib, math
import time, datetime
import matplotlib.pyplot as plt


class Plot(object):

	def __init__(self, data, title = None, subplot_titles = None, c_map = 'copper'):

		subplot_size = math.sqrt(data.shape[1])
		subplot_shape = (subplot_size, subplot_size)
	
		if len(data.shape) > 2:
			subplot_shape = (data.shape[1], data.shape[2])
			
		self.data = data
		self.title = title
		self.subplot_titles = subplot_titles

	def plot_all(self):
		# lets figure this out based on the shape passed
		plot(self.data, subplot_titles, title)

		i = 0
		for r in range(num_rows):
			for c in range(num_cols):
				if i < len(data):
					plot_subplot(grid_idx)
				else:
					break;
				i = i + 1 

		plt.show()




	def plot_subplot(self, grid_idx):
		plt.title = title
		plt.subplot(num_rows,num_cols, grid_idx + 1)
		plt.imshow(np.reshape(data[i],subplot_shape), interpolation = 'nearest', cmap = 'copper', vmin = 0, vmax = 1)
		plt.axis('off')
		if subplot_titles is not None:
			plt.title(subplot_titles[i])



		grid = np.random.rand(4, 4)

		fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6),
		                         subplot_kw={'xticks': [], 'yticks': []})

		fig.subplots_adjust(hspace=0.3, wspace=0.05)

		for ax, interp_method in zip(axes.flat, methods):
		    ax.imshow(grid, interpolation=interp_method)
		    ax.set_title(interp_method)

		plt.show()


class Plotter(object):

	def __init__(self):
		self.plots = []

	def add_plot(self, plot):
		self.plots.append(data)

	def plot_all(self):
		for plot in self.plots:
			plot.plot_all()


def plot_weights(weights):
	subplot_size = int(math.sqrt(weights.shape[1]))
	n_hidden = weights.shape[0]
	n_vis = weights.shape[1]
	n_vis_dim = 28

	W_per_h = weights.reshape(n_hidden,n_vis_dim, n_vis_dim)
	nc = int(math.sqrt(n_hidden))
	nr = n_hidden / nc
	i=0
	for r in range(int(nr)):
		for c in range(nc):
			if i<=n_hidden:
				plt.subplot(nr,nc,i+1)
				plt.imshow(W_per_h[i],interpolation='nearest', cmap='seismic')
				plt.axis('off')
				# plt.title('hid {}'.format(i))
			i = i+1

	plt.show()


def plot(data, subplot_titles = None, title = None,num_rows = 6, num_cols = 9):
	# if it's not the shape i expect

	subplot_size = math.sqrt(data.shape[1])
	subplot_shape = (subplot_size, subplot_size)
	
	if len(data.shape) > 2:
		subplot_shape = (data.shape[1], data.shape[2])

	i = 0
	for r in range(num_rows):
		for c in range(num_cols):
			if i < len(data):
				plt.title = title
				plt.subplot(num_rows,num_cols, i + 1)
				plt.imshow(np.reshape(data[i],subplot_shape), interpolation = 'nearest', cmap = 'copper', vmin = 0, vmax = 1)
				plt.axis('off')
				if subplot_titles is not None:
					plt.title(subplot_titles[i])
			i = i + 1 

	plt.show()

def save_plot(data, subplot_titles = None, title = None,num_rows = 6, num_cols = 9, filename = None):
	# if it's not the shape i expect
	if(filename == None):
		filename = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S') + ".png"

	subplot_size = math.sqrt(data.shape[1])
	subplot_shape = (subplot_size, subplot_size)
	
	if len(data.shape) > 2:
		subplot_shape = (data.shape[1], data.shape[2])

	i = 0
	for r in range(num_rows):
		for c in range(num_cols):
			if i < len(data):
				plt.title = title
				plt.subplot(num_rows,num_cols, i + 1)
				plt.imshow(np.reshape(data[i],subplot_shape), interpolation = 'nearest', cmap = 'copper', vmin = 0, vmax = 1)
				plt.axis('off')
				if subplot_titles is not None:
					plt.title(subplot_titles[i])
			i = i + 1 

	plt.savefig(filename)

def plot_matrix(matrix, columns = None, rows = None, title = None):
	# Add a table at the bottom of the axes
	print(title)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
	plt.axis('off')
	the_table = plt.table(cellText = matrix, colLabels = columns, rowLabels = rows, loc = "center")
	
	plt.show()



def print_matrix(matrix, titles = []):
	mat_str = ''
	for title in title:
		pass


	
	for i in range(len(matrix)):
		for j in range(len(matrix[i])):
			mat_str = "{}\t{}".format(mat_str, matrix[i][j])

		mat_str +=  '\n'

	print(mat_str)


def plot_dict(to_plot, title = "", size = None):
    keys = []
    vals = []
    for key in to_plot:
        keys.append(key)
        vals.append(to_plot[key])

    if size is not None:
   		plt.figure(figsize=(30,15))
    plt.title(title)
    plt.bar(range(len(vals)), vals, align='center')
    plt.xticks(range(len(keys)), keys, rotation='vertical')
    plt.show()





