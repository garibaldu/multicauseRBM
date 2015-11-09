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


def plot_weights(weights, vis_dim = 28):
	subplot_size = int(math.sqrt(weights.shape[1]))
	n_hidden = weights.shape[0]
	n_vis = weights.shape[1]
	n_vis_dim = vis_dim

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
	labels, values = zip(*to_plot.items())

	indexes = np.arange(len(labels))
	width = 1
	plt.suptitle(title)
	plt.bar(indexes, values, width)
	plt.xticks(indexes + width * 0.5, labels, rotation='vertical')
	plt.show()


def image(data,title="",cmap = 'cool', show_colorbar = True, filename= None, color_range = None):
	plt.title = title
	vmin = None
	vmax = None
	if color_range:
		vmin, vmax = color_range
	plt.imshow(data, interpolation = 'nearest', cmap=cmap,vmin=vmin, vmax=vmax )
	if show_colorbar:
		plt.colorbar()

	if filename is not None:
		plt.savefig(filename)
	plt.show()

def images(data,title ="", titles = None, cmap = 'cool',filename = None, color_range = None, fig_size = None):

	num_cols = 5
	num_rows = math.ceil(data.shape[0]/num_cols)
	plots_so_far = 0
	vmin = None
	vmax = None
	if color_range:
		vmin, vmax = color_range



	ax = plt.gca()
	ax.text(1,1,title)
	if fig_size:
		plt.figsize = fig_size
	for r in range(num_rows):
		for c in range(num_cols):
			if plots_so_far < len(data):
				plt.subplot(num_rows,num_cols, plots_so_far+1)
				plt.axis('off')

				plt.imshow(data[plots_so_far], interpolation = 'nearest', cmap=cmap,vmin =vmin, vmax = vmax)

			else:
				break
			plots_so_far +=1
	# plt.tight_layout()
	if filename is not None:
		plt.savefig(filename)
	plt.show()


"""
Demo of a function to create Hinton diagrams.

Hinton diagrams are useful for visualizing the values of a 2D array (e.g.
a weight matrix): Positive and negative values are represented by white and
black squares, respectively, and the size of each square represents the
magnitude of each value.

Initial idea from David Warde-Farley on the SciPy Cookbook
"""

def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x,y),w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
