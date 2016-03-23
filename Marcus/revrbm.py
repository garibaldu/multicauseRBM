import math, time
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rng
from scipy.special import expit as sigmoid
np.set_printoptions(precision=2)

def inverse_sigmoid(prob1):
    return np.log(prob1/(1-prob1))

class RBM(object):
    """ 
    An RBM has weights, visible biases, and hidden biases.

    You can either make a new one, or read an old one in from a .npz file.

    An RBM can be told to train itself, given some data set of visible patterns. eg: rbm.train(indata, learning_params)

    It can save itself in pickled form

    It can make pretty pics of itself in PNG.
    """

    def __init__(self, filename, num_hid=0, num_vis=0, DROPOUT=True):
        if (num_hid>0 and num_vis>0):
            self.name = filename
            self.num_hid = num_hid
            self.num_vis = num_vis
            self.W = np.asarray( 0.1*rng.normal(size = (num_hid, num_vis)),order= 'fortran' )
            self.hid_bias = 0.01 * rng.normal(size = (1, num_hid))
            self.vis_bias = 0.01 * rng.normal(size = (1, num_vis))
        else:
            print('Reading in pickled RBM from %s' % (filename))
            # read in from a saved npz file
            if not(filename.endswith('.npz')):
                filename = filename + '.npz'
            with np.load('./saved_nets/' + filename) as data:
                self.name = filename[:-4]
                self.W = data['W']
                self.hid_bias = data['hid_bias']
                self.vis_bias = data['vis_bias']
                self.num_hid = self.W.shape[0]
                self.num_vis = self.W.shape[1]
            print ('NAME: %s, is an RBM with %d hids and %d vis' % (self.name, self.num_hid, self.num_vis))

        self.DROPOUT = DROPOUT
        self.W_change = 0.0
        self.hid_bias_change = 0.0
        self.vis_bias_change = 0.0

        
    def rename(self, newname):
        """ give the RBM a new name """
        self.name = newname
        
    def pushup(self, vis_pats):
        """ push visible pats into hidden, AND DRAW BERNOULLI SAMPLES """
        hid_prob1 = sigmoid(np.dot(vis_pats, self.W.T) + self.hid_bias)
        return 1*(hid_prob1 > rng.random(size=hid_prob1.shape))

    def pushdown(self, hid_pats):
        """ push hidden pats into visible, BUT JUST CALC PROBS """
        if self.DROPOUT:
            dropout = rng.randint(2, size=(hid_pats.shape[0], self.num_hid))
            vis = np.dot(hid_pats*dropout, self.W) + self.vis_bias
        else:
            FACTOR = 0.5
            vis = np.dot(hid_pats, FACTOR*self.W) + self.vis_bias
        return vis
        # OR....  
        #return 1*(vis_prob1 > rng.random(size=vis_prob1.shape))


    def explainaway(self, other_phi, my_h, v):
        psi = np.dot(v - other_phi, self.W.T) + self.hid_bias
        #psi += (my_h - 0.5)*np.sum(self.W**2,axis=1)
        return psi

        
    def train(self, indata, num_iterations, rate, momentum, L1_penalty, minibatch_size):
        """
        Train the RBM's weights on the supplied data, using CD1 with momentum, an L1 penalty, and (optionally) dropout.
        """
        print('training with rate %.5f, momentum %.2f, L1 penalty %.6f, minibatches of %d' % (rate, momentum, L1_penalty, minibatch_size))
        announce_every = num_iterations / 5
        start = time.time()
        num_pats = indata.shape[0]

        for t in range(num_iterations+1):
            start_index = 0
            while start_index < num_pats-1:
                next_index = min(start_index + minibatch_size, num_pats)
                vis_minibatch = indata[start_index : next_index]
                start_index = next_index  # ready for next time

                # push visible pats into hidden
                hid_first = self.pushup(vis_minibatch)
                # Einstein alternative suggested by Paul Mathews.
                Hebb = np.einsum('ij,ik->jk', hid_first, vis_minibatch) 

                # push hidden pats into visible 
                vis_reconstruction = self.pushdown(hid_first)
                # push reconstructed visible pats back into hidden
                hid_second = self.pushup(vis_reconstruction)

                AntiHebb = np.einsum('ij,ik->jk', hid_second, vis_reconstruction) # sim to Hebb

                self.W_change = rate * (Hebb - AntiHebb)/minibatch_size  +  momentum * self.W_change
                self.W += self.W_change - L1_penalty * np.sign(self.W)

                # Now we have to do the visible and hidden bias weights as well.
                self.hid_bias_change = rate * (hid_first.mean(0) - hid_second.mean(0))   +  momentum * self.hid_bias_change
                self.hid_bias += self.hid_bias_change
                self.vis_bias_change = rate * (vis_minibatch.mean(0) - vis_reconstruction.mean(0))   + momentum * self.vis_bias_change
                self.vis_bias += self.vis_bias_change

                self.vis_bias = 0.0*self.vis_bias_change

            
            if (t % announce_every == 0): 
                C = np.power(self.pushdown(self.pushup(vis_minibatch)) - vis_minibatch, 2.0).mean()
                print ('Iteration %5d \t TIME (secs): %.1f,  RMSreconstruction: %.4f' % (t, time.time() - start, C))

        return

    def get_num_vis(self):
        return self.num_vis
    
    def get_num_hid(self):
        return self.num_hid

    def make_weights_figure(self):
        """ 
        reality-check by looking at the weights, and their updates, for some particular hidden units.
        """
        plt.clf()
        rows, cols = 7, 8

        i=0
        maxw = np.max(np.abs(self.W))
        for r in range(rows):
            for c in range(cols):
                i += 1
                plt.subplot(rows,cols,i)
                if (i == 1):  # the very first one will be the bias weights
                    img = self.vis_bias.reshape(28,28)
                    plt.text(0,-2,'bias', fontsize=8, color='red')
                else:
                    j = rng.randint(self.num_hid) # random choice of hidden node
                    img = self.W[j].reshape(28,28)
                    plt.text(0,-2,'hid %d' %(j), fontsize=8)

                plt.imshow(img, interpolation='nearest',cmap='RdBu', vmin=-maxw, vmax=maxw)
                # setting vmin and vmax there ensures zero weights aren't coloured.
                plt.axis('off')
        
        filename = '%s_weights.png' % (self.name)
        plt.savefig(filename)
        print('Saved figure named %s' % (filename))


    def show_patterns(self, vis_pats):
        num_pats = vis_pats.shape[0]
        num_rows, num_cols = 7, 10
        num_examples = num_rows*num_cols + 1
        Vis_test = np.copy(vis_pats[rng.randint(0, num_pats, size=(num_examples)), :])
        i = 0
        plt.clf()
        for r in range(num_rows):
            for c in range(num_cols):
                i += 1
                plt.subplot(num_rows,num_cols,i)
                plt.imshow(Vis_test[i].reshape(28,28), cmap='Greys', vmin=-1., vmax=1., interpolation='nearest')
                plt.axis('off')

        filename = '%s_visibles.png' % (self.name)
        plt.savefig(filename)
        print('Saved %s' % (filename))
        

    def make_dynamics_figure(self, indata, SCRAMBLE=False):
        if SCRAMBLE: # initialise with completely scrambled training pics.
            safe = np.copy(indata)
            num_pixels = indata.shape[1]
            for i in range(indata.shape[0]):
                img = safe[i]
                rand_order = rng.permutation(np.arange(num_pixels))
                indata[i] = img[rand_order]


        ## WATCH OUT FOR CONSEQUENCES!!!
        # self.DROPOUT = True
        ### WATCH OUT FOR CONSEQUENCES!!!

        num_pats = indata.shape[0]
        num_examples = 5
        eg_indices = rng.randint(0, num_pats, size=(num_examples))
        Vis_test = np.copy(indata[eg_indices, :])
        i = 0
        next_stop = 0
        num_rows = 6
        plt.clf()
        total_time = 0
        print('here we go...')
        for s in range(num_rows):
            print('doing alternating Gibbs Sampling until t=%d' % (next_stop))
            while total_time < next_stop:
                hid = self.pushup(Vis_test)
                Vis_test = self.pushdown(hid)
                total_time += 1
                
            for n in range(num_examples):
                i += 1
                plt.subplot(num_rows,num_examples,i)
                plt.imshow(Vis_test[n].reshape(28,28), cmap='Greys', vmin=-1., vmax=1., interpolation='nearest')
                plt.axis('off')
                plt.text(0,-2,'iter %d' %(total_time), fontsize=8)
            next_stop = max(1, next_stop) * 2  # wait X times longer each time before showing the next sample.

        filename = '%s_gibbs_chains.png' % (self.name)
        plt.savefig(filename)
        print('Saved %s' % (filename))


    def save_as_pickle(self, annotation=''):
        """
        Save this RBM's weights and biases.
        """
        name = self.name
        if len(annotation)>0: 
            name = name + annotation
        filename = './saved_nets/%s.npz' % (name)
        np.savez(filename, W=self.W, hid_bias=self.hid_bias, vis_bias=self.vis_bias)
        print('Saved the pickle of %s' % (filename))
        
def random_visibles_for_rbm(rbm, num_items):
    return np.random.randint(0,2,(num_items, rbm.get_num_vis()))

def random_hiddens_for_rbm(rbm, num_items):
    return np.random.randint(0,2,(num_items, rbm.get_num_hid()))

def random_hidden_for_rbm(rbm):
    return np.random.randint(0,2,rbm.get_num_hid())

def weights_into_hiddens(weights):
    num_vis = math.sqrt(weights.shape[1])
    return weights.reshape(weights.shape[0],num_vis, num_vis)

def load_mnist_digits(digits, dataset_size):
    vis_train_pats = flatten_dataset(load_mnist_digit(digits[0],dataset_size))
    for i in digits[1:]:
        vis_train_pats = np.vstack((vis_train_pats, flatten_dataset(load_mnist_digit(i,dataset_size))))
    # Now scramble the order.
    num_pats = vis_train_pats.shape[0]
    rand_order = rng.permutation(np.arange(num_pats))
    vis_train_pats = vis_train_pats[rand_order]
    # THE FOLLOWING WRITES LIST OF DIGIT IMAGES AS A CSV TO A PLAIN TXT FILE
    # np.savetxt(fname='mnist_digits.txt', X=vis_train_pats, fmt='%.2f', delimiter=',')
    vis_train_pats = vis_train_pats*2.0 - 1.0 # so range is somethigng...now.
    vis_train_pats /= 2.0
    print('visibles range from %.2f to %.2f' % (vis_train_pats.min(), vis_train_pats.max()))
    return vis_train_pats

def generate_smooth_bkgd(dataset_size):
    bkgd_imgs = np.ones((dataset_size, 28, 28), dtype=float)
    x = np.linspace(-1.0,1.0,28)
    y = np.linspace(-1.0,1.0,28)
    X, Y = np.meshgrid(x,y)
    print('dataset size: ', dataset_size)
    for i in range(dataset_size):
        xslope, yslope = .5*(2.*rng.rand()-1.), .5*(2*rng.rand()-1.)
        intercept = 0.5*(2.*rng.rand()-1.)
        img = xslope*X + yslope*Y + intercept
        img = img - np.min(np.ravel(img))
        img = img / np.max(np.ravel(img))
        bkgd_imgs[i] = 2*img - 1.0
    vis_train_pats = flatten_dataset(bkgd_imgs)
    #print('vis_train_pats shape is ', vis_train_pats.shape) 
    print('gradient visibles range from %.2f to %.2f' % (vis_train_pats.min(), vis_train_pats.max()))
    return vis_train_pats


def load_mnist_digit(digit, dataset_size):
    assert(digit >= 0 and digit < 10)
    with open("datasets/{}.npy".format(digit),'rb') as f:
        return np.load(f)[:dataset_size]
    
def flatten_dataset(images):
    smushed = images.copy()
    return smushed.reshape((smushed.shape[0], -1))

def show_example_images(pats, filename='examples.png'):
    rows = 7
    cols = 10
    i=0
    plt.clf()
    for r in range(rows):
        for c in range(cols):
            plt.subplot(rows,cols,i+1)
            j = rng.randint(0,len(pats))
            plt.imshow(pats[j].reshape(28,28), cmap='Greys', interpolation='nearest', vmin=-1.0, vmax=1.0)
            maxval = pats[j].max()
            minval = pats[j].min()
            plt.text(0,0,"%.1f, %.1f" %(minval, maxval), fontsize=7, color='b')
            plt.axis('off')
            i += 1
    plt.savefig(filename)
    print('Saved figure named %s' % (filename))


def make_2layer_dynamics_figure(L1, L2):
    ## WATCH OUT FOR CONSEQUENCES!!!
    L1.DROPOUT = False
    L2.DROPOUT = False
    ### WATCH OUT FOR CONSEQUENCES!!!

    num_examples = 5    
    mid_pats =  random_visibles_for_rbm(L2, num_examples)
    i = 0
    next_stop = 0
    num_rows = 6
    plt.clf()
    total_time = 0
    for s in range(num_rows):
        print ('alternating Gibbs to iter %d' % (next_stop))
        while total_time < next_stop:
            top_pats = L2.pushup(mid_pats)
            mid_pats = L2.pushdown(top_pats)
            total_time += 1
                
        vis_pats = L1.pushdown(mid_pats)
        for n in range(num_examples):
            i += 1
            plt.subplot(num_rows,num_examples,i)
            plt.imshow(vis_pats[n].reshape(28,28), cmap='Greys', vmin=-1., vmax=1., interpolation='nearest')
            plt.axis('off')
            plt.text(0,-2,'iter %d' %(total_time), fontsize=8)
        next_stop = max(1, next_stop) * 5  # wait X times longer each time before showing the next sample.

    filename = '%s_2layer_chains.png' % (L2.name)
    plt.savefig(filename)
    print('Saved %s' % (filename))
