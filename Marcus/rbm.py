import math, time
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rng
from scipy.special import expit as sigmoid
np.set_printoptions(precision=2)
from orbiumutils import *



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
            with np.load(filename) as data:
                self.name = filename[:-4]
                self.W = data['W']
                self.hid_bias = data['hid_bias']
                self.vis_bias = data['vis_bias']
                self.num_hid = self.W.shape[0]
                self.num_vis = self.W.shape[1]
            print ('%d hids, and %d vis in %s' % (self.num_hid, self.num_vis, self.name))

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
        vis_prob1 = sigmoid(np.dot(hid_pats, self.W) + self.vis_bias)
        return vis_prob1  # OR....  1*(v_prob1 > rng.random(size=v_prob1.shape))

        
    def train(self, indata, num_iterations, rate, momentum, L1_penalty, minibatch_size):
        """
        Train the RBM's weights on the supplied data, using CD1 with momentum, an L1 penalty, and (optionally) dropout.
        """
        announce_every = num_iterations / 5
        start = time.time()
        num_pats = indata.shape[0]
        rand_order = rng.permutation(np.arange(num_pats))

        for t in range(num_iterations+1):
            if self.DROPOUT:
                dropout = rng.randint(2, size=(minibatch_size, self.num_hid))
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
                if self.DROPOUT:
                    hid_first = hid_first * dropout
                vis_reconstruction = self.pushdown(hid_first)

                # push reconstructed visible pats back into hidden
                hid_second = self.pushup(vis_reconstruction)

                AntiHebb = np.einsum('ij,ik->jk', hid_second, vis_reconstruction) 

                self.W_change = rate * (Hebb - AntiHebb)/minibatch_size  +  momentum * self.W_change
                self.W += self.W_change - L1_penalty * np.sign(self.W)

                # Now we have to do the visible and hidden bias weights as well.
                self.hid_bias_change = rate * (hid_first.mean(0) - hid_second.mean(0))   +  momentum * self.hid_bias_change
                self.hid_bias += self.hid_bias_change
                self.vis_bias_change = rate * (vis_minibatch.mean(0) - vis_reconstruction.mean(0))   +  momentum * self.vis_bias_change
                self.vis_bias += self.vis_bias_change
            
            if (t % announce_every == 0): 
                C = np.power(vis_reconstruction - vis_minibatch, 2.0).mean()
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
        nc = 5
        plt.clf()
        plt.subplot(2,nc,1)
        plt.imshow(self.vis_bias.reshape(28,28), interpolation='nearest',cmap='Blues')
        plt.text(0,-2,'bias weights')
        plt.axis('off')

        plt.subplot(2,nc, 1 + nc)
        plt.imshow(self.vis_bias_change.reshape(28,28), interpolation='nearest',cmap='Reds')
        plt.text(0,-2,'bias wgt changes')
        plt.axis('off')
        col_counter = 1
        maxw = np.max(np.abs(self.W))
        maxc = np.max(np.abs(self.W_change))
        for c in range(2, nc + 1):
            i = rng.randint(self.num_hid) # random choice of hidden node
    
            plt.subplot(2, nc, 1 + col_counter)
            plt.imshow(self.W[i].reshape(28,28), interpolation='nearest',cmap='RdBu', vmin=-maxw, vmax=maxw)
            plt.text(0,-2,'hid%d wgts' %(i))
            plt.axis('off')
    
            plt.subplot(2, nc, 1 + nc + col_counter)
            plt.imshow(self.W_change[i].reshape(28,28), interpolation='nearest',cmap='RdBu', vmin=-maxc, vmax=maxc)
            plt.text(0,-2,'hid%d changes' %(i))
            plt.axis('off')
            col_counter += 1

        filename = '%s_weights.png' % (self.name)
        plt.savefig(filename)
        print('Saved figure named %s' % (filename))


    def show_patterns(self, vis_pats):
        num_pats = vis_pats.shape[0]
        num_rows, num_cols = 5, 6
        num_examples = num_rows*num_cols + 1
        Vis_test = np.copy(vis_pats[rng.randint(0, num_pats, size=(num_examples)), :])
        i = 0
        plt.clf()
        for r in range(num_rows):
            for c in range(num_cols):
                i += 1
                plt.subplot(num_rows,num_cols,i)
                plt.imshow(Vis_test[i].reshape(28,28), cmap='Greys', vmin=0., vmax=1., interpolation='nearest')
                plt.axis('off')

        filename = '%s_visibles.png' % (self.name)
        plt.savefig(filename)
        print('Saved %s' % (filename))
        
    def make_dynamics_figure(self, indata):
        num_pats = indata.shape[0]
        num_examples = 5
        Vis_test = np.copy(indata[rng.randint(0, num_pats, size=(num_examples)), :])
        if self.DROPOUT:
            dropout = rng.randint(2, size=(num_pats, self.num_hid))
        i = 0
        num_Gibbs = 0
        num_rows = 8
        plt.clf()
        for s in range(num_rows):
            for t in range(num_Gibbs):
                hid = self.pushup(Vis_test)
                Vis_test = self.pushdown(hid)
                
            for n in range(num_examples):
                i += 1
                plt.subplot(num_rows,num_examples,i)
                plt.imshow(Vis_test[n].reshape(28,28), cmap='Greys', vmin=0., vmax=1., interpolation='nearest')
                plt.axis('off')
        
            num_Gibbs = max(1, num_Gibbs * 4)  # wait X times longer each time before showing the next sample.

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
        filename = '%s.npz' % (name)
        np.savez(filename, W=self.W, hid_bias=self.hid_bias, vis_bias=self.vis_bias)
        print('Saved the pickle of %s' % (filename))
        
#-----------------------------------------------------------    
def random_visibles_for_rbm(rbm, num_items):
    return np.random.randint(0,2,(num_items, rbm.get_num_vis()))

def random_hiddens_for_rbm(rbm, num_items):
    return np.random.randint(0,2,(num_items, rbm.get_num_hid()))

def random_hidden_for_rbm(rbm):
    return np.random.randint(0,2,rbm.get_num_hid())

def weights_into_hiddens(weights):
    num_vis = math.sqrt(weights.shape[1])
    return weights.reshape(weights.shape[0],num_vis, num_vis)
