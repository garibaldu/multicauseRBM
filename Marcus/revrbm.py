import math, time, sys
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rng
from scipy.special import expit as sigmoid
np.set_printoptions(precision=2)

def inverse_sigmoid(prob1):
    return np.log(prob1/(1-prob1))

class RBM(object):
    """An RBM has weights, visible biases, and hidden biases.

    You can either make a new one, or read an old one in from a .npz file.

    An RBM can be told to train itself, given some data set of visible
    patterns. eg: rbm.train(indata, learning_params). The training can
    be via Contrastive Divergence, or as an auto-encoder.

    It can save itself in pickled form.

    It can make pretty pics of itself.

    """

    def __init__(self, filename, num_hid=0, num_vis=0, DROPOUT=True, hid_type='logistic'):
        f = open(filename+'.txt', 'a')
        if (num_hid>0 and num_vis>0):
            self.name = filename
            self.num_hid = num_hid
            self.num_vis = num_vis
            rng.seed(99)
            self.W = np.asarray( 0.1*rng.normal(size = (num_hid, num_vis)),order= 'fortran' )
            self.hid_bias = 0.001 * rng.normal(size = (1, num_hid))
            self.vis_bias = 0.001 * rng.normal(size = (1, num_vis))
        else:
            print('Reading in pickled RBM from %s' % (filename))
            f.write('Reading in pickled RBM from %s' % (filename))
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
            print('NAME: %s, is an RBM with %d hids and %d vis' % (self.name, self.num_hid, self.num_vis))
            f.write('NAME: %s, is an RBM with %d hids and %d vis' % (self.name, self.num_hid, self.num_vis))

        self.DROPOUT = DROPOUT
        self.hid_type = hid_type
        self.vis_type = 'linear'
        print ('dropout is %s, hidden units of type %s' %(self.DROPOUT, self.hid_type))
        f.write('dropout is %s, hidden units of type %s' %(self.DROPOUT, self.hid_type))
        f.close()
        
    def rename(self, newname):
        """ give the RBM a new name """
        self.name = newname
        
    def pushup(self, vis_pats, noise=True):
        """ push visible pats into hidden layer"""
        psi = np.dot(vis_pats, self.W.T) + self.hid_bias
        if self.hid_type == 'logistic': # BERNOULLI units
            hid_prob1 = sigmoid(psi)
            if noise == False:
                return hid_prob1
            elif noise == True:
                return 1*(hid_prob1 > rng.random(size=hid_prob1.shape))
        elif self.hid_type == 'relu':  # ReLU units
            if noise == True:
                psi = psi + rng.normal(0.,0.001+np.sqrt(sigmoid(psi)), size=psi.shape)
            return np.maximum(0.0, psi)


    def pushdown(self, hid_pats, noise=True):
        """ push hidden pats into visible """
        if self.DROPOUT:
            dropout = rng.randint(2, size=(hid_pats.shape[0], self.num_hid))
            vis = np.dot(hid_pats*dropout, self.W) + self.vis_bias
        else:
            FACTOR = 0.5
            vis = np.dot(hid_pats, FACTOR*self.W) + self.vis_bias
        if noise == True:
            vis += 0.5*rng.normal(size = (self.num_vis))
        return vis
        # OR....  
        #return 1*(vis_prob1 > rng.random(size=vis_prob1.shape))


    def explainaway(self, other_phi, my_h, v):
        psi = np.dot(v - other_phi, self.W.T) + self.hid_bias
        #psi += (my_h - 0.5)*np.sum(self.W**2,axis=1)
        return psi

        
    def train(self, indata, num_iterations, Loss, rate, momentum, L1_penalty, minibatch_size):
        """
        Train the RBM's weights on the supplied data, and (optionally) use dropout.
        Loss can be CD or AE (contrastive divergence or auto-encoder).
        """
        f = open(self.name+'.txt', 'a')
        print('training with Loss %s and L1 penalty %.6f' % (Loss, L1_penalty))
        print('rate %.5f, momentum %.2f, minibatches of %d' % (rate, momentum, minibatch_size))
        f.write('training with Loss %s and L1 penalty %.6f' % (Loss, L1_penalty))
        f.write('rate %.5f, momentum %.2f, minibatches of %d' % (rate, momentum, minibatch_size))
        announce_every = num_iterations / 5
        start = time.time()
        num_pats = indata.shape[0]
        W_change = 0.0
        hid_bias_change = 0.0
        vis_bias_change = 0.0


        for t in range(num_iterations+1):
            outputs = self.pushdown(self.pushup(indata,noise=False),noise=False)
            C = 0.5*np.sum((outputs - indata)**2)
            if (t % announce_every == 0):
                print ('Iteration %5d \t TIME (secs): %.1f,  RMSreconstruction: %.1f' % (t, time.time() - start, C/num_pats))
                f.write('Iteration %5d \t TIME (secs): %.1f,  RMSreconstruction: %.1f' % (t, time.time() - start, C/num_pats))


            start_index = 0
            C = 0.0
            ######## training loop starts
            while start_index < num_pats-1:
                next_index = min(start_index + minibatch_size, num_pats)
                vis_minibatch = indata[start_index : next_index]
                ndata = np.shape(vis_minibatch)[0] # how many in this minibatch
                start_index = next_index  # ready for next time

                if Loss == 'CD':
                    W_grad, hid_bias_grad =  self.CD_gradient(vis_minibatch, CD_steps=2)
                elif Loss == 'AE':
                    W_grad, hid_bias_grad =  self.autoencoder_gradient(vis_minibatch)

                W_change = rate * W_grad  +  momentum * W_change
                self.W += W_change - L1_penalty * np.sign(self.W)
                
                # Now we have to do the visible and hidden bias weights as well.
                hid_bias_change = rate * hid_bias_grad  + momentum * hid_bias_change
                self.hid_bias += hid_bias_change

                # IGNORING VISIBLE BIASES STILL???????????????
                # self.vis_bias_change = rate * (vis_minibatch.mean(0) - vis_reconstruction.mean(0))   + momentum * self.vis_bias_change
                # self.vis_bias += self.vis_bias_change

            ######## training loop ends
        
        f.close()
        return

    
    def CD_gradient(self, inputs, CD_steps=1):
        """This RBM, with this data, can calculate the gradient for its
        weights and biases under the CD loss. So do it...
        """
        ndata = np.shape(inputs)[0]
        assert (CD_steps > 0)

        # WAKE PHASE followed by HEBB
        # push visible pats into hidden
        hid_first = self.pushup(inputs)
        # (Einstein alternative suggested by Paul Mathews)
        Hebb = np.einsum('ij,ik->jk', hid_first, inputs) 

        # SLEEP PHASE followed by HEBB
        hiddens = hid_first
        for step in range(CD_steps):
            # push hidden pats into visible 
            vis_reconstruction = self.pushdown(hiddens, noise=True)
            # push reconstructed visible pats back into hidden
            hiddens = self.pushup(vis_reconstruction, noise=True)

        # the final step is noiseless.
        vis_reconstruction = self.pushdown(hiddens, noise=False)
        # push reconstructed visible pats back into hidden
        hiddens = self.pushup(vis_reconstruction, noise=False)

        hid_second = hiddens
                
        # push hidden pats into visible 
        vis_reconstruction = self.pushdown(hid_first, noise=False)
        # push reconstructed visible pats back into hidden
        hid_second = self.pushup(vis_reconstruction)

        AntiHebb = np.einsum('ij,ik->jk', hid_second, vis_reconstruction)
        
        weights_gradient = (Hebb - AntiHebb)/ndata
        hid_bias_gradient = hid_first.mean(0) - hid_second.mean(0)
        return weights_gradient, hid_bias_gradient
            

    def autoencoder_gradient(self,inputs):
        """This RBM, with this data, can calculate the gradient for its
        weights and biases under the auto-encoder loss. So do it...
        """
        ndata = np.shape(inputs)[0]
        targets = inputs # it's an autoencoder...
        hiddens = self.pushup(inputs, noise=False)
        outputs = self.pushdown(hiddens, noise=False)

        
        out_type = 'linear'
        # Different types of output neurons
        if self.vis_type == 'linear':
            deltao = (outputs-targets)
        elif self.vis_type == 'logistic':
            deltao = (outputs-targets)*outputs*(1.0-outputs)
        elif self.vis_type == 'softmax':
            deltao = (outputs-targets)*(outputs*(-outputs)+outputs) #WHAT??!
        else:
            print ("bogus vis_type")
        if self.hid_type == 'linear':
            deltah = np.dot(deltao,np.transpose(self.W))
        elif self.hid_type == 'relu':
            deltah = np.maximum(0,np.sign(hiddens)) * (np.dot(deltao,np.transpose(self.W)))
        elif self.hid_type == 'logistic':
            deltah = hiddens*(1.0-hiddens)*(np.dot(deltao,np.transpose(self.W)))
        else:
            print ("bogus hid_type")
                   
        w1_gradient = (np.dot(np.transpose(inputs),deltah))
        w2_gradient = (np.dot(np.transpose(hiddens),deltao))
        weights_gradient =  0.5*(w1_gradient.T + w2_gradient) / ndata
        hid_bias_gradient =  np.sum(deltah,0) / ndata
        #error = 0.5*np.sum((outputs-inputs)**2) / ndata
        #print ("\t\t error: %.1f" % error)
        return -1.0 * weights_gradient, -1.* hid_bias_gradient

    
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
                    j = i % self.num_hid #rng.randint(self.num_hid) # random choice of hidden node
                    img = self.W[j].reshape(28,28)
                    plt.text(0,-2,'h%d %.1f, %.1f' %(j, np.min(self.W[j]), np.max(self.W[j])), fontsize=8)

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
        for s in range(num_rows):
            #print('doing alternating Gibbs Sampling until t=%d' % (next_stop))
            while total_time < next_stop:
                hid = self.pushup(Vis_test, noise=True)
                Vis_test = self.pushdown(hid, noise=False)
                total_time += 1
                
            for n in range(num_examples):
                i += 1
                plt.subplot(num_rows,num_examples,i)
                plt.imshow(Vis_test[n].reshape(28,28), cmap='Greys',interpolation='nearest') #, vmin=-1., vmax=1.,
                plt.axis('off')
                plt.text(0,-2,'t=%d %.1f, %.1f' %(total_time, np.min(Vis_test[n]), np.max(Vis_test[n])), fontsize=8)
            next_stop = 2**s #max(1, next_stop) * 2  # wait X times longer each time before showing the next sample.

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
    vis_train_pats = vis_train_pats*2.0 - 1.0 # so range is something...now.
    vis_train_pats *= 2.0 # 0.5
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
    rows = 6
    cols = 6
    i=0
    plt.clf()
    for r in range(rows):
        for c in range(cols):
            plt.subplot(rows,cols,i+1)
            j = r+rows*c # rng.randint(0,len(pats))
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
