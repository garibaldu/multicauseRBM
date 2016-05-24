import pylab as pl
import numpy as np
import mlp_autoenc 
import cPickle, gzip
import numpy.random as rng

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
    #vis_train_pats = vis_train_pats*2.0 - 1.0 # so range is somethigng...now.
    print('visibles range from %.2f to %.2f' % (vis_train_pats.min(), vis_train_pats.max()))
    return vis_train_pats


def load_mnist_digit(digit, dataset_size):
    assert(digit >= 0 and digit < 10)
    with open("../Marcus/datasets/{}.npy".format(digit),'rb') as f:
        return np.load(f)[:dataset_size]
    
def flatten_dataset(images):
    smushed = images.copy()
    return smushed.reshape((smushed.shape[0], -1))

# Read the dataset in (code from sheet)
train_in = load_mnist_digits([5], 200)
test_in = load_mnist_digits([5], 200)
#f = gzip.open('/Users/srmarsla/Teaching/372/Labs/Code/Pcn/mnist.pkl.gz','rb')
#tset, vset, teset = cPickle.load(f)
#f.close()


train_tgt = train_in # we're doing autoencoding...
print "ranges: ", np.min(train_tgt), np.max(train_tgt)
test_tgt = test_in   # we're doing autoencoding...

print np.shape(train_in)
learning_rate = 0.01
nhidden = 20
niterations = 2500
hidden_type='relu' #linear, relu, or logistic

net = mlp_autoenc.mlp(train_in,train_tgt,nhidden,outtype='linear',hidtype=hidden_type)
net.mlptrain(train_in, train_tgt, 0.001, niterations=101, momentum=0.5)
net.mlptrain(train_in, train_tgt, learning_rate, niterations)
outputs = net.mlpfwd(test_in)



pl.figure()
i=0
pair_num = 1
for r in range(5):
    pl.subplot(5,5,i+1)
    pl.imshow(test_in[pair_num].reshape(28,28),interpolation='nearest', cmap=pl.cm.gray, vmin=-1.,vmax=1.)
    pl.text(0,0,'input %d' %(pair_num) )
    pl.axis('off')
    i += 1
    pl.subplot(5,5,i+1)
    pl.imshow(outputs[pair_num].reshape(28,28),interpolation='nearest', cmap=pl.cm.gray, vmin=-1.,vmax=1.)
    pl.text(0,0,'recon %d' %(pair_num) )
    pl.axis('off')
    i += 2
    pair_num += 1

    pl.subplot(5,5,i+1)
    pl.imshow(test_in[pair_num].reshape(28,28),interpolation='nearest', cmap=pl.cm.gray, vmin=-1.,vmax=1.)
    pl.text(0,0,'input %d' %(pair_num) )
    pl.axis('off')
    i += 1
    pl.subplot(5,5,i+1)
    pl.imshow(outputs[pair_num].reshape(28,28),interpolation='nearest', cmap=pl.cm.gray, vmin=-1.,vmax=1.)
    pl.text(0,0,'recon %d' %(pair_num) )
    pl.axis('off')
    i += 1
    pair_num += 1

outfile = hidden_type + '_outputs.png'
pl.savefig(outfile)

pl.figure()
i=0
for c in range(5):
    for r in range(4):
        pl.subplot(4,5,i+1)
        pl.imshow(net.weights1[:,i].reshape(28,28),interpolation='nearest', cmap=pl.cm.gray, vmin=-1.,vmax=1.)
        pl.text(0,0,'hid %d' %(i) )
        pl.axis('off')
        i += 1
pl.draw()
outfile = hidden_type + '_weights.png'
pl.savefig(outfile)
#pl.savetxt('net.out',net.weights1)
