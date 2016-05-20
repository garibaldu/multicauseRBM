import pylab as pl
import numpy as np
import mlp
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
    with open("../Notebooks/MachineLearning/Partitioned RBM/multicauseRBM/Max/datasets/{}.npy".format(digit),'rb') as f:
        return np.load(f)[:dataset_size]
    
def flatten_dataset(images):
    smushed = images.copy()
    return smushed.reshape((smushed.shape[0], -1))

pl.ion()
# Read the dataset in (code from sheet)
train_in = load_mnist_digits([2,5], 200)
test_in = load_mnist_digits([2,5], 200)
#f = gzip.open('/Users/srmarsla/Teaching/372/Labs/Code/Pcn/mnist.pkl.gz','rb')
#tset, vset, teset = cPickle.load(f)
#f.close()

print np.shape(train_in)

train_tgt = train_in
test_tgt = test_in

for nhidden in [10]:  
    net = mlp.mlp(train_in,train_tgt,nhidden,outtype='linear')
    net.mlptrain(train_in,train_tgt,0.2,1000)
outputs = net.mlpfwd(test_in)
print np.sum((outputs-test_in)**2)

pl.figure()
i=0
for c in range(5):
	for r in range(5):
		pl.subplot(5,5,i+1)
		pl.imshow(outputs[i].reshape(28,28),interpolation='nearest', cmap=pl.cm.gray, vmin=-1.,vmax=1.)
		pl.axis('off')
		i += 1

pl.figure()
i=0
for c in range(5):
	for r in range(4):
		pl.subplot(5,4,i+1)
		pl.imshow(net.weights1[:,i].reshape(28,28),interpolation='nearest', cmap=pl.cm.gray, vmin=-1.,vmax=1.)
		pl.axis('off')
		i += 1
pl.savefig('out.png')
pl.savetxt('net.out',net.weights1)

pl.show()
