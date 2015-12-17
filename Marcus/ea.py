import numpy as np
import lbm
import os, sys, optparse
import numpy.random as rng
from scipy.special import expit as sigmoid
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = optparse.OptionParser(usage="usage %prog [options]")
    parser.add_option("-f", "--nameA", type = "str", dest = "nameA", help = "name for LBM A")
    parser.add_option("-g", "--nameB", type = "str", dest = "nameB", help = "name for LBM B")
    parser.add_option("-d", "--nitems", type = "int", dest = "nitems", default = 70, help = "number of training examplars PER DIGIT")
    opts, args = parser.parse_args()
    EXIT = False
    if (opts.nameA is None) or (opts.nameB is None):
        print ("ERROR: you must supply names for the trained LBM\n")
        EXIT = True
    if EXIT: 
        parser.print_help()
        sys.exit(-1)


    digits = [6,9]
    inpats = lbm.load_mnist_digits(digits, opts.nitems)
    
    if os.path.isfile('./saved_nets/' + opts.nameA + '.npz'):
        A = lbm.RBM(opts.nameA) # attempt to read an existing RBM in.
    else:
        sys.exit('no file for A')

    if os.path.isfile('./saved_nets/' + opts.nameB + '.npz'):
        B = lbm.RBM(opts.nameB) # attempt to read an existing RBM in.
    else:
        sys.exit('no file for B')

    A_inpats = inpats

    num_pats = inpats.shape[0]
    rand_order = rng.permutation(np.arange(num_pats))
    B_inpats = inpats[rand_order]


    v = A_inpats + B_inpats

    A.DROPOUT = False
    B.DROPOUT = False

    # This would be cheating...
    #hA = A.pushup(A_inpats)
    #hB = B.pushup(B_inpats)

    # initialise the two hidden layers (not cheating anymore)
    hA = lbm.random_hiddens_for_rbm(A, num_pats)
    hB = lbm.random_hiddens_for_rbm(B, num_pats)
    hA = A.pushup(v)
    hB = A.pushup(v)
    


    plt.subplot(121)
    plt.imshow(hA)
    plt.subplot(122)
    plt.imshow(hB)
    plt.savefig('thing.png')

    print ('---------------')

    for t in range(500):
        phiB = B.pushdown(hB)
        psiA = A.explainaway(phiB, hA, v)
        hA = 1*(sigmoid(psiA) > rng.random(size=psiA.shape))

        phiA = A.pushdown(hA)
        psiB = B.explainaway(phiA, hB, v)
        hB = 1*(sigmoid(psiB) > rng.random(size=psiB.shape))

    lbm.show_example_images(v, 'unexplained.png')
    lbm.show_example_images(phiA, 'explainedA.png')
    lbm.show_example_images(phiB, 'explainedB.png')
