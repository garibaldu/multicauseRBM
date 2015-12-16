import numpy as np
import lbm
import os, sys, optparse
import numpy.random as rng
from scipy.special import expit as sigmoid
np.set_printoptions(precision=2)

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


    digits = [2,4]
    inpats = lbm.load_mnist_digits(digits, opts.nitems)
    
    if os.path.isfile('./saved_nets/' + opts.nameA + '.npz'):
        A = lbm.RBM(opts.nameA) # attempt to read an existing RBM in.
        B = lbm.RBM(opts.nameB) # attempt to read an existing RBM in.
    else:
        sys.exit('you fucked up')

    A_inpats = inpats

    num_pats = inpats.shape[0]
    rand_order = rng.permutation(np.arange(num_pats))
    B_inpats = inpats[rand_order]


    #lbm.show_example_images(A_inpats, 'A_inpats.png')
    #lbm.show_example_images(B_inpats, 'B_inpats.png')
    v = A_inpats + B_inpats
    #lbm.show_example_images(v, 'combos.png')

    # initialise the two hidden layers (cheating a bit)
    hA = A.pushup(A_inpats)
    hB = B.pushup(B_inpats)
    print (hA)
    print (hB)

    print ('---------------')
    
    phiB = B.pushdown(hB)
    psiA = A.explainaway(phiB, hA, v)
    hA = 1*(sigmoid(psiA) > rng.random(size=psiA.shape))
    print (hA)

    phiA = A.pushdown(hA)
    psiB = B.explainaway(phiA, hB, v)
    hB = 1*(sigmoid(psiB) > rng.random(size=psiB.shape))
    print (hB)

