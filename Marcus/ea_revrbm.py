import numpy as np
import revrbm
import os, sys, optparse
import numpy.random as rng
from scipy.special import expit as sigmoid
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = optparse.OptionParser(usage="usage %prog [options]")
    parser.add_option("-f", "--nameA", type = "str", dest = "nameA", help = "name for REVRBM A")
    parser.add_option("-g", "--nameB", type = "str", dest = "nameB", help = "name for REVRBM B")
    parser.add_option("-d", "--nitems", type = "int", dest = "nitems", default = 70, help = "number of training examplars PER DIGIT")
    parser.add_option("-D", type = "str", dest = "digitsAsStr", help = "digits to use, as comma-separated list (e.g. -D 4,5,6)", default = '4')
    opts, args = parser.parse_args()
    EXIT = False
    if (opts.nameA is None) or (opts.nameB is None):
        print ("ERROR: you must supply names for the trained REVRBM\n")
        EXIT = True
    if EXIT: 
        parser.print_help()
        sys.exit(-1)


    
    if os.path.isfile('./saved_nets/' + opts.nameA + '.npz'):
        A = revrbm.RBM(opts.nameA) # attempt to read an existing RBM in.
    else:
        sys.exit('no file for A')

    if os.path.isfile('./saved_nets/' + opts.nameB + '.npz'):
        B = revrbm.RBM(opts.nameB) # attempt to read an existing RBM in.
    else:
        sys.exit('no file for B')

    digits = [int(s) for s in opts.digitsAsStr.split(',')]
    A_inpats = revrbm.load_mnist_digits(digits, opts.nitems)
    B_inpats = revrbm.generate_smooth_bkgd(opts.nitems)
    num_pats = A_inpats.shape[0]
    
    v = A_inpats + B_inpats

    A.DROPOUT = False
    B.DROPOUT = False

    # This would be cheating...
    #hA = A.pushup(A_inpats)
    #hB = B.pushup(B_inpats)

    # initialise the two hidden layers (not cheating anymore)
    #hA = revrbm.random_hiddens_for_rbm(A, num_pats)
    #hB = revrbm.random_hiddens_for_rbm(B, num_pats)
    hA = A.pushup(v)
    hB = B.pushup(v)
    

    print ('---------------')

    
    if (rng.random() > 0.5): # HACK - don't always start with the same net..
        phiA = A.pushdown(hA)
        psiB = B.explainaway(phiA, hB, v)
        hB = 1*(sigmoid(psiB) > rng.random(size=psiB.shape))
        

    for t in range(25):
        phiB = B.pushdown(hB)
        psiA = A.explainaway(phiB, hA, v)
        hA = 1*(sigmoid(psiA) > rng.random(size=psiA.shape))

        phiA = A.pushdown(hA)
        psiB = B.explainaway(phiA, hB, v)
        hB = 1*(sigmoid(psiB) > rng.random(size=psiB.shape))

        
    rows = 8
    cols = 5
    i=0
    plt.clf()
    for r in range(rows):
        j = rng.randint(0,len(v)) # pick a random example to show
        plt.subplot(rows,cols,i+1)
        plt.imshow(v[j].reshape(28,28), cmap='Greys', interpolation='nearest', vmin=-1.0, vmax=1.0)
        plt.axis('off')
        if r==0 : plt.text(0,0,'visible')
        i += 1

        plt.subplot(rows,cols,i+1)
        plt.imshow((phiA[j]+phiB[j]).reshape(28,28), cmap='Greys', interpolation='nearest', vmin=-1.0, vmax=1.0)
        plt.axis('off')
        if r==0 : plt.text(0,0,'A + B')
        i += 1

        plt.subplot(rows,cols,i+1)
        plt.imshow(phiA[j].reshape(28,28), cmap='Greys', interpolation='nearest', vmin=-1.0, vmax=1.0)
        plt.axis('off')
        if r==0 : plt.text(0,0,'A construct', color='blue')
        i += 1

        plt.subplot(rows,cols,i+1)
        plt.imshow(phiB[j].reshape(28,28), cmap='Greys', interpolation='nearest', vmin=-1.0, vmax=1.0)
        plt.axis('off')
        if r==0 : plt.text(0,0,'B construct', color='blue')
        i += 1

        hA = A.pushup(v[j])
        naive = A.pushdown(hA)
        plt.subplot(rows,cols,i+1)
        plt.imshow(naive.reshape(28,28), cmap='Greys', interpolation='nearest', vmin=-1.0, vmax=1.0)
        if r==0 : plt.text(0,0,'cf. no EA', color='blue')
        plt.axis('off')
        i += 1

    filename = 'eatest'
    plt.savefig(filename)
    print('Saved figure named %s' % (filename))
    
    revrbm.show_example_images(v, 'unexplained.png')
    revrbm.show_example_images(phiA, 'explainedA.png')
    revrbm.show_example_images(phiB, 'explainedB.png')
