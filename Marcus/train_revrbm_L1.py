import numpy as np
import revrbm
import os, sys, optparse

if __name__ == '__main__':

    parser = optparse.OptionParser(usage="usage %prog [options]")
    parser.add_option("-f", "--name", type = "str", dest = "name", help = "name for RBM")
    parser.add_option("-t", "--niters", type = "int", dest = "iterations", default = 100, help = "number of iterations of training")
    parser.add_option("-d", "--nitems", type = "int", dest = "nitems", default = 500, help = "number of training examplars PER DIGIT")
    parser.add_option("-n", "--nhids", type = "int", dest = "num_hids", default = 200, help = "number of hidden units (ignored unless new net)")
    parser.add_option("-r", "--rate", type = "float", dest = "rate", help = "learning rate", default = 0.001)
    parser.add_option("-m", "--mom", type = "float", dest = "momentum", help = "momentum", default = 0.9)
    parser.add_option("-p", "--penalty", type = "float", dest = "L1_penalty", help = "L1 penalty", default = 0.0001)
    parser.add_option("-F", "--newname", type = "str", dest = "newname", help = "name for trained RBM  (defaults to existing name, ignored if new RBM)")
    parser.add_option("-D", type = "str", dest = "digitsAsStr", help = "digits to train on, as comma-separated list (e.g. -D 4,5,6)", default = '4')
    opts, args = parser.parse_args()
    EXIT = False
    if (opts.name is None):
        print ("ERROR: you must supply a name for the trained RBM\n")
        EXIT = True
    if EXIT: 
        parser.print_help()
        sys.exit(-1)


    digits = [int(s) for s in opts.digitsAsStr.split(',')]
    inpats = revrbm.load_mnist_digits(digits, opts.nitems) 
    """
    tmpN = opts.nitems * len(digits)
    print ("opts.nitems = ", opts.nitems)
    inpats = np.zeros((tmpN * 2, 28*28), dtype=float)
    inpats[:tmpN] = revrbm.load_mnist_digits(digits, opts.nitems) 
    inpats[tmpN:] = revrbm.generate_smooth_bkgd(tmpN)
    """
    revrbm.show_example_images(inpats, filename='examples.png')
    #sys.exit(0)
        
    if os.path.isfile('./saved_nets/' + opts.name + '.npz'):
        r = revrbm.RBM(opts.name) # attempt to read an existing RBM in.
        if (opts.newname is not None): # rename the net if a new name is provided
            r.rename(opts.newname)
    else:
        r = revrbm.RBM(opts.name, opts.num_hids, num_vis=inpats.shape[1], DROPOUT=True)


    r.train(inpats, opts.iterations, opts.rate, opts.momentum, opts.L1_penalty, minibatch_size=50)
    r.save_as_pickle()

