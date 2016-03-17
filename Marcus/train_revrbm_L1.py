import numpy as np
import lbm
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
    opts, args = parser.parse_args()
    EXIT = False
    if (opts.name is None):
        print ("ERROR: you must supply a name for the trained RBM\n")
        EXIT = True
    if EXIT: 
        parser.print_help()
        sys.exit(-1)


    digits = [6]  # the digits we'll train on.
    print ("opts.nitems = ", opts.nitems)
    inpats = np.zeros((opts.nitems * 2, 28*28), dtype=float)
    inpats[:opts.nitems] = lbm.load_mnist_digits(digits, opts.nitems) 
    inpats[opts.nitems:] = lbm.generate_smooth_bkgd(opts.nitems)
    #sys.exit(0)
    lbm.show_example_images(inpats, filename='examples.png')
    #sys.exit(0)
        
    if os.path.isfile('./saved_nets/' + opts.name + '.npz'):
        r = lbm.RBM(opts.name) # attempt to read an existing RBM in.
        if (opts.newname is not None): # rename the net if a new name is provided
            r.rename(opts.newname)
    else:
        r = lbm.RBM(opts.name, opts.num_hids, num_vis=inpats.shape[1], DROPOUT=True)


    r.train(inpats, opts.iterations, opts.rate, opts.momentum, opts.L1_penalty, minibatch_size=50)
    r.save_as_pickle()

