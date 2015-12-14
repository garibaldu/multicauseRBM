import rbm
import os, sys, optparse

if __name__ == '__main__':

    parser = optparse.OptionParser(usage="usage %prog [options]")
    parser.add_option("-f", "--name", type = "str", dest = "name", help = "name for RBM")
    parser.add_option("-t", "--niters", type = "int", dest = "iterations", default = 100, help = "number of iterations of training")
    parser.add_option("-d", "--nitems", type = "int", dest = "nitems", default = 500, help = "number of training examplars PER DIGIT")
    parser.add_option("-n", "--nhids", type = "int", dest = "num_hids", default = 200, help = "number of hidden units (ignored unless new net)")
    parser.add_option("-r", "--rate", type = "float", dest = "rate", help = "learning rate", default = 0.005)
    parser.add_option("-m", "--mom", type = "float", dest = "momentum", help = "momentum", default = 0.9)
    parser.add_option("-F", "--newname", type = "str", dest = "newname", help = "name for trained RBM  (defaults to existing name, ignored if new RBM)")
    opts, args = parser.parse_args()
    EXIT = False
    if (opts.name is None):
        print ("ERROR: you must supply a name for the trained RBM\n")
        EXIT = True
    if EXIT: 
        parser.print_help()
        sys.exit(-1)


    digits = [0,1,2,3,4,5,6,7,8,9]
    inpats = rbm.load_mnist_digits(digits, opts.nitems)
    
    if os.path.isfile('./saved_nets/' + opts.name + '.npz'):
        r = rbm.RBM(opts.name) # attempt to read an existing RBM in.
        if (opts.newname is not None): # rename the net if a new name is provided
            r.rename(opts.newname)
    else:
        r = rbm.RBM(opts.name, opts.num_hids, num_vis=inpats.shape[1], DROPOUT=True)

    r.train(inpats, 20, rate=0.01, momentum=0.2, L1_penalty=0.0, minibatch_size=32)
    r.train(inpats, opts.iterations, opts.rate, opts.momentum, L1_penalty=0.00001, minibatch_size=50)
    r.save_as_pickle()

