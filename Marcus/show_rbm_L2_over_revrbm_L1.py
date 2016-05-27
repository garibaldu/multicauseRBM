import rbm, revrbm
import os, sys, optparse

if __name__ == '__main__':

    parser = optparse.OptionParser(usage="usage %prog [options]")
    parser.add_option("-f", "--L1name", type = "str", dest = "L1name", help = "name for the Layer 1 revRBM")
    parser.add_option("-g", "--L2name", type = "str", dest = "L2name", help = "name for the Layer 2 RBM")
    parser.add_option("-D", type = "str", dest = "digitsAsStr", help = "digits to use, as comma-separated list (e.g. -D 4,5,6)", default = '4')
    opts, args = parser.parse_args()
    EXIT = False
    if (opts.L1name is None) or (opts.L2name is None):
        print ("ERROR: you must supply a name for both of the trained RBMs\n")
        EXIT = True
    if EXIT: 
        parser.print_help()
        sys.exit(-1)


    digits = [int(s) for s in opts.digitsAsStr.split(',')]
    L1_inpats = revrbm.load_mnist_digits(digits, dataset_size=100)

    # read in the layer 1 revRBM
    if os.path.isfile('./saved_nets/' + opts.L1name + '.npz'):
        L1 = revrbm.RBM(opts.L1name) # attempt to read an existing RBM in.
    else:
        sys.exit("No such L1 revRBM! (%s)" % (opts.L1name))

    # read in the layer 2 RBM
    if os.path.isfile('./saved_nets/' + opts.L2name + '.npz'):
        L2 = rbm.RBM(opts.L2name) # attempt to read an existing RBM in.
    else:
        sys.exit("No such L2 RBM! (%s)" % (opts.L2name))

