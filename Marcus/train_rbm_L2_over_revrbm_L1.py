import rbm, revrbm
import os, sys, optparse


if __name__ == '__main__':

    parser = optparse.OptionParser(usage="usage %prog [options]")
    parser.add_option("-f", "--L1name", type = "str", dest = "L1name", help = "name for the (existing) Layer 1 revRBM")
    parser.add_option("-g", "--L2name", type = "str", dest = "L2name", help = "name for the (new) Layer 2 RBM")
    parser.add_option("-n", "--nhids", type = "int", dest = "num_hids", default = 50, help = "number of hidden units in layer 2")
    parser.add_option("-t", "--niters", type = "int", dest = "iterations", default = 500, help = "number of iterations of training")
    parser.add_option("-d", "--nitems", type = "int", dest = "nitems", default = 500, help = "number of training examplars PER DIGIT")
    parser.add_option("-r", "--rate", type = "float", dest = "rate", help = "learning rate", default = 0.005)
    parser.add_option("-m", "--mom", type = "float", dest = "momentum", help = "momentum", default = 0.9)
    parser.add_option("-D", type = "str", dest = "digitsAsStr", help = "digits to train on, as comma-separated list (e.g. -D 4,5,6)", default = '1')
    #parser.add_option("-d", "--data", type = "str", dest = "indata", help = "name of a dataset")

    opts, args = parser.parse_args()
    EXIT = False
    if (opts.L1name is None) or (opts.L2name is None):
        print ("ERROR: you must supply a name for both of the RBMs\n")
        EXIT = True
    if EXIT: 
        parser.print_help()
        sys.exit(-1)

    
    digits = [int(s) for s in opts.digitsAsStr.split(',')]
    print(digits)
    L1_inpats = revrbm.load_mnist_digits(digits, dataset_size=opts.nitems)

    # read in the layer 1 RBM
    if os.path.isfile('./saved_nets/' + opts.L1name + '.npz'):
        L1 = revrbm.RBM(opts.L1name) # attempt to read an existing RBM in.
    else:
        sys.exit("No such L1 revRBM! (%s)" % (opts.L1name))

    ####################################################################################
 
    # create the layer 2 RBM
    L2 = rbm.RBM(opts.L2name, num_hid=opts.num_hids, num_vis=L1.num_hid)
    L2_inpats = L1.pushup(L1_inpats)

    L2.train(L2_inpats, opts.iterations, opts.rate, opts.momentum, L1_penalty=0.0001, minibatch_size=32)
    L2.save_as_pickle()



    ########################################################
    # DISPLAY STUFF, COPIED FROM show_rbm_L2_over_L1.py
    ########################################################
    L1.DROPOUT = False
    L2.DROPOUT = False
    print("make_2layer_dynamics_figure..."),
    revrbm.make_2layer_dynamics_figure(L1, L2)
    print("done")
    # We start by generating random patterns from the top layer RBM.
    # The points where DROPOUT is/isn't used appears to be critical to success :-(
    L2.DROPOUT = True
    top_pats = rbm.random_hiddens_for_rbm(L2, 100)
    Steps = 10
    for t in range(Steps):
        mid_pats = L2.pushdown(top_pats)
        top_pats = L2.pushup(mid_pats)
        if (t >= Steps - 3): # a couple of "smoother" iterations at the end. Bah, humbug.
            L2.DROPOUT = False

    L1.DROPOUT = False # this seems to matter, which is a little annoying.
    vis_dreams = L1.pushdown(mid_pats)

    print("show_example_images..."),
    revrbm.show_example_images(vis_dreams, filename='%s_combo_dreams.png'%(L2.name))
    print("done")

