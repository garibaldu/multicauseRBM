import rbm
import os, sys, optparse

if __name__ == '__main__':

    parser = optparse.OptionParser(usage="usage %prog [options]")
    parser.add_option("-f", "--L1name", type = "str", dest = "L1name", help = "name for the Layer 1 RBM")
    parser.add_option("-g", "--L2name", type = "str", dest = "L2name", help = "name for the Layer 2 RBM")
    opts, args = parser.parse_args()
    EXIT = False
    if (opts.L1name is None) or (opts.L2name is None):
        print ("ERROR: you must supply a name for both of the trained RBMs\n")
        EXIT = True
    if EXIT: 
        parser.print_help()
        sys.exit(-1)


    digits = [0,1,2,3,4,5,6,7,8,9]
    L1_inpats = rbm.load_mnist_digits(digits, dataset_size=100)

    # read in the layer 1 RBM
    if os.path.isfile('./saved_nets/' + opts.L1name + '.npz'):
        L1 = rbm.RBM(opts.L1name) # attempt to read an existing RBM in.
    else:
        sys.exit("No such L1 RBM! (%s)" % (opts.L1name))

    # read in the layer 2 RBM
    if os.path.isfile('./saved_nets/' + opts.L2name + '.npz'):
        L2 = rbm.RBM(opts.L2name) # attempt to read an existing RBM in.
    else:
        sys.exit("No such L2 RBM! (%s)" % (opts.L2name))

    ####################################################################################
   

    L1.DROPOUT = False
    L2.DROPOUT = False
    rbm.make_2layer_dynamics_figure(L1, L2)

   
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

    rbm.show_example_images(vis_dreams, filename='%s_combo_dreams.png'%(L2.name))

