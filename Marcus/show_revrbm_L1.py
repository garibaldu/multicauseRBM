import revrbm
import os, sys, optparse

if __name__ == '__main__':

    parser = optparse.OptionParser(usage="usage %prog [options]")
    parser.add_option("-f", "--name", type = "str", dest = "name", help = "name of the revRBM")
    parser.add_option("-D", type = "str", dest = "digitsAsStr", help = "digits to show, as comma-separated list (e.g. -D 4,5,6)", default = '4')
    opts, args = parser.parse_args()
    EXIT = False
    if (opts.name is None):
        print("ERROR: you must supply a name for the trained RBM\n")
        EXIT = True
    if EXIT: 
        parser.print_help()
        sys.exit(-1)


    digits = [int(s) for s in opts.digitsAsStr.split(',')]
    inpats = revrbm.load_mnist_digits(digits, 100)
    #rbm.show_example_images(inpats)


    
    if os.path.isfile('./saved_nets/' + opts.name + '.npz'):
        r = revrbm.RBM(opts.name) # attempt to read an existing RBM in.
    else:
        sys.exit("No such revRBM! (%s)" % (opts.name))

    #######################################
   
    r.DROPOUT = False
    r.make_weights_figure()
    r.make_dynamics_figure(inpats)
    #r.rename(r.name + '_scrambled')
    #xr.make_dynamics_figure(inpats, SCRAMBLE=True)

    hid_pats = revrbm.random_hiddens_for_rbm(r, 100)
    Steps = 10
    for t in range(Steps):
        print ('dream step %d' %(t))
        vis_dreams = r.pushdown(hid_pats)
        hid_pats = r.pushup(vis_dreams)

    revrbm.show_example_images(vis_dreams, filename='%s_dreams.png'%(r.name))

