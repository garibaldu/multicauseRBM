import rbm
import os, sys, optparse

if __name__ == '__main__':

    parser = optparse.OptionParser(usage="usage %prog [options]")
    parser.add_option("-f", "--name", type = "str", dest = "name", help = "name of the RBM")
    opts, args = parser.parse_args()
    EXIT = False
    if (opts.name is None):
        print("ERROR: you must supply a name for the trained RBM\n")
        EXIT = True
    if EXIT: 
        parser.print_help()
        sys.exit(-1)


    digits = [0,1,2,3,4,5,6,7,8,9]
    inpats = rbm.load_mnist_digits(digits, 100)
    rbm.show_example_images(inpats)


    
    if os.path.isfile('./saved_nets/' + opts.name + '.npz'):
        r = rbm.RBM(opts.name) # attempt to read an existing RBM in.
    else:
        sys.exit("No such RBM! (%s)" % (opts.name))

    ####################################################################################
   
    r.DROPOUT = False
    r.make_weights_figure()
    r.make_dynamics_figure(inpats)
    r.rename(r.name + '_scrambled')
    r.make_dynamics_figure(inpats, SCRAMBLE=True)

