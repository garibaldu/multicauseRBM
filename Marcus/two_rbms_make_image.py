import rbm
import os, sys, optparse

if __name__ == '__main__':

    parser = optparse.OptionParser(usage="usage %prog [options]")
    parser.add_option("-f", "--name", type = "str", dest = "name", help = "name of the pre-trained RBM")
    opts, args = parser.parse_args()
    EXIT = False
    if (opts.name is None):
        print ("ERROR: you must supply a name of a pre-trained RBM\n")
        EXIT = True
    if EXIT: 
        parser.print_help()
        sys.exit(-1)


    digits = [0,1,2,3,4,5,6,7,8,9]
    inpats = rbm.load_mnist_digits(digits, 100)
    rbm.show_example_images(inpats)

    if os.path.isfile('./saved_nets/' + opts.name + '.npz'):
        A = rbm.RBM(opts.name) # attempt to read an existing RBM in.
        B = rbm.RBM(opts.name) # attempt to read an existing RBM in.
    else:
        sys.exit("No such RBM! (%s)" % (opts.name))

    #############################################
    A.rename('orbmA')
    B.rename('orbmB')   
    A.DROPOUT = True
    B.DROPOUT = True

    # We start by generating random hidden patterns from the two RBMs.
    Avis = inpats[0:100]
    Bvis = inpats[100:200]
    Steps = 2
    for t in range(Steps):
        Ahid = A.pushup(Avis)
        Avis = A.pushdown(Ahid)
        Bhid = B.pushup(Bvis)
        Bvis = B.pushdown(Bhid)
        if (t >= Steps - 3): # a couple of "smoother" iterations at the end. Bah, humbug.
            A.DROPOUT = False
            B.DROPOUT = False
    A.show_patterns(Avis)
    B.show_patterns(Bvis)
    
    #combo = rbm.sigmoid(rbm.inverse_sigmoid(Avis) + rbm.inverse_sigmoid(Bvis))
    combo = (Avis + Bvis)/2.0
    rbm.show_example_images(combo, filename='combo_examples.png')
    #rbm.show_example_images(vis_dreams, filename='%s_combo_dreams.png'%(L2.name))

