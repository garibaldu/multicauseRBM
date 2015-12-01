from orbiumutils import *
import rbm 

if __name__ == '__main__':

    inpats = load_mnist_digits([0,1,2,3,4,5,6,7,8,9], dataset_size=50)

    r = rbm.RBM('works') # should attempt to read an existing RBM in.

    r.make_weights_figure()
    r.make_dynamics_figure(inpats)
    r.rename('works_scrambled')
    r.make_dynamics_figure(inpats, SCRAMBLE=True)

