from orbiumutils import *
import rbm 


if __name__ == '__main__':

    iterations = 250
    learning_rate, momentum, L1_penalty, minibatch_size = 0.01, 0.25, 0.0, 50


    L1 = rbm.RBM('RBM9') # should attempt to read an existing RBM in.
    digits = [0,1,2,3,4]
    L1_inpats = load_mnist_digits(digits, dataset_size=100)

    L2 = rbm.RBM('lay2_RBM9', num_hid=100, num_vis=L1.num_hid) # a new layer 2.
    L2_inpats = L1.pushup(L1_inpats)

    L2.train(L2_inpats, iterations, learning_rate, momentum, L1_penalty, minibatch_size)
    L2.save_as_pickle()

    # generate random patterns from the top layer RBM.
    top_pats = rbm.random_hiddens_for_rbm(L2, 100)
    for t in range(2):
        mid_pats = L2.pushdown(top_pats)
        top_pats = L2.pushup(mid_pats)
    vis_dreams = L1.pushdown(mid_pats)
    print(vis_dreams.shape)
    L2.show_patterns(vis_dreams)

    # Now I'd like to try to make some fantasies from the combined network.
