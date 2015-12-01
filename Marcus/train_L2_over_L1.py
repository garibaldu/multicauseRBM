import rbm


if __name__ == '__main__':

    L1 = rbm.RBM('works') # should attempt to read an existing RBM in.
    L1.DROPOUT = False
    L2 = rbm.RBM('works_L2', num_hid=50, num_vis=L1.num_hid) # 2nd layer
    
    L1_inpats = rbm.load_mnist_digits([0,1,2,3,4,5,6,7,8,9], dataset_size=500)
    L2_inpats = L1.pushup(L1_inpats)

    # parameters of net and learning
    num_visibles = L2_inpats.shape[1]
    num_hiddens = 100
    iterations = 200
    learn_rate = 0.001
    momentum = 0.5
    L1_penalty = 0.000001
    minibatch_size = 32


    L2.train(L2_inpats, iterations, learn_rate, momentum, L1_penalty, minibatch_size)
    L2.save_as_pickle()

