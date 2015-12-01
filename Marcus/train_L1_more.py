import rbm 

if __name__ == '__main__':

    inpats = load_mnist_digits([9], dataset_size=500)

    iterations = 100
    learning_rate, momentum, L1_penalty, minibatch_size = 0.1, 0.95, 0.005, 50


    r = rbm.RBM('RBM9') # should attempt to read an existing RBM in.
    #r.rename('RBM6')

    r.train(inpats, iterations, learning_rate, momentum, L1_penalty, minibatch_size)

    r.make_weights_figure()
    r.make_dynamics_figure(inpats)
    r.save_as_pickle()

