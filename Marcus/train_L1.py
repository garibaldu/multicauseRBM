import rbm

if __name__ == '__main__':

    digits = [0,1,2,3,4,5,6,7,8,9]
    dataset_size = 500
    inpats = rbm.load_mnist_digits(digits, dataset_size)
    rbm.show_example_images(inpats)

    # parameters of net and learning
    num_visibles = inpats.shape[1]
    num_hiddens = 200
    iterations = 1000
    learn_rate = 0.01
    momentum = 0.9
    L1_penalty = 0.0001
    minibatch_size = 32

    r = rbm.RBM('works', num_hiddens, num_visibles, DROPOUT=True)

    r.train(inpats, iterations, learn_rate, momentum, L1_penalty, minibatch_size)
    r.save_as_pickle()

    # optionally....
    #r.make_weights_figure()
    #r.make_dynamics_figure(inpats)
    
