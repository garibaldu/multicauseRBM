from orbiumutils import *
import rbm 

if __name__ == '__main__':

    digits = [0,1,2,3,4]
    dataset_size = 500
    inpats = load_mnist_digits(digits, dataset_size)
    show_example_images(inpats)

    num_hiddens = 100
    iterations = 100
    learn_rate, momentum, L1_penalty, minibatch_size = 0.01, 0.9, 0.005, 50

    (num_hiddens, num_visibles) = inpats.shape
    rbmA = rbm.RBM('RBM9', num_hiddens, num_visibles)

    rbmA.train(inpats, 50, 0.001, 0.5, L1_penalty, minibatch_size) # early training
    rbmA.train(inpats, iterations, learn_rate, momentum, L1_penalty, minibatch_size)

    rbmA.make_weights_figure()
    rbmA.make_dynamics_figure(inpats)
    rbmA.save_as_pickle()
