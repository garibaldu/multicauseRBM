import rbm

if __name__ == '__main__':

    L1 = rbm.RBM('works') # should attempt to read an existing RBM in.
    L1.DROPOUT = False

    L1_inpats = rbm.load_mnist_digits([0,1,2,3,4,5,6,7,8,9], dataset_size=50)

    L1.make_weights_figure()
    L1.make_dynamics_figure(L1_inpats)
    L1.rename('works_scrambled')
    L1.make_dynamics_figure(L1_inpats, SCRAMBLE=True)


    L2 = rbm.RBM('works_L2')
    L2.DROPOUT = False
    L2_inpats = L1.pushup(L1_inpats)


    # generate random patterns from the top layer RBM.
    top_pats = rbm.random_hiddens_for_rbm(L2, 100)
    for t in range(2):
        mid_pats = L2.pushdown(top_pats)
        top_pats = L2.pushup(mid_pats)
    vis = L1.pushdown(mid_pats)
    L2.show_patterns(vis)

    rbm.make_2layer_dynamics_figure(L1, L2)


