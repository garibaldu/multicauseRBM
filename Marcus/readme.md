The defaults should work, e.g. if you do this....

python train_L1.py -f testLay1

   * reads in MNIST data (500 of each digit so training set of 5000 all up).
   * trains a default RBM with 200 hidden units, using default learning algorithm settings (about 10 mins at ECS)
   * saves the trained RBM in ./saved_nets

python train_L1.py -f testLay1

   * running this again should read in the existing network and just train it further
   * you should be able to see this in the initial value of the RMS reconstruction error

python show_L1.py -f testLay1

   * the Gibbs chains started in "scrambled" initial conditions will be pretty shitty
   * I think that's due to using CD1, and perhaps not training long enough

python train_L2_over_L1.py -f testLay1 -g testLay2

   * uses the existing testLay1 RBM to generate training data for a new RBM, testLay2, which gets trained and saved

python show_L1L2.py -f testLay1 -g testLay2

   * shows the outputs (dreams) that result from running the two-layered architecture in generative mode.
