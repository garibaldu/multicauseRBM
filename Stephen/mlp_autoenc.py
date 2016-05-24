
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Yay! I'm changing it to have explicit bias weights, on hiddens and outputs. M.

# Stephen Marsland, 2008, 2014

import numpy as np

class mlp:
    """ A Multi-Layer Perceptron"""
    
    def __init__(self,inputs,targets,nhidden,outtype='linear',hidtype='logistic'):
        """ Constructor """
        # Set up network size
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden

        self.outtype = outtype
        self.hidtype = hidtype
    
        # Initialise network
        self.weights1 = (np.random.rand(self.nin,self.nhidden)-0.5)*2/np.sqrt(self.nin)
        self.hid_bias = (np.random.rand(1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
        self.vis_bias = (np.random.rand(1,self.nin)-0.5)*2/np.sqrt(self.nin)


    def mlptrain(self,inputs,targets,eta,niterations,momentum=0.9):
        """ Train the thing """
        # temporary space for the weight changes
        w1_update = np.zeros((np.shape(self.weights1)))
        w2_update = np.zeros((np.shape(self.weights1.T)))
        # temporary space for the bias changes
        hid_bias_update = np.zeros((np.shape(self.hid_bias)))
        vis_bias_update = np.zeros((np.shape(self.vis_bias)))
            
        for n in range(niterations):
            
            self.outputs = self.mlpfwd(inputs)
            
            error = 0.5*np.sum((self.outputs-targets)**2)

            if (np.mod(n,100)==0):  print "Iteration: ",n, " Error: ",error    

            # Different types of output neurons
            if self.outtype == 'linear':
            	deltao = (self.outputs-targets)/self.ndata  # why the division??
            elif self.outtype == 'logistic':
            	deltao = (self.outputs-targets)*self.outputs*(1.0-self.outputs)
            elif self.outtype == 'softmax':
                deltao = (self.outputs-targets)*(self.outputs*(-self.outputs)+self.outputs)/self.ndata
            else:
                print "bogus outtype"

                
            if self.hidtype == 'linear':
                deltah = np.dot(deltao,np.transpose(self.weights1.T))
            elif self.hidtype == 'relu':
                deltah = np.maximum(0,np.sign(self.hidden)) * (np.dot(deltao,np.transpose(self.weights1.T)))
            elif self.hidtype == 'logistic':
                deltah = self.hidden*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights1.T)))
            else:
                print "bogus hidtype"
                   
            w1_update = eta*(np.dot(np.transpose(inputs),deltah)) + momentum*w1_update
            w2_update = eta*(np.dot(np.transpose(self.hidden),deltao)) + momentum*w2_update
            update = 0.5*(w1_update + w2_update.T)
            self.weights1 -= update   #+ 0.2*np.sign(self.weights1)

            hid_bias_update = eta*(np.sum(deltah,0)) + momentum*hid_bias_update
            self.hid_bias -= hid_bias_update



    def mlpfwd(self,inputs):
        """ Run the network forward """

        hidden_psi = np.dot(inputs,self.weights1) + self.hid_bias
        # Different types of hidden neurons
        if self.hidtype == 'linear':
            self.hidden = hidden_psi
        elif self.hidtype == 'logistic':
            self.hidden = 1.0/(1.0+np.exp(-hidden_psi))
        elif self.hidtype == 'relu':
            self.hidden = np.maximum(0.0, hidden_psi)
        else:
            print "bogus hidtype"

        linear_out = np.dot(self.hidden,self.weights1.T) + self.vis_bias

        # Different types of output neurons
        if self.outtype == 'linear':
            outputs = linear_out
        elif self.outtype == 'logistic':
            outputs = 1.0/(1.0+np.exp(-linear_out))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(linear_out),axis=1)*np.ones((1,np.shape(linear_out)[0]))
            outputs = np.transpose(np.transpose(np.exp(linear_out))/normalisers)
        else:
            print "bogus outtype"

        return outputs
