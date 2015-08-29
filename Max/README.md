# My RBM slash ORBM Implementations

### ORBM-Training.ipynb ###

In this document I explore training an ORBM architecture. That is, can I train two
RBM's given a composite/multi-source input.

### RBM-XOR.ipynb ###

In this notebook I explore how I can evaluate these models, both my RBM Implementation and the SKLearn variant.
This was the first time I could emperically evaluate the reconstructions of an RBM (being a 2 bit case)

I also examined the dreams of the RBM for the first time to get a sense of what the generative model looked like.
In this two bit case it is obvious that it should produce 1 bit on a time.

Finally I kept scaling up until I reached a 3 bit rbm. I wasn't comparing to the ORBM at this point I was simply
exploring what an RBM could learn.

### ORBM-Inference-XOR.ipynb ###

This notebook is where I explored the results of the ApproximatedSampler/FullSampler and ensured that the probabilities
of the various hidden configurations made sense given an input.

I also explored the reconstructions for the xor model using this approximated sampler, to ensure that the two extracted
reconstructions matched the expected outputs (xor). I also examined how they would cope with both the sources causing the
same point. Also this is just looking at two of the same model, acting indepednatly to create a composite image.

I also compared this to the traditional/vanilla RBM - it did not know what to do! Ha!

### ORBM-XOR-X-bits ###

In this notebook I extend ORBM-Inference-XOR to more bits. The results continue to work which is excellent.

I also examined the affect of updating a single layer at once in the ORBM hidden to visible (i.e. sampling) to see
if updating both versus one at a time had a significant effect. It didn't.

### ORBM-XOR-More_bits ###

In this notebook I further extend ORBM-Inference-XOR and ORBM-XOR-X-bits, in that I also try it with more than one bit
on at once, where bits on must be ajdancent to each other i.e. for a **3** bit pattern with **2** bits on, the possible
inputs are
```
[1,1,0] and [0,1,1] NOT [1,0,1].
```

This continued to work even for larger bits. However it was becoming clear that I needed equal if not more hidden units
to visible units to achieve a good model. I suspect this problems almost, convolutional aspect was the result of this.

### RBM-ORBM-Single-Models ###

In this notebook I make the jump from just a bit string to an entire image. Admittedly the difference is not that large,
because the RBM will be given the flattened or raveled image anyway - I digress.

In this notebook I ensured that the ORBM could separate, that is make correct reconstructions for input data of overlaid
sqaures. It could! Also evaluating the models became more difficult, as I had to instead calculate the approximate log
likelyhood of reconstructing the dataset given the datset as input. This was my performance measure to ensure my model was
**good**.

### RBM-ORBM-Single-Models-Non-Binary ###

This notebook extends RBM-ORBM-Single-Models to try having continous visible pixels. The results were not great. I struggled
to train a vanilla RBM to be able to learn about continous visible. My approach was creating images of squares of 0.5 instensity,
that is half, then when overlaid they would have an intensity of 1.

This was not the case. mostl likely because it's already a hard problem to learn let alone adding instensity into the mix.

### RBM-ORBM-Dual-Models.ipynb ###

In this notebook I explore how well the ORBM can make separate reconstructions of
multi-source images with two different (albiet similar) models. I create a sqaure and
rectangle model (RBMs), take an item from their respective training sets and composite
them. reconstructions are then generated from this composite input. I can then compare This
to the ground truth.

The ApproximatedSampler was used and did really well for these small models. There is also
some repeated code around verifying if a model/rbm is 'good'. This amounts to examining it's
dreams and ensuring they make sense.


### (Deprecated) MNIST-ORBM ###

My attempt at applying the ORBM to MNIST. Currently Deprecated.
