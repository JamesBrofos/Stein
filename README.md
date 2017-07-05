# Stein

A library for general Bayesian inference using Stein variational gradient descent.

## Augmentations

1. This library should be rewritten to leverage TensorFlow instead of Theano.
2. The way internal particles are handled is unwieldy. Instead of vectors, let's work intuitively with dictionaries. This will allow us to correctly access the variables we need; the logic of concatenating dictionary values consistently into vectors can be handled behind the scenes.
3. Nobody likes this business of partial functions for evaluation. Would it be possible to incorporate a "train-on-batch" function?
