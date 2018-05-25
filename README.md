# MLlib
This is a repository contains simple machine learning algorithms

So far, it contains a kernel SVM, which use sgd to optimize, and a k nearest neighbour classifier, a PCA tool, and a SVM with gradient descent.

The motivation to write the SVM is the representer theorem, which means that the best classifier on training
set can be represented just with linear combinations of samples with kernel applied to them. Normally speaking, SVM is trained with algorithms 
like SMO. However, with representer theorem, a lot of machine learning models can be represented with a loss and a regularization term, and the 
only difference is loss. Thus, we can use algorithms like SGD to train, which is fast. Most importantly, an unconstrained convex optimization problem, good.

This project is for learning usage.
