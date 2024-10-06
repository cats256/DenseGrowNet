Update 10/5: Benchmarking results will be uploaded hopefully in the next couple of weeks to two months

![image](https://github.com/user-attachments/assets/775d3268-776f-4179-b5d1-871223d6718a)

### Intro
Tree-based models have long been and continue to be state-of-the-art on medium-sized tabular data. This is despite extensive deep-learning research on tabular data. Gradient-boosted decision trees (GDBTs), in particular, tend to outperform other types of tree-based models. Certain inductive biases of tree-based models, such as their rotationally variant learning procedure, which extracts information based on features' orientation, and robustness against uninformative features (1), contribute to their strong performance on tabular data, in contrast to MLPs' rotationally invariant learning procedure and high capacity for overfitting. Gradient boosting's inductive bias can be explained as the bias towards explaining the largest proportion of variance through simpler interaction terms, with the contribution to variance decreasing as the order of interaction increases, rather than a large amount of high-order interaction terms, each explaining a small amount of variance. See more explanation at (9). This project aims to improve neural networks' performance on tabular data through a focus on investigating and applying the inductive biases of tree-based models, particularly, GDBTs, on MLP-like neural nets.

### Method
A gradient boosting technique for neural networks is proposed, with superior performance on tabular data and applicability to other forms of data and neural networks.  The first, base model is a one-layer zero-initialized neural network, which is a generalized linear model. A dedicated solver can be used for this step as we are only interested in the raw outputs of the first model. The second model, with the first layer "looks linear" initialized, the second layer zero-initialized, and the activation function belonging to the ReLU-family function, is then trained to predict the residuals or correct the error of the previous model. 'looks linear' initialization can be described as initializing in this pattern. [[1  0 ... 0  0], [-1 0 ... 0  0], [0  1 ... 0  0], [0 -1 ... 0  0], ..., [0  0 ... 1  0], [0  0 ... -1 0], [0  0 ... 0  1], [0  0 ... 0 -1]], where there are 2N neurons for N features and the second, final layer can easily replicate linear inputs as max(0, x) - max(0, -x) = x. For further explanation of 'looks linear' initialization see (4). The third model is trained in a similar fashion, however, the intermediate features produced by the first layer of the previous, or second, model are scaled and concatenated to the features used for training. Similarly, train the next k models as the process for the third model. For further clarification, see the diagram above. Adjust regularization, learning rate, and epoch as appropriate.

### Explanation
The original orientation of features at initialization is preserved through 'looks linear' initialization. This is important as it alleviates the downside of a rotationally invariant learning procedure, where "intuitively, to remove uninformative features, a rotationally invariant algorithm has to first find the original orientation of the features, and then select the least informative ones: the information contained in the orientation of the data is lost." (1). This also achieves dynamical isometry, where the singular values of the network's input-output Jacobian concentrate near 1, which "has been shown to dramatically speed up learning", avoid vanishing/exploding gradients, and appears to improve generalization performance (5). Gradient boosting for neural networks where each model is only trained on the original features proved infeasible to train as training loss tends to decrease very slowly. Hence, intermediate features from previous models along with the original features are used to train each model instead, which demonstrated greatly improved training speed. This also appears to have slightly greater generalization performance compared to just training on original features for each model based on a preliminary experiment.

### Additional
Elastic net regularization, which is L1 + L2 regularization, should be included as part of the training procedure, where both L1 and L2 regularization promote smaller weights. L1 regularization corresponds to the Laplace distribution, where weights tend to be sparse and are penalized proportionally based on their sizes. L2 regularization corresponds to the normal distribution, where weights are penalized quadratically based on their sizes. It is known that L1 regularization is rotationally invariant and logistic regression with L1 regularization is robust against uninformative features, where sample complexity grows only logarithmically with the number of irrelevant features (6). One can, then, see that L1 regularization is also desirable to include as part of the learning procedure, as it mirrors the inductive biases that contribute to tree-based models' strong performance on tabular data. 

Note: Softplus appears to improve generalization and trainability, provided that the curvature of softplus is small, eg. softplus(4 * x) / 4. This is because softplus has a tendency to mimic a linear or identity activation. If most of your values are concentrated at a very small range around 0, say from -0.2 to 0.2, softplus will look more like a linear or identity function rather than a piecewise linear function like ReLU, see example below. This project, however, will not benchmark the result due to lack of time.

![image](https://github.com/user-attachments/assets/6f822155-aeea-45fe-8114-fb6c46a2d893)

Contact: nhatbui@tamu.edu (would be great if someone is looking to discuss, collaborate, or act as a mentor on this research project :D )

Cool read:
- https://stats.stackexchange.com/questions/339054/what-values-should-initial-weights-for-a-relu-network-be
- https://proceedings.neurips.cc/paper/2020/file/f3f27a324736617f20abbf2ffd806f6d-Paper.pdf
- https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html
- https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html
- https://mindfulmodeler.substack.com/p/inductive-biases-of-the-random-forest
- https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html
- https://openreview.net/forum?id=UgBo_nhiHl (not the same thing as what I'm doing, this was discovered after)

Keywords spam:
GrowNet, DenseNet, ResNet, neural networks, deep learning, dynamical isometry, 'looks linear' initialization, ReLU, Softplus, activation function, gradient boosting, inductive bias, decision trees, regularization, L1, L2, elastic net

To do:
- Smooth versions of ReLUs and their advantages in README need to be included. This includes improved robustness (7) and smoother optimization landscapes (8) with little accuracy trade-off.
- Look at deep learning significance testing. https://deep-significance.readthedocs.io/en/latest
- Check out the convex optimization of a two-layer ReLU neural network.
   - https://scnn.readthedocs.io/en/latest/quick_start.html.
   - https://stanford.edu/~wangyf18/tutorial_cvxnn.html.
   - https://arxiv.org/abs/2002.10553.
   - https://web.stanford.edu/class/ee364b/lectures/convexNN.pdf
   - https://iclr-blogposts.github.io/2024/blog/hidden-convex-relu/


### References: 
1) Why do tree-based models still outperform deep learning on typical tabular data?

   https://openreview.net/pdf?id=Fp7__phQszn
2) Densely Connected Convolutional Networks

   https://arxiv.org/pdf/1608.06993
3) The Shattered Gradients Problem: If resnets are the answer, then what is the question?

   https://proceedings.mlr.press/v70/balduzzi17b/balduzzi17b.pdf
4) "looks-linear" initialization explanation

   https://www.reddit.com/r/MachineLearning/comments/5yo30r/comment/desyjot/
5) Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice

   https://arxiv.org/pdf/1711.04735
6) Feature selection, L1 vs. L2 regularization, and rotational invariance

   https://icml.cc/Conferences/2004/proceedings/papers/354.pdf
7) Smooth Adversarial Training

   https://arxiv.org/abs/2006.14536
8) Reproducibility in Deep Learning and Smooth Activations

   https://research.google/blog/reproducibility-in-deep-learning-and-smooth-activations/

9) Tim Goodman's explanation of gradient boosting's inductive bias

    https://stats.stackexchange.com/questions/173390/gradient-boosting-tree-vs-random-forest#comment945015_174020
