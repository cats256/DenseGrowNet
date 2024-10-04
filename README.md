![image](https://github.com/user-attachments/assets/0f45b34b-19de-4abe-aba1-238d207e85a4)

Update 10/2: Check out dense-grownet-benchmark (preliminary) and dense-grownet-architecture.ipynb

Update 9/25: Benchmarking results will be uploaded hopefully in the next couple of weeks

Tree-based models have long been state-of-the-art on medium-sized tabular data, despite extensive research on deep learning for this type of data. Certain inductive biases of tree-based models, such as their rotationally variant learning procedure and robustness against uninformative features (1), contribute to their strong performance on tabular data, in contrast to MLPs' rotationally invariant learning procedure and high capacity for overfitting. This project aims to apply the inductive biases of tree-based methods on MLP-like neural nets and improve their performance on tabular data.

This architecture proposes a DenseNet-like design (2) with a specific form of 'looks linear' initialization (3), where the network preserves the original orientation of features at initialization and can produce an identity input-output mapping. L1 and L2 regularization is also proposed in addition to this architecture. A further explanation of 'looks linear' initialization is that a network with 'looks linear' initialization "starts out as a linear network with latent nonlinear behavior that becomes more pronounced as the two blocks diverge through updates". (4)

The preservation of the original orientation of features at initialization and initial identity input-output mapping is important as it alleviates the downside of a rotationally invariant learning procedure, where "intuitively, to remove uninformative features, a rotationally invariant algorithm has to first find the original orientation of the features, and then select the least informative ones: the information contained in the orientation of the data is lost." (1). This design also achieves dynamical isometry (4), where the singular values of the network's input-output Jacobian concentrate near 1, which "has been shown to dramatically speed up learning", avoid vanishing/exploding gradients, and appears to improve generalization performance. The residual connections in a DenseNet-like architecture also pose a similarity in its focus on learning the residuals, similar to boosting algorithms, which also tend to be the best-performing class of tree-based models. This similarity is apparent in the many papers published that aim to understand residual networks through boosting theory.

It is known that L1 regularization is rotationally invariant and logistic regression with L1 regularization is robust against uninformative features, where sample complexity grows only logarithmically with the number of irrelevant features (5). One can, then, see that L1 regularization is desirable to include as part of the learning procedure, as it mirrors the inductive biases that contribute to tree-based models' strong performance on tabular data. Overparameterization and the replacement of ReLU with Softplus under this architecture also appear to improve generalization performance, although the effect is very minor.

To do:
- Check out the convex optimization of a two-layer neural network. https://scnn.readthedocs.io/en/latest/quick_start.html. https://stanford.edu/~wangyf18/tutorial_cvxnn.html. https://arxiv.org/abs/2002.10553. https://web.stanford.edu/class/ee364b/lectures/convexNN.pdf
- Needs to include smooth versions of ReLUs and their advantages in README, which include improved robustness (7) and smoother optimization landscapes (8) at very little trade-off to accuracy.
- Take a look at memory-efficient implementations of DenseNets. In current implementations, DenseNet consumes quadratic memory with respect to depth but since feature maps are reused almost everywhere, through some implementation tricks DenseNet can also be implemented in linear memory (9) (10).
- Look at deep learning significance testing https://deep-significance.readthedocs.io/en/latest/
- Experiment with progressively growing a 2-layer neural network where after training/validation loss plateau, add more neurons to each layer, but not more layers, and keep the weights of the neuron previously trained frozen. This is similar to gradient-boosting
- Experiment with progressively growing a neural network where a new layer is added after training/validation loss plateau, where the previous layers are frozen but all of their feature maps or outputs are used for the new layer and the final layer, and the weights of the original final layer is copied over to the new final layer (each weight still correspond to the same feature map), where it is bigger. 

Contact: nhatbui@tamu.edu (would be great if someone is looking to discuss, collaborate, or act as a mentor on this research project XD ) 

Cool read:
- https://stats.stackexchange.com/questions/339054/what-values-should-initial-weights-for-a-relu-network-be
- https://proceedings.neurips.cc/paper/2020/file/f3f27a324736617f20abbf2ffd806f6d-Paper.pdf
- https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html
- https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html
- https://mindfulmodeler.substack.com/p/inductive-biases-of-the-random-forest
- https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html
- https://openreview.net/forum?id=UgBo_nhiHl (not the same thing as what I'm doing, this was discovered after)

Reference: 
1) Why do tree-based models still outperform deep learning on typical tabular data?

   https://openreview.net/pdf?id=Fp7__phQszn
3) Densely Connected Convolutional Networks

   https://arxiv.org/pdf/1608.06993
4) The Shattered Gradients Problem: If resnets are the answer, then what is the question?

   https://proceedings.mlr.press/v70/balduzzi17b/balduzzi17b.pdf
5) "looks-linear" initialization explanation

   https://www.reddit.com/r/MachineLearning/comments/5yo30r/comment/desyjot/
6) Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice

   https://arxiv.org/pdf/1711.04735
7) Feature selection, L1 vs. L2 regularization, and rotational invariance

   https://icml.cc/Conferences/2004/proceedings/papers/354.pdf
8) Smooth Adversarial Training

   https://arxiv.org/abs/2006.14536
9) Reproducibility in Deep Learning and Smooth Activations

   https://research.google/blog/reproducibility-in-deep-learning-and-smooth-activations/

10) How does DenseNet compare to ResNet and Inception?

    https://www.reddit.com/r/MachineLearning/comments/67fds7/comment/dgrrp54/
11) Memory-Efficient Implementation of DenseNets

    https://arxiv.org/abs/1707.06990
