Update 9/25: Pretty busy at the moment but benchmarking results will be uploaded hopefully in the next couple of weeks

Tree-based models have long been state-of-the-art on medium-sized data despite extensive research on deep learning for tabular data. It is known that certain inductive biases of tree-based models, such as their rotationally variant learning procedure and robustness against uninformative features, contribute to their strong performance on tabular data, in contrast to MLPs' rotationally variant learning procedure and high capacity for overfitting. This project aims to apply the inductive biases of tree-based methods on MLP-like neural nets and improve their performance on tabular data.

This architecture proposes a DenseNet-like design (2) with 'looks linear' initialization (3), in addition to L1 and L2 regularization. A DenseNet-like design with 'looks linear' initialization preserves the original orientation of features at initialization. This is important as "intuitively, to remove uninformative features, a rotationally invariant algorithm has to first find the original orientation of the features, and then select the least informative ones: the information contained in the orientation of the data is lost." (1). This design also achieves dynamical isometry (4), where the singular values of the network's input-outputs concentrate near 1, which "has been shown to dramatically speed up learning", avoid vanishing/exploding gradients, and appears to improve generalization performance. The residual connections in a DenseNet-like architecture also pose a similarity in its focus on learning the residuals, similar to boosting algorithms, which also tend to be the best-performing class of tree-based models. This similarity is apparent in the many papers published that aim to understand residual networks through boosting theory. It is also known that logistic regression with L1 regularization is a rotationally variant learning procedure and is robust against uninformative features, where sample complexity grows only logarithmically with the number of irrelevant features (5). Then, one can find that L1 regularization is desirable to include as part of the learning procedure, as this mirrors the inductive biases that contribute to tree-based models' strong performance on tabular data. Overparameterization and the replacement of ReLU with Softplus under this architecture also appear to improve validation loss, although the effect is very minor.

To do:
- Need to include smooth versions of ReLUs and their advantages in README, which include smoother optimization landscapes and improved robustness at very little trade-off to accuracy.
- Take a look at memory-efficient implementations of DenseNets. In current implementations, DenseNet consumes quadratic memory with respect to depth but since feature maps are reused almost everywhere, through some implementation tricks DenseNet can also be implemented in linear memory (7) (8).

Contact: nhatbui@tamu.edu (would be great if someone is looking to collaborate with or act as a mentor on this research project XD)

Reference: 
1) Why do tree-based models still outperform deep learning on typical tabular data? https://openreview.net/pdf?id=Fp7__phQszn
2) Densely Connected Convolutional Networks https://arxiv.org/pdf/1608.06993
3) The Shattered Gradients Problem: If resnets are the answer, then what is the question? https://proceedings.mlr.press/v70/balduzzi17b/balduzzi17b.pdf
4) Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice https://arxiv.org/pdf/1711.04735
5) Feature selection, L1 vs. L2 regularization, and rotational invariance https://icml.cc/Conferences/2004/proceedings/papers/354.pdf
6) Smooth Adversarial Training https://arxiv.org/abs/2006.14536
7) https://www.reddit.com/r/MachineLearning/comments/67fds7/comment/dgrrp54/
8) Memory-Efficient Implementation of DenseNets https://arxiv.org/abs/1707.06990
