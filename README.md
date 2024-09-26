It is known that one inductive bias for tree-based models that enables them to perform so well is their rotationally invariant learning procedures and robustness against uninformative features, in contrast to MLP-like neural networks (1). This project aims to apply the inductive biases of tree-based methods that enable them to perform so well on MLP-like neural networks. This architecture involves the preservation of the orientation of features at initialization across layers and achieving dynamical isometry by utilizing residual connections and "looks linear" initialization (2), in addition to L1 and L2 regularization. The preservation of the original orientation of features is important as "Intuitively, to remove uninformative features, a rotationaly invariant algorithm has to first find the original orientation of the features, and then select the least informative ones: the information contained in the orientation of the data is lost." (1). It is known that L1 regularization is a rotationally invariant learning procedure and is robust against uninformative features, where sample complexity grows only logarithmically with the number of irrelevant features (3). Overparameterization and replacement of ReLU with Softplus under this architecture also appears to improve validation loss, although the effect is very minor.

Contact: nhatbui@tamu.edu

Update 9/25: Pretty busy at the moment but results will be uploaded hopefully in the next couple of weeks

Reference: 
1) Why do tree-based models still outperform deep learning on typical tabular data? https://openreview.net/pdf?id=Fp7__phQszn
2) The Shattered Gradients Problem: If resnets are the answer, then what is the question? https://proceedings.mlr.press/v70/balduzzi17b/balduzzi17b.pdf
3) Feature selection, L1 vs. L2 regularization, and rotational invariance https://icml.cc/Conferences/2004/proceedings/papers/354.pdf
