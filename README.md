It is known that some inductive biases of tree-based models that enable them to perform so well on tabular data are their rotationally invariant learning nature and robustness against uninformative features, in contrast to MLP-like neural networks (1). This project aims to apply the inductive biases of tree-based methods on MLP-like neural networks. This architecture involves the preservation of the orientation of features at initialization across layers and achieving dynamical isometry by utilizing residual connections and "looks linear" initialization (2), in addition to L1 and L2 regularization. The preservation of the original orientation of features is important as "intuitively, to remove uninformative features, a rotationally invariant algorithm has to first find the original orientation of the features, and then select the least informative ones: the information contained in the orientation of the data is lost." (1). It is known that L1 regularization is a rotationally invariant learning procedure and is robust against uninformative features, where sample complexity grows only logarithmically with the number of irrelevant features (3). Overparameterization and replacement of ReLU with Softplus under this architecture also appear to improve validation loss, although the effect is very minor.

Contact: nhatbui@tamu.edu (would be great if someone is looking to collaborate with or act as a mentor on this research project XD)

Update 9/25: Pretty busy at the moment but benchmarking results will be uploaded hopefully in the next couple of weeks

Reference: 
1) Why do tree-based models still outperform deep learning on typical tabular data? https://openreview.net/pdf?id=Fp7__phQszn
2) The Shattered Gradients Problem: If resnets are the answer, then what is the question? https://proceedings.mlr.press/v70/balduzzi17b/balduzzi17b.pdf
3) Feature selection, L1 vs. L2 regularization, and rotational invariance https://icml.cc/Conferences/2004/proceedings/papers/354.pdf
