# ML-Assignments
These are my submissions for CSE343 Machine Learning assignments. 

# Highlights Assignment1
1. Used TSNE (t-distributed stochastic neighbour embedding) algorithm to reduce
data dimensions of MNIST dataset.

2. Implemented Linear Regression for the Abalone Dataset. Modified Regression implementation by including L1 (LASSO)
and L2 (Ridge Regression) regularization.

3. Implemented Binary Logistic Regression for the UCI Ionosphere dataset. Reduced the number of features via Principal Component
Analysis (PCA), compared with scikit-learn's Logistic Regression with L1 and L2 regularization, plotted ROC-AUC curve.

4. Implemented Multiclass Logistic Regression for the MNIST dataset using training methodologies OVO(One-vs-One) and OVR(One-vs-Rest)

# Highlights Assignment2
1. Wrote code from scratch for Neural Networks and tuned the model on FASHION MNIST.

2. Implemented the same architecture with the same hyperparameters using sklearn’s
MLP classifier 

3. Used the existing VGG16 model in Pytorch (pretrained on ImageNet) on the FASHION MNIST dataset and fine tuned it by freezing lower layers
and replacing upper layers with dense layers. Plotted graphs and results using WandB.

4. Visualized Convolution layers of VGG16.

