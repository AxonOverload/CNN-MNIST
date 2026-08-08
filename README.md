# CNN-MNIST
An MNIST digit classifier CNN built from scratch using NumPy and PyTorch.

## How the Model Works

### Forward Pass
Convolution: Applied eight 3,3 filters across the input images to extract low-level spatial features.
ReLU Activation: Applied the Rectified Linear Unit (ReLu) element-wise to introduce non-linearity.
Max Pooling: Reduced spatial dimensions by extracting the maximum value within each pooling window.
Flattening & Dense Layers: Reshaped the pooled feature maps into a 1D vector and passed them through fully connected layers via linear transformation
SoftMax Output: Applied the SoftMax function to convert raw output logits into class probability distributions.

### Backward Pass
Chain Rule & Dense Gradients: Used the multivariable chain rule to compute partial derivatives for dense weights and biases.
UnPooling & Masking: Reshaped the input gradients back to the pooled feature map dimensions. A binary mask matrix (1 for max values, 0 otherwise) routed gradients exclusively through the max pooling indices.
Convolutional Gradients: Computed filter gradients by convolving image patches with output feature gradients (handling dimensions appropriately to support array broadcasting).
Optimization: Updated weights, biases, and convolutional filters using Stochastic Gradient Descent (SGD).

## Results
Accuracy achieved on the test set:
- CNN in NumPy — 97.57% MNIST
- CNN in PyTorch — 98.86% MNIST
| **CNN in PyTorch** | **98.86%** |
| **CNN in NumPy** | **97.57%** |
