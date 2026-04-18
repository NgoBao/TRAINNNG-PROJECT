# CSC4851/6851 Course Project - MNIST Classification

## Project Overview
This project manually implements two neural network models for MNIST handwritten digit classification:

1. **MLP (Multi-Layer Perceptron)**
   - Input layer: 784 features
   - Hidden layer: fully connected + Sigmoid
   - Output layer: 10 neurons + Softmax

2. **CNN (Convolutional Neural Network)**
   - One valid convolution kernel
   - ReLU activation
   - Fully connected output layer with 10 neurons + Softmax

The project follows the course requirement of implementing the neural network logic manually without using deep learning libraries such as `torch.nn` or `tf.keras`.

## Files
- `main.py` - project entry point
- `train.py` - training and testing pipeline
- `mlp.py` - manual MLP implementation
- `cnn.py` - manual CNN implementation
- `utils.py` - helper functions
- `read_MNIST.py` - dataloader for MNIST

## Requirements
Recommended environment:
- Python 3.10 or newer
- NumPy
- PyTorch
- torchvision

## Install Dependencies
```bash
pip install numpy torch torchvision