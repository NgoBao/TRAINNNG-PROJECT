import numpy as np
from utils import sigmoid, sigmoid_derivative, softmax, cross_entropy_loss


class MLP:
    """
    2-layer MLP for MNIST classification.
    Architecture:
        Input(784) -> FC(hidden_size) + Sigmoid -> FC(10) + Softmax
    """

    def __init__(self, input_size=784, hidden_size=128, output_size=10, learning_rate=0.1, seed=42):
        np.random.seed(seed)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Random initialization for weights, zero initialization for biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size), dtype=np.float64)

        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size), dtype=np.float64)

        # Cache for forward pass
        self.X = None
        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.Y_hat = None

    def forward(self, X):
        """
        Forward propagation.
        X shape: [batch_size, 784]
        Returns:
            probabilities shape: [batch_size, 10]
        """
        self.X = X
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.Y_hat = softmax(self.Z2)
        return self.Y_hat

    def compute_loss(self, y_true):
        """
        Compute cross-entropy loss.
        y_true shape: [batch_size, 10]
        """
        return cross_entropy_loss(self.Y_hat, y_true)

    def backward(self, y_true):
        """
        Backward propagation and parameter update.
        y_true shape: [batch_size, 10]
        """
        batch_size = y_true.shape[0]

        # Output layer gradient
        dZ2 = (self.Y_hat - y_true) / batch_size
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer gradient
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(self.A1)
        dW1 = np.dot(self.X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Gradient descent update
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def predict(self, X):
        """
        Return predicted class indices.
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def evaluate_accuracy(self, X, labels):
        """
        Evaluate accuracy on a batch.
        """
        preds = self.predict(X)
        labels = labels.reshape(-1)
        return np.mean(preds == labels)