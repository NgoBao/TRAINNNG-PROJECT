import numpy as np
from utils import relu, relu_derivative, softmax, cross_entropy_loss


class SimpleCNN:
    """
    Manual CNN for MNIST classification.
    Architecture:
        Input(28x28) -> Conv(valid, 1 kernel) + ReLU -> Flatten -> FC(10) + Softmax
    """

    def __init__(self, image_size=28, kernel_size=3, num_classes=10, learning_rate=0.01, seed=42):
        np.random.seed(seed)

        self.image_size = image_size
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # One convolution kernel
        self.kernel = np.random.randn(kernel_size, kernel_size) * 0.01
        self.conv_bias = 0.0

        self.conv_output_size = image_size - kernel_size + 1
        self.flatten_size = self.conv_output_size * self.conv_output_size

        # Fully connected layer
        self.W = np.random.randn(self.flatten_size, num_classes) * 0.01
        self.b = np.zeros((1, num_classes), dtype=np.float64)

        # Cache
        self.X = None
        self.conv_out = None
        self.relu_out = None
        self.flat = None
        self.logits = None
        self.Y_hat = None

    def convolve_single_image(self, image):
        """
        Perform valid convolution on one image using one kernel.
        image shape: [28, 28]
        output shape: [28-k+1, 28-k+1]
        """
        h, w = image.shape
        k = self.kernel_size
        out_h = h - k + 1
        out_w = w - k + 1

        output = np.zeros((out_h, out_w), dtype=np.float64)

        for i in range(out_h):
            for j in range(out_w):
                region = image[i:i + k, j:j + k]
                output[i, j] = np.sum(region * self.kernel) + self.conv_bias

        return output

    def forward(self, X):
        """
        Forward propagation.
        X shape: [batch_size, 28, 28]
        """
        self.X = X
        batch_size = X.shape[0]

        conv_results = []
        relu_results = []

        for n in range(batch_size):
            conv_map = self.convolve_single_image(X[n])
            activated = relu(conv_map)
            conv_results.append(conv_map)
            relu_results.append(activated)

        self.conv_out = np.array(conv_results, dtype=np.float64)
        self.relu_out = np.array(relu_results, dtype=np.float64)

        self.flat = self.relu_out.reshape(batch_size, -1)
        self.logits = np.dot(self.flat, self.W) + self.b
        self.Y_hat = softmax(self.logits)

        return self.Y_hat

    def compute_loss(self, y_true):
        """
        Compute cross-entropy loss.
        """
        return cross_entropy_loss(self.Y_hat, y_true)

    def backward(self, y_true):
        """
        Backward propagation and parameter update.
        Computes gradients for:
        - FC weights/bias
        - Conv kernel/bias
        """
        batch_size = y_true.shape[0]
        k = self.kernel_size

        # FC layer gradients
        dLogits = (self.Y_hat - y_true) / batch_size
        dW = np.dot(self.flat.T, dLogits)
        db = np.sum(dLogits, axis=0, keepdims=True)

        dFlat = np.dot(dLogits, self.W.T)
        dRelu = dFlat.reshape(self.relu_out.shape)

        # Conv layer gradients
        dKernel = np.zeros_like(self.kernel, dtype=np.float64)
        dConvBias = 0.0

        for n in range(batch_size):
            dConv = dRelu[n] * relu_derivative(self.conv_out[n])
            dConvBias += np.sum(dConv)

            for i in range(dConv.shape[0]):
                for j in range(dConv.shape[1]):
                    region = self.X[n, i:i + k, j:j + k]
                    dKernel += dConv[i, j] * region

        # Parameter update
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db
        self.kernel -= self.learning_rate * dKernel
        self.conv_bias -= self.learning_rate * dConvBias

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