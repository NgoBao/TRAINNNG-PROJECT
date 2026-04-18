import numpy as np


def to_numpy(x):
    """
    Convert input to numpy array.
    Works for numpy arrays and torch tensors.
    """
    if isinstance(x, np.ndarray):
        return x
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return np.array(x)


def one_hot_encode(labels, num_classes=10):
    """
    Convert integer labels to one-hot vectors.
    labels shape: [batch_size]
    output shape: [batch_size, num_classes]
    """
    labels = to_numpy(labels).astype(int).reshape(-1)
    one_hot = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


def accuracy_score(y_pred_probs, labels):
    """
    Compute classification accuracy from predicted probabilities.
    """
    labels = to_numpy(labels).astype(int).reshape(-1)
    preds = np.argmax(y_pred_probs, axis=1)
    return np.mean(preds == labels)


def sigmoid(x):
    """
    Sigmoid activation.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(sigmoid_output):
    """
    Derivative of sigmoid using already-computed sigmoid output.
    """
    return sigmoid_output * (1.0 - sigmoid_output)


def relu(x):
    """
    ReLU activation.
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    Derivative of ReLU with respect to input x.
    """
    return (x > 0).astype(np.float64)


def softmax(x):
    """
    Numerically stable softmax.
    x shape: [batch_size, num_classes]
    """
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    """
    Cross-entropy loss for one-hot targets.
    y_pred shape: [batch_size, num_classes]
    y_true shape: [batch_size, num_classes]
    """
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def flatten_images(images):
    """
    Flatten images for MLP.
    Accepts:
    - [batch_size, 1, 28, 28]
    - [batch_size, 28, 28]
    Returns:
    - [batch_size, 784]
    """
    images = to_numpy(images).astype(np.float64)

    if images.ndim == 4:
        images = images.reshape(images.shape[0], -1)
    elif images.ndim == 3:
        images = images.reshape(images.shape[0], -1)
    else:
        raise ValueError(f"Unsupported image shape for flatten_images: {images.shape}")

    return images


def prepare_cnn_images(images):
    """
    Prepare images for CNN.
    Accepts:
    - [batch_size, 1, 28, 28]
    - [batch_size, 28, 28]
    Returns:
    - [batch_size, 28, 28]
    """
    images = to_numpy(images).astype(np.float64)

    if images.ndim == 4:
        if images.shape[1] != 1:
            raise ValueError("This CNN implementation expects grayscale images with 1 channel.")
        images = images[:, 0, :, :]
    elif images.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported image shape for prepare_cnn_images: {images.shape}")

    return images


def normalize_images(images):
    """
    Normalize image values to [0, 1] if needed.
    """
    images = images.astype(np.float64)
    if np.max(images) > 1.0:
        images = images / 255.0
    return images