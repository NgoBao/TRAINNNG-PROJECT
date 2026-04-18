import numpy as np
from mlp import MLP
from cnn import SimpleCNN
from utils import (
    to_numpy,
    one_hot_encode,
    flatten_images,
    prepare_cnn_images,
    normalize_images,
    accuracy_score,
)


def train_mlp(model, train_loader, epochs=30):
    """
    Train the MLP model.
    """
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for images, labels in train_loader:
            images = flatten_images(images)
            images = normalize_images(images)

            labels_np = to_numpy(labels).astype(int).reshape(-1)
            y_true = one_hot_encode(labels_np, num_classes=10)

            y_pred = model.forward(images)
            loss = model.compute_loss(y_true)
            model.backward(y_true)

            acc = accuracy_score(y_pred, labels_np)

            epoch_loss += loss
            epoch_acc += acc
            num_batches += 1

        print(
            f"[MLP] Epoch {epoch + 1:02d}/{epochs} | "
            f"Loss: {epoch_loss / num_batches:.4f} | "
            f"Accuracy: {epoch_acc / num_batches:.4f}"
        )


def test_mlp(model, test_loader):
    """
    Evaluate the MLP model.
    """
    total_correct = 0
    total_samples = 0

    for images, labels in test_loader:
        images = flatten_images(images)
        images = normalize_images(images)

        labels_np = to_numpy(labels).astype(int).reshape(-1)

        preds = model.predict(images)
        total_correct += np.sum(preds == labels_np)
        total_samples += labels_np.shape[0]

    test_acc = total_correct / total_samples
    print(f"[MLP] Test Accuracy: {test_acc:.4f}")
    return test_acc


def train_cnn(model, train_loader, epochs=5):
    """
    Train the CNN model.
    """
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for images, labels in train_loader:
            images = prepare_cnn_images(images)
            images = normalize_images(images)

            labels_np = to_numpy(labels).astype(int).reshape(-1)
            y_true = one_hot_encode(labels_np, num_classes=10)

            y_pred = model.forward(images)
            loss = model.compute_loss(y_true)
            model.backward(y_true)

            acc = accuracy_score(y_pred, labels_np)

            epoch_loss += loss
            epoch_acc += acc
            num_batches += 1

        print(
            f"[CNN] Epoch {epoch + 1:02d}/{epochs} | "
            f"Loss: {epoch_loss / num_batches:.4f} | "
            f"Accuracy: {epoch_acc / num_batches:.4f}"
        )


def test_cnn(model, test_loader):
    """
    Evaluate the CNN model.
    """
    total_correct = 0
    total_samples = 0

    for images, labels in test_loader:
        images = prepare_cnn_images(images)
        images = normalize_images(images)

        labels_np = to_numpy(labels).astype(int).reshape(-1)

        preds = model.predict(images)
        total_correct += np.sum(preds == labels_np)
        total_samples += labels_np.shape[0]

    test_acc = total_correct / total_samples
    print(f"[CNN] Test Accuracy: {test_acc:.4f}")
    return test_acc


def main(train_loader, test_loader):
    """
    Main function for training and testing both MLP and CNN.
    """

    print("\n" + "=" * 60)
    print("Training MLP")
    print("=" * 60)
    mlp_model = MLP(
        input_size=784,
        hidden_size=128,
        output_size=10,
        learning_rate=0.1,
        seed=42
    )
    train_mlp(mlp_model, train_loader, epochs=30)
    test_mlp(mlp_model, test_loader)

    print("\n" + "=" * 60)
    print("Training CNN")
    print("=" * 60)
    cnn_model = SimpleCNN(
        image_size=28,
        kernel_size=3,
        num_classes=10,
        learning_rate=0.01,
        seed=42
    )
    train_cnn(cnn_model, train_loader, epochs=5)
    test_cnn(cnn_model, test_loader)