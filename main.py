from read_MNIST import load_data
from train import main as run_training


if __name__ == "__main__":
    print("=" * 60)
    print("Loading MNIST dataset...")
    print("=" * 60)

    train_loader, test_loader = load_data()

    print("=" * 60)
    print("Starting project training and testing")
    print("=" * 60)

    run_training(train_loader, test_loader)
