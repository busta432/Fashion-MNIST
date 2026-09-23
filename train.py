import numpy as np
import gzip
import argparse
import matplotlib.pyplot as plt
import autograd.numpy as anp
from autograd import grad
import time
import os
import copy

from nn import LinearClassifier, TwoLayerMLP, ThreeLayerMLP


def create_validation_split(X_train, y_train, val_ratio=0.2, random_seed=42):
    """
    Create validation set by randomly selecting samples from each class
    Args:
        X_train: Training features
        y_train: Training labels
        val_ratio: Fraction of training data to use for validation
        random_seed: Random seed for reproducibility
    Returns:
        Tuple of (X_train_new, y_train_new, X_val, y_val)
    """
    np.random.seed(random_seed)
    
    print(f"\nCreating validation split with {val_ratio*100:.0f}% of training data...")
    
    # Get unique classes
    unique_classes = np.unique(y_train)
    
    # Store indices for train and validation sets
    train_indices = []
    val_indices = []
    
    for class_id in unique_classes:
        # Get indices for this class
        class_indices = np.where(y_train == class_id)[0]
        
        # Calculate validation size for this class
        class_size = len(class_indices)
        val_size = int(class_size * val_ratio)
        train_size = class_size - val_size
        
        # Randomly split the class indices
        np.random.shuffle(class_indices)
        val_class_indices = class_indices[:val_size]
        train_class_indices = class_indices[val_size:]
        
        train_indices.extend(train_class_indices)
        val_indices.extend(val_class_indices)
        
        print(f"  Class {class_id}: {class_size} -> Train: {train_size}, Val: {val_size}")
    
    # Convert to numpy arrays and sort
    train_indices = np.sort(np.array(train_indices))
    val_indices = np.sort(np.array(val_indices))
    
    # Split the data
    X_train_new = X_train[train_indices]
    y_train_new = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    
    print(f"\nFinal split:")
    print(f"  Training set: {X_train_new.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")
    
    return X_train_new, y_train_new, X_val, y_val


def load_fashion_mnist_data(train_file, test_file):
    """
    Load Fashion-MNIST dataset from compressed CSV files
    Args:
        train_file: Path to training data file
        test_file: Path to test data file
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    print("Loading Fashion-MNIST dataset...")
    
    # Load training data
    with gzip.open(train_file, 'rt') as f:
        train_data = np.loadtxt(f, delimiter=',', skiprows=1)
    
    # Load test data
    with gzip.open(test_file, 'rt') as f:
        test_data = np.loadtxt(f, delimiter=',', skiprows=1)
    
    # Extract features and labels
    # First column is label, rest are pixel values
    X_train = train_data[:, 1:].astype(np.float32)
    y_train = train_data[:, 0].astype(np.int32)
    
    X_test = test_data[:, 1:].astype(np.float32)
    y_test = test_data[:, 0].astype(np.int32)
    
    # Normalize pixel values to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    return X_train, y_train, X_test, y_test


def one_hot_encode(labels, num_classes):
    """
    Convert labels to one-hot encoding
    Args:
        labels: Array of integer labels
        num_classes: Number of classes
    Returns:
        One-hot encoded labels
    """
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def softmax(z):
    """Softmax activation function"""
    exp_z = anp.exp(z)
    return exp_z / anp.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    """
    Compute cross-entropy loss with softmax
    Args:
        y_pred: Predicted logits of shape (batch_size, num_classes)
        y_true: True one-hot encoded labels of shape (batch_size, num_classes)
    Returns:
        Average cross-entropy loss
    """
    # Apply softmax to get probabilities
    probs = softmax(y_pred)
    
    # TODO: Compute cross-entropy loss with softmax
    #
    # Cross-entropy loss measures the difference between predicted probabilities
    # and true labels. It's commonly used for classification problems.
    #
    # Mathematical formula: 
    # loss = -mean(sum(y_true * log(probs), axis=1))
    #
    # Steps:
    # 1. Apply softmax to convert logits to probabilities: probs = softmax(y_pred)
    # 2. Compute element-wise: y_true * log(probs)
    #    - Use anp.log() for autograd compatibility
    # 3. Sum across classes for each sample
    # 4. Apply negative sign and take mean
    #
    # Example:
    # If y_true = [[0, 1, 0], [1, 0, 0]] and probs = [[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]]
    # Then loss = -mean([log(0.8), log(0.9)]) = -mean([-0.223, -0.105]) = 0.164
    
    raise NotImplementedError
    
    return loss


def create_batches(X, y, batch_size, shuffle=True):
    """
    Create mini-batches for training
    Args:
        X: Input features
        y: Labels
        batch_size: Size of each batch
        shuffle: Whether to shuffle the data
    Returns:
        Generator yielding batches of (X_batch, y_batch)
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


def compute_accuracy(model, X, y):
    """
    Compute accuracy of the model
    Args:
        model: Trained model
        X: Input features
        y: True labels
    Returns:
        Accuracy as a percentage
    """
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y) * 100
    return accuracy


def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs, learning_rate, batch_size):
    """
    Train the model using gradient descent with validation-based model selection
    Args:
        model: Model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data for model selection
        X_test, y_test: Test data for final evaluation
        epochs: Number of training epochs
        learning_rate: Learning rate for gradient descent
        batch_size: Batch size for mini-batch gradient descent
    Returns:
        Tuple of (best_model, training_history, best_test_accuracy)
    """
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Learning rate: {learning_rate}, Batch size: {batch_size}")
    
    # Convert labels to one-hot encoding
    num_classes = len(np.unique(y_train))
    y_train_one_hot = one_hot_encode(y_train, num_classes)
    y_val_one_hot = one_hot_encode(y_val, num_classes)
    y_test_one_hot = one_hot_encode(y_test, num_classes)
    
    # Training history
    train_losses = []
    val_losses = []
    test_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    
    # Model selection variables
    best_val_accuracy = 0.0
    best_model_params = None
    best_epoch = 0
    
    # Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        # Mini-batch gradient descent
        for X_batch, y_batch in create_batches(X_train, y_train, batch_size):
            y_batch_one_hot = one_hot_encode(y_batch, num_classes)
            
            # Define loss function for this batch
            def batch_loss_fn(params):
                model.set_params(params)
                logits = model.forward(X_batch)
                return cross_entropy_loss(logits, y_batch_one_hot)
            
            # Get current parameters
            params = model.get_params()
            
            # Compute gradients
            grad_fn_batch = grad(batch_loss_fn)
            gradients = grad_fn_batch(params)
            
            # TODO: Update parameters using gradient descent
            #
            # Gradient descent updates parameters by moving in the opposite direction
            # of the gradient (steepest descent). This minimizes the loss function.
            #
            # Steps:
            # 1. Multiply gradients by learning rate: learning_rate * gradients
            # 2. Subtract from current parameters: params - (learning_rate * gradients)
            # 3. Store result in new_params variable
            #
            # Note: 'learning_rate' is passed as a parameter to this function
            
            raise NotImplementedError
            model.set_params(new_params)
            
            # Compute loss for recording with updated parameters
            logits = model.forward(X_batch)
            batch_loss = cross_entropy_loss(logits, y_batch_one_hot)
            epoch_loss += batch_loss
            num_batches += 1
            
        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)
        
        # Compute validation loss
        val_logits = model.forward(X_val)
        val_loss = cross_entropy_loss(val_logits, y_val_one_hot)
        val_losses.append(val_loss)
        
        # Compute test loss
        test_logits = model.forward(X_test)
        test_loss = cross_entropy_loss(test_logits, y_test_one_hot)
        test_losses.append(test_loss)
        
        # Compute accuracies
        train_acc = compute_accuracy(model, X_train, y_train)
        val_acc = compute_accuracy(model, X_val, y_val)
        test_acc = compute_accuracy(model, X_test, y_test)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)
        
        # Model selection based on validation accuracy
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_params = copy.deepcopy(model.get_params())
            best_epoch = epoch
        
        epoch_time = time.time() - epoch_start_time
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Test Loss: {test_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
                  f"Time: {epoch_time:.2f}s")
            if val_acc == best_val_accuracy:
                print(f"         *** New best validation accuracy! ***")
    
    # Load best model parameters
    model.set_params(best_model_params)
    best_test_accuracy = compute_accuracy(model, X_test, y_test)
    
    print(f"\nModel selection results:")
    print(f"  Best validation accuracy: {best_val_accuracy:.2f}% (epoch {best_epoch+1})")
    print(f"  Test accuracy of best model: {best_test_accuracy:.2f}%")
    
    # Training history dictionary
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'test_accuracies': test_accuracies,
        'best_epoch': best_epoch,
        'best_val_accuracy': best_val_accuracy
    }
    
    return model, training_history, best_test_accuracy


def plot_training_history(train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies, model_name, best_epoch=None):
    """
    Plot training history including validation curves
    Args:
        train_losses, val_losses, test_losses: Loss values over epochs
        train_accuracies, val_accuracies, test_accuracies: Accuracy values over epochs
        model_name: Name of the model for plot title
        best_epoch: Epoch with best validation performance (optional)
    """
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='green', linewidth=2)
    plt.plot(test_losses, label='Test Loss', color='red', linewidth=2)
    if best_epoch is not None:
        plt.axvline(x=best_epoch, color='black', linestyle='--', alpha=0.7, label=f'Best Val (Epoch {best_epoch+1})')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracies
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='blue', linewidth=2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green', linewidth=2)
    plt.plot(test_accuracies, label='Test Accuracy', color='red', linewidth=2)
    if best_epoch is not None:
        plt.axvline(x=best_epoch, color='black', linestyle='--', alpha=0.7, label=f'Best Val (Epoch {best_epoch+1})')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot validation vs test accuracy comparison
    plt.subplot(1, 3, 3)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green', linewidth=2)
    plt.plot(test_accuracies, label='Test Accuracy', color='red', linewidth=2)
    if best_epoch is not None:
        plt.axvline(x=best_epoch, color='black', linestyle='--', alpha=0.7, label=f'Best Val (Epoch {best_epoch+1})')
        plt.plot(best_epoch, val_accuracies[best_epoch], 'go', markersize=8, label=f'Selected Model')
    plt.title(f'{model_name} - Val vs Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train neural networks on Fashion-MNIST')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['linear', 'mlp', 'mlp3'],
                       help='Model type: linear, mlp (2-layer), or mlp3 (3-layer)')
    parser.add_argument('--ninput', type=int, default=784,
                       help='Number of input features (default: 784)')
    parser.add_argument('--nhidden', type=int, default=30,
                       help='Number of hidden units for MLP (default: 30)')
    parser.add_argument('--nhidden2', type=int, default=30,
                       help='Number of second hidden layer units for 3-layer MLP (default: 30)')
    parser.add_argument('--noutput', type=int, default=10,
                       help='Number of output classes (default: 10)')
    parser.add_argument('--residual', action='store_true',
                       help='Use residual connections for 3-layer MLP')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--train_file', type=str, default='fashion-mnist_train.csv.gz',
                       help='Path to training data file')
    parser.add_argument('--test_file', type=str, default='fashion-mnist_test.csv.gz',
                       help='Path to test data file')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory containing the data files (default: current directory)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Fraction of training data to use for validation (default: 0.2)')
    parser.add_argument('--plot', action='store_true',
                       help='Whether to plot training history figures')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*60)
    print("FASHION-MNIST NEURAL NETWORK TRAINING")
    print("="*60)
    print(f"Model: {args.model.upper()}")
    print(f"Input size: {args.ninput}")
    if args.model in ['mlp', 'mlp3']:
        print(f"Hidden size 1: {args.nhidden}")
    if args.model == 'mlp3':
        print(f"Hidden size 2: {args.nhidden2}")
        print(f"Residual connections: {args.residual}")
    print(f"Output size: {args.noutput}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Data directory: {args.data_dir}")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Random seed: {args.seed}")
    print("="*60)
    
    # Load data with full paths
    train_file_path = os.path.join(args.data_dir, args.train_file)
    test_file_path = os.path.join(args.data_dir, args.test_file)
    X_train_full, y_train_full, X_test, y_test = load_fashion_mnist_data(train_file_path, test_file_path)
    
    # Create validation split
    X_train, y_train, X_val, y_val = create_validation_split(
        X_train_full, y_train_full, args.val_ratio, args.seed
    )
    
    # Create model based on type
    if args.model == 'linear':
        model = LinearClassifier(args.ninput, args.noutput)
        model_name = "Linear Classifier"
    elif args.model == 'mlp':
        model = TwoLayerMLP(args.ninput, args.nhidden, args.noutput)
        model_name = "Two-Layer MLP"
    elif args.model == 'mlp3':
        model = ThreeLayerMLP(args.ninput, args.nhidden, args.nhidden2, args.noutput, args.residual)
        model_name = "Three-Layer MLP"
    
    print(f"\nCreated {model_name}")
    
    # Train the model
    start_time = time.time()
    model, training_history, best_test_accuracy = train_model(
        model, X_train, y_train, X_val, y_val, X_test, y_test, 
        args.epochs, args.lr, args.batch_size
    )
    training_time = time.time() - start_time
    
    # Final evaluation
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Final training loss: {training_history['train_losses'][-1]:.4f}")
    print(f"Final validation loss: {training_history['val_losses'][-1]:.4f}")
    print(f"Final test loss: {training_history['test_losses'][-1]:.4f}")
    print(f"Final training accuracy: {training_history['train_accuracies'][-1]:.2f}%")
    print(f"Best validation accuracy: {training_history['best_val_accuracy']:.2f}% (epoch {training_history['best_epoch']+1})")
    print(f"Test accuracy of best model: {best_test_accuracy:.2f}%")
    
    # Plot training history
    if args.plot:
        plot_training_history(
            training_history['train_losses'], training_history['val_losses'], training_history['test_losses'],
            training_history['train_accuracies'], training_history['val_accuracies'], training_history['test_accuracies'], 
            model_name, training_history['best_epoch']
        )
        plt.savefig(f"{model_name.lower().replace(' ', '_')}_training_history.png")
        plt.close()
    
        print(f"\nTraining history plot saved as '{model_name.lower().replace(' ', '_')}_training_history.png'")
    
    print("="*60)


if __name__ == "__main__":
    main() 
