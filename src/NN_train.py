import numpy as np
from src.NN_model import (
    initialize_parameters,
    forward_propagation,
    backward_propagation,
    update_parameters,
    cross_entropy_loss,
)


def one_hot_encode(y, n_classes=35):
    """
    Converts integer class labels into one-hot encoded columns -- all
    zeros except a single 1 at the true class position. This is the
    Y format cross-entropy's dZ2 = A2 - Y shortcut requires.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        Integer class labels.
    n_classes : int
        Total number of classes.

    Returns
    -------
    np.ndarray, shape (n_classes, n_samples)
        One-hot encoded labels, one column per sample.
    """
    Y = np.zeros((n_classes, y.shape[0]))
    Y[y, np.arange(y.shape[0])] = 1
    return Y


def get_mini_batches(X, y, batch_size, rng):
    """
    Shuffles the dataset and splits it into mini-batches for one epoch.
    Shuffling each epoch prevents the network from learning any
    accidental ordering in the stored data.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data, rows are samples (as stored in the .npz file).
    y : np.ndarray, shape (n_samples,)
        Integer class labels.
    batch_size : int
        Number of samples per mini-batch.
    rng : np.random.Generator
        Seeded random generator, for reproducible shuffling.

    Yields
    ------
    X_batch : np.ndarray, shape (n_features, batch_size)
        Transposed so it's ready to feed directly into
        forward_propagation (features as rows, samples as columns).
    y_batch : np.ndarray, shape (batch_size,)
        Integer class labels for this batch.
    """
    n_samples = X.shape[0]
    perm = rng.permutation(n_samples)
    X_shuffled, y_shuffled = X[perm], y[perm]
    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        yield X_shuffled[start:end].T, y_shuffled[start:end]


def compute_accuracy(A2, y):
    """
    Computes the fraction of samples where the network's highest-
    probability class matches the true label.

    Parameters
    ----------
    A2 : np.ndarray, shape (n_classes, n_samples)
        Predicted probabilities from forward_propagation.
    y : np.ndarray, shape (n_samples,)
        True integer class labels.

    Returns
    -------
    float
        Accuracy, between 0 and 1.
    """
    return np.mean(np.argmax(A2, axis=0) == y)


def train(X_train, y_train, X_val, y_val, layer_dims,
          epochs=30, batch_size=64, learning_rate=0.05, seed=1):
    """
    Trains the network with mini-batch gradient descent, running
    forward -> loss -> backward -> update once per mini-batch, and
    evaluating on the validation set after every epoch.

    Tracks the parameters that achieved the best validation accuracy
    seen during training, and returns those instead of whatever the
    final epoch happened to land on -- validation accuracy can swing
    between epochs, so the last epoch isn't necessarily the best one.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data, rows are samples.
    X_val, y_val : np.ndarray
        Validation data, same format as training data.
    layer_dims : list of int
        Network shape, e.g. [784, 128, 35].
    epochs : int
        Number of full passes over the training set.
    batch_size : int
        Number of samples per mini-batch update.
    learning_rate : float
        Step size for gradient descent.
    seed : int
        Random seed for reproducible initialization and shuffling.

    Returns
    -------
    best_params : dict
        Weights and biases (W1, b1, W2, b2) from the epoch with the
        highest validation accuracy seen during training.
    history : dict
        Lists of train_loss, val_loss, and val_accuracy, one entry
        per epoch, for plotting learning curves.
    """
    rng = np.random.default_rng(seed)
    params = initialize_parameters(layer_dims, seed=seed)
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    n_classes = layer_dims[-1]

    best_val_acc = -1
    best_params = None

    X_val_T = X_val.T
    for epoch in range(epochs):
        batch_losses = []
        for X_batch, y_batch in get_mini_batches(X_train, y_train, batch_size, rng):
            Y_batch = one_hot_encode(y_batch, n_classes)
            A2, cache = forward_propagation(X_batch, params)
            batch_losses.append(cross_entropy_loss(A2, y_batch))
            grads = backward_propagation(X_batch, Y_batch, params, cache)
            params = update_parameters(params, grads, learning_rate)

        A2_val, _ = forward_propagation(X_val_T, params)
        history['train_loss'].append(np.mean(batch_losses))
        history['val_loss'].append(cross_entropy_loss(A2_val, y_val))
        history['val_accuracy'].append(compute_accuracy(A2_val, y_val))

        if history['val_accuracy'][-1] > best_val_acc:
            best_val_acc = history['val_accuracy'][-1]
            best_params = {k: v.copy() for k, v in params.items()}

        print(f"Epoch {epoch+1:2d}/{epochs} — train_loss: {history['train_loss'][-1]:.4f}  "
              f"val_loss: {history['val_loss'][-1]:.4f}  val_acc: {history['val_accuracy'][-1]:.4f}")

    return best_params, history