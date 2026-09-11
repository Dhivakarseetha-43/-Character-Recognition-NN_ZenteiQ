import numpy as np


def initialize_parameters(layer_dims, seed=1):
    """
    Initializes weights and biases for a fully-connected network.

    Uses He initialization (std = sqrt(2/n_inputs)) for layers followed
    by ReLU, since ReLU zeroes out ~half of activations and He init
    compensates by starting with more variance. Uses Xavier initialization
    (std = sqrt(1/n_inputs)) for the final layer feeding into Softmax.

    Biases are initialized to zero -- this is safe (unlike weights) because
    biases don't cause the symmetry problem: each neuron's bias only
    affects that one neuron, it isn't shared/mirrored across neurons the
    way a constant weight matrix would be.

    Parameters
    ----------
    layer_dims : list of int
        Sizes of each layer including input and output, e.g. [784, 128, 35].
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys 'W1', 'b1', 'W2', 'b2', ... containing the initialized
        weight matrices and bias vectors for each layer.
    """
    rng = np.random.default_rng(seed)
    params = {}
    L = len(layer_dims) - 1  # number of layers with weights

    for l in range(1, L + 1):
        n_in = layer_dims[l - 1]
        n_out = layer_dims[l]

        if l < L:  # hidden layer -> He init (ReLU follows)
            std = np.sqrt(2.0 / n_in)
        else:      # output layer -> Xavier init (Softmax follows)
            std = np.sqrt(1.0 / n_in)

        params[f'W{l}'] = rng.normal(0, std, size=(n_out, n_in))
        params[f'b{l}'] = np.zeros((n_out, 1))

    return params


def relu(Z):
    """
    Applies the ReLU (Rectified Linear Unit) activation elementwise.

    ReLU keeps positive values unchanged and zeroes out negative ones.
    This is what introduces non-linearity into the network -- without
    it, stacking layers would collapse into a single linear function.

    Parameters
    ----------
    Z : np.ndarray
        Pre-activation values, any shape.

    Returns
    -------
    np.ndarray
        Same shape as Z, with all negative values replaced by 0.
    """
    return np.maximum(0, Z)


def relu_derivative(Z):
    """
    Computes the derivative of ReLU, elementwise.

    ReLU's derivative is 1 wherever the original input was positive
    (that neuron was active and passed a value through), and 0 wherever
    it was negative or zero (that neuron was inactive, so it can't have
    contributed to the forward pass -- and shouldn't receive any blame
    during backprop).

    Parameters
    ----------
    Z : np.ndarray
        The pre-activation values from the forward pass (before ReLU
        was applied) -- NOT the activated output.

    Returns
    -------
    np.ndarray, same shape as Z
        1 where Z > 0, else 0.
    """
    return (Z > 0).astype(float)


def Softmax(Z):
    """
    Converts raw output scores into a probability distribution over classes.

    Exponentiates each score, then normalizes by the sum of all
    exponentials so outputs form a valid probability distribution
    (values in [0,1], each column summing to 1). Subtracts the max
    value per column before exponentiating for numerical stability --
    this doesn't change the result mathematically (it cancels out in
    the division) but avoids overflow from large raw scores.

    Parameters
    ----------
    Z : np.ndarray, shape (n_classes, n_samples)
        Raw output scores, one column per sample.

    Returns
    -------
    np.ndarray, same shape as Z
        Probabilities, each column summing to 1.
    """
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


def forward_propagation(X, params):
    """
    Runs a full forward pass through the network: input -> hidden layer
    (linear + ReLU) -> output layer (linear + Softmax).

    Parameters
    ----------
    X : np.ndarray, shape (n_features, n_samples)
        Input data, one column per sample.
    params : dict
        Weight matrices and bias vectors: W1, b1, W2, b2.

    Returns
    -------
    A2 : np.ndarray, shape (n_classes, n_samples)
        Final predicted probabilities for each sample.
    cache : dict
        Intermediate values (Z1, A1, Z2, A2) saved for use in backprop.
    """
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']

    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = Softmax(Z2)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache


def cross_entropy_loss(A2, y):
    """
    Computes the average cross-entropy loss across a batch of samples.

    For each sample, looks up the probability the network assigned to
    the true class, then computes -log(that probability). Averages
    this across all samples in the batch. Loss is low when the network
    assigns high probability to the correct class, and grows sharply
    as that probability approaches zero -- penalizing confident wrong
    answers far more than mild ones.

    Parameters
    ----------
    A2 : np.ndarray, shape (n_classes, n_samples)
        Predicted probabilities from forward_propagation.
    y : np.ndarray, shape (n_samples,)
        True integer class labels.

    Returns
    -------
    float
        Average loss across the batch.
    """
    n_samples = y.shape[0]
    correct_class_probs = A2[y, np.arange(n_samples)]
    # Tiny epsilon avoids log(0) = -infinity if a probability ever underflows to exactly 0
    loss = -np.mean(np.log(correct_class_probs + 1e-9))
    return loss


def backward_propagation(X, Y, params, cache):
    """
    Computes gradients of the loss with respect to every weight and
    bias, by propagating the error backward through the network --
    output layer first, then hidden layer -- using the chain rule.

    Parameters
    ----------
    X : np.ndarray, shape (n_features, n_samples)
        The original input batch (needed to compute dW1).
    Y : np.ndarray, shape (n_classes, n_samples)
        One-hot encoded true labels.
    params : dict
        Current W1, b1, W2, b2.
    cache : dict
        Z1, A1, Z2, A2 saved from forward_propagation.

    Returns
    -------
    dict
        Gradients dW1, db1, dW2, db2 -- same shapes as the
        corresponding parameters, ready to be used in a weight update.
    """
    n_samples = X.shape[1]
    W2 = params['W2']
    Z1, A1 = cache['Z1'], cache['A1']
    A2 = cache['A2']

    # Output layer
    dZ2 = A2 - Y
    dW2 = (dZ2 @ A1.T) / n_samples
    db2 = np.sum(dZ2, axis=1, keepdims=True) / n_samples

    # Step backward into the hidden layer
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * relu_derivative(Z1)

    # Hidden layer
    dW1 = (dZ1 @ X.T) / n_samples
    db1 = np.sum(dZ1, axis=1, keepdims=True) / n_samples

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}


def update_parameters(params, grads, learning_rate=0.1):
    """
    Applies one step of gradient descent: nudges every weight and bias
    in the direction that reduces loss, scaled by the learning rate.

    Parameters
    ----------
    params : dict
        Current W1, b1, W2, b2.
    grads : dict
        Gradients dW1, db1, dW2, db2 from backward_propagation.
    learning_rate : float
        Step size for each update. Larger values learn faster but risk
        overshooting; smaller values are more stable but slower.

    Returns
    -------
    dict
        Updated W1, b1, W2, b2.
    """
    params['W1'] -= learning_rate * grads['dW1']
    params['b1'] -= learning_rate * grads['db1']
    params['W2'] -= learning_rate * grads['dW2']
    params['b2'] -= learning_rate * grads['db2']
    return params