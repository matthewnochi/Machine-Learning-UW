from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
    
    """
    xTwb = np.dot(X, weight) + bias
    gradb = np.sum(xTwb - y)
    bias -= 2 * eta * gradb

    gradw = np.dot(np.transpose(X), (xTwb - y))
    weight -= 2 * eta * gradw
    for k in range(0, len(weight)):
        if weight[k] < (-2 * eta * _lambda):
            weight[k] += (2 * eta * _lambda)
        elif weight[k] > (2 * eta * _lambda):
            weight[k] -= (2 * eta * _lambda)
        else:
            weight[k] = 0

    return weight, bias

@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized SSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    predict = np.dot(X, weight) + bias
    SSE = np.sum(np.power(predict - y, 2))
    l1 = _lambda * np.sum(np.abs(weight))

    return SSE + l1

@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.00001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (float, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You will also have to keep an old copy of bias for convergence criterion function.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
        start_bias = 0
    old_w: Optional[np.ndarray] = None
    old_b: float = None
    
    old_w = np.copy(start_weight)
    old_b = start_bias
    weight, bias = step(X, y, start_weight, start_bias, _lambda, eta)

    while not convergence_criterion(weight, old_w, bias, old_b, convergence_delta):
        old_w = weight
        old_b = bias
        weight, bias = step(X, y, weight, bias, _lambda, eta)

    return weight, bias 



@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight and bias has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compare it to convergence delta.
    It should also calculate the maximum absolute change between the bias and old_b, and compare it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of gradient descent.
        old_w (np.ndarray): Weight from previous iteration of gradient descent.
        bias (float): Bias from current iteration of gradient descent.
        old_b (float): Bias from previous iteration of gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight and bias has not converged yet. True otherwise.
    """
    change_weight = np.max(np.abs(weight - old_w))
    change_bias = np.max(np.abs(bias - old_b))
    return (change_weight < convergence_delta and change_bias < convergence_delta)


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    n = 500
    d = 1000
    k = 100
    sigma = 1

    X = np.random.randn(n, d)
    w = np.zeros(d)
    w[:k] = np.arange(1, k + 1) / k
    epsilon = np.random.normal(0, sigma, n)
    y = np.dot(X, w) + epsilon

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    lambda_max = np.max(np.abs(np.dot(np.transpose(X), y - np.mean(y)))) / n

    lambdas = []
    nonzeros = []
    fdr = []
    tpr = []

    curr_lambda = lambda_max
    while curr_lambda > 0.0000001:
        weight, bias = train(X, y, _lambda=curr_lambda, eta=0.00001)
        nonzeros.append(np.count_nonzero(weight))
        lambdas.append(curr_lambda)

        model_nonzero = weight != 0
        w_nonzero = w != 0

        FP = np.sum(model_nonzero & ~w_nonzero)
        fdr.append(FP / np.sum(model_nonzero) if np.sum(model_nonzero) > 0 else 0)

        TP = np.sum(model_nonzero & w_nonzero)
        tpr.append(TP / k)

        curr_lambda *= 0.5
    
    # A5 a- 
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, nonzeros, marker='o')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Number of Non-Zero Weights')
    plt.title('Number of Non-Zero Weights vs Lambda')
    plt.grid(True)
    plt.show()
    
    # A5 b-
    plt.figure(figsize=(10, 6))
    plt.plot(fdr, tpr, marker='o')
    plt.xlabel('False Discovery Rate')
    plt.ylabel('True Positive Rate')
    plt.title('FDR vs TPR')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
