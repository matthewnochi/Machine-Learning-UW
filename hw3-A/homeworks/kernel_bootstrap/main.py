from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 6 * np.sin(np.pi * x) * np.cos(4 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    result = np.outer(x_i, x_j)
    result += 1
    return np.power(result, d)


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    result = np.subtract.outer(x_i, x_j)
    result = result ** 2
    return np.exp(result * -gamma)
    


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """

    K = kernel_function(x, x, kernel_param)
    I = np.eye(K.shape[0])
    result = (K + _lambda * I)
    return np.linalg.solve(result, y)


@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across all folds.
    """
    fold_size = len(x) // num_folds
    loss = []

    for fold in range(num_folds):
        valx = x[fold * fold_size:(fold + 1) * fold_size]
        valy = y[fold * fold_size:(fold + 1) * fold_size]

        trainx = np.concatenate((x[:fold * fold_size], x[(fold + 1) * fold_size:]))
        trainy = np.concatenate((y[:fold * fold_size], y[(fold + 1) * fold_size:]))

        a_hat = train(trainx, trainy, kernel_function, kernel_param, _lambda)
        valK = kernel_function(trainx, valx, kernel_param)
        predictions = np.dot(valK.T, a_hat)

        loss.append(np.mean((valy - predictions) ** 2))

    return np.mean(loss)




@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be len(x) for LOO.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / (median(dist(x_i, x_j)^2) for all unique pairs x_i, x_j in x
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    median_dist_sq = np.median(np.sum((x[:, np.newaxis] - x) ** 2, axis=-1))
    gamma_values = 1 / np.linspace(median_dist_sq * 0.1, median_dist_sq * 10, num = 10) # Grid search
    lambdas = 10 ** np.linspace(-5, -1, num = 10) # Grid search

    best_lambda = None
    best_loss = float('inf')
    best_gamma = None

    for _lambda in lambdas: 
        for gamma in gamma_values:
            loss = cross_validation(x, y, rbf_kernel, gamma, _lambda, num_folds)

            if loss < best_loss:
                best_loss = loss
                best_lambda = _lambda
                best_gamma = gamma
    
    return best_lambda, best_gamma




@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution [5, 6, ..., 24, 25]
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [5, 6, ..., 24, 25]
    """

    lambdas = 10 ** np.linspace(-5, -1, num = 10) # Grid search
    ds = np.arange(5, 26)

    best_lambda = None
    best_d = None
    best_loss = float('inf')

    for _lambda in lambdas:
        for d in ds:
            loss = cross_validation(x, y, poly_kernel, d, _lambda, num_folds)

            if loss < best_loss:
                best_loss = loss
                best_lambda = _lambda
                best_d = d

    return best_lambda, best_d



@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
            Note that x_30, y_30 has been loaded in for you. You do not need to use (x_300, y_300) or (x_1000, y_1000).
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")

    lambda_rbf, gamma_rbf = rbf_param_search(x_30, y_30, num_folds=len(x_30))
    lambda_poly, d_poly = poly_param_search(x_30, y_30, num_folds=len(x_30))

    print(f"RBF Kernel: Best lambda = {lambda_rbf}, Best gamma = {gamma_rbf}")
    print(f"Poly Kernel: Best lambda = {lambda_poly}, Best d = {d_poly}")

    a_hat_rbf = train(x_30, y_30, rbf_kernel, gamma_rbf, lambda_rbf)
    a_hat_poly = train(x_30, y_30, poly_kernel, d_poly, lambda_poly)

    x_grid = np.linspace(0, 1, 100)
    
    K_rbf_grid = rbf_kernel(x_30, x_grid, gamma_rbf)
    f_rbf = np.dot(K_rbf_grid.T, a_hat_rbf)
    
    K_poly_grid = poly_kernel(x_30, x_grid, d_poly)
    f_poly = np.dot(K_poly_grid.T, a_hat_poly)
    
    true_function = f_true(x_grid)
    
    plt.figure(figsize=(12, 6))
    
    # Plot for Polynomial Kernel
    plt.subplot(1, 2, 1)
    plt.plot(x_30, y_30, 'o', label='Data')
    plt.plot(x_grid, true_function, label='True Function')
    plt.plot(x_grid, f_poly, label='Poly Prediction')
    plt.title("Polynomial Kernel")
    plt.ylim(-6, 6)
    plt.legend()

    # Plot for RBF Kernel
    plt.subplot(1, 2, 2)
    plt.plot(x_30, y_30, 'o', label='Data')
    plt.plot(x_grid, true_function, label='True Function')
    plt.plot(x_grid, f_rbf, label='RBF Prediction')
    plt.title("RBF Kernel")
    plt.ylim(-6, 6)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
