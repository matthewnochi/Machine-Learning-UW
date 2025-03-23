if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    X_train = df_train.drop("ViolentCrimesPerPop", axis=1).values
    y_train = df_train["ViolentCrimesPerPop"].values

    X_test = df_test.drop("ViolentCrimesPerPop", axis=1).values
    y_test = df_test["ViolentCrimesPerPop"].values

    # variables for A6d
    variables = ["agePct12t29", "pctWSocSec", "pctUrban", "agePct65up", "householdsize"]
    y_centered = y_train - np.mean(y_train)
    lambda_max = np.max(2 * np.abs(np.dot(np.transpose(X_train), y_centered)))
    lambdas = []
    nonzeros = []
    
    coef_paths = {var: [] for var in variables}

    train_errors = []
    test_errors = []

    curr_lambda = lambda_max
    while curr_lambda > 0.01:
        weight, bias = train(X_train, y_train, _lambda=curr_lambda, eta=0.00001)

        nonzeros.append(np.count_nonzero(weight))
        lambdas.append(curr_lambda)
        
        # variables for A6d
        for var in variables:
            index = df_train.columns.get_loc(var)
            coef_paths[var].append(weight[index])
            
        # error for A6e
        train_errors.append(np.mean((np.dot(X_train, weight) - y_train) ** 2))
        test_errors.append(np.mean((np.dot(X_test, weight) - y_test) ** 2))

        curr_lambda *= 0.5
    
    # A6c
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, nonzeros, marker='o')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Number of Non-Zero Weights')
    plt.title('Number of Non-Zero Weights vs Lambda')
    plt.grid(True)
    plt.show()

    # A6d
    plt.figure(figsize=(10, 6))
    for var in variables:
        plt.plot(lambdas, coef_paths[var], marker='o', label=var)
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Coefficient')
    plt.title('Regularization Paths')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    # A6e
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, train_errors, label='Training Error', marker='o')
    plt.plot(lambdas, test_errors, label='Test Error', marker='o')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Squared Error')
    plt.title('Squared Error vs Lambda')
    plt.legend()
    plt.grid(True)
    plt.show()

    # A6f
    weight, bias = train(X_train, y_train, _lambda=30, eta=0.00001)
    largest_weight = np.max(weight)
    smallest_weight = np.min(weight)
    largest_feature = df_train.columns[np.argmax(weight)]
    smallest_feature = df_train.columns[np.argmin(weight)]
    print(f"Largest coefficient: {largest_feature} with value {largest_weight}")
    print(f"Smallest coefficient: {smallest_feature} with value {smallest_weight}")

if __name__ == "__main__":
    main()
