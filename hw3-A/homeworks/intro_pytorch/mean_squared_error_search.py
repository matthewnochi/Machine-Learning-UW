if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer
    from losses import MSELossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer
    from .optimizers import SGDOptimizer
    from .losses import MSELossLayer
    from .train import plot_model_guesses, train


from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def accuracy_score(model: nn.Module, dataloader: DataLoader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for MSE.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is also a 2-d vector of floats, but specifically with one being 1.0, while other is 0.0.
            Index of 1.0 in target corresponds to the true class.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to CrossEntropy accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
            total += labels.size(0)

    return correct / total


@problem.tag("hw3-A")
def mse_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the MSE problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers

    Notes:
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.

    Args:
        dataset_train (TensorDataset): Training dataset.
        dataset_val (TensorDataset): Validation dataset.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """

    models = {
        "Linear Regression": nn.Sequential(
            LinearLayer(2, 2)
        ),
        "NN One Hidden w/Sigmoid": nn.Sequential(
            LinearLayer(2, 2), 
            SigmoidLayer(),
            LinearLayer(2, 2)
        ),
        "NN One Hidden w/ReLU": nn.Sequential(
            LinearLayer(2, 2), 
            ReLULayer(),
            LinearLayer(2, 2) 
        ),
        "NN Two Hidden w/Sigmoid then ReLU": nn.Sequential(
            LinearLayer(2, 2),
            SigmoidLayer(),
            LinearLayer(2, 2),
            ReLULayer(),
            LinearLayer(2, 2)
        ),
        "NN Two Hidden w/ReLU then Sigmoid": nn.Sequential(
            LinearLayer(2, 2),
            ReLULayer(),
            LinearLayer(2, 2),
            SigmoidLayer(),
            LinearLayer(2, 2)
        ),
        "NN Two Hidden w/ReLU": nn.Sequential(
            LinearLayer(2, 2),
            ReLULayer(),
            LinearLayer(2, 2),
            ReLULayer(),
            LinearLayer(2, 2)
        )
    }

    results = {}

    for model_name, model in models.items():
        criterion = MSELossLayer()
        optimizer = SGDOptimizer(model.parameters(), lr=0.02)
        train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True, generator=RNG)
        val_loader = DataLoader(dataset_val, batch_size=16, shuffle=True, generator=RNG)

        history = train(train_loader, model, criterion, optimizer, val_loader, epochs=100)
        
        results[model_name] = {
            "train": history["train"],
            "val": history["val"],
            "model": model,
        }

    return results



@problem.tag("hw3-A", start_line=11)
def main():
    """
    Main function of the MSE problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me MSE loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(to_one_hot(y)))
    dataset_val = TensorDataset(
        torch.from_numpy(x_val).float(), torch.from_numpy(to_one_hot(y_val))
    )
    dataset_test = TensorDataset(
        torch.from_numpy(x_test).float(), torch.from_numpy(to_one_hot(y_test))
    )

    mse_configs = mse_parameter_search(dataset_train, dataset_val)
    plt.figure(figsize=(10, 6))
    for model_name, history in mse_configs.items():
        plt.plot(history["train"], label=f"{model_name} - Train")
        plt.plot(history["val"], label=f"{model_name} - Val")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Train and Validation MSE Loss for Different Models")
    plt.show()

    # Find best model (based on lowest validation loss)
    best_model_name = min(mse_configs, key=lambda x: min(mse_configs[x]["val"]))
    best_model = mse_configs[best_model_name]["model"]
    print(f"Best Model: {best_model_name}")

    # Report the accuracy of the best model on the test set
    accuracy = accuracy_score(best_model, DataLoader(dataset_test, batch_size=16, shuffle=True, generator=RNG))
    print(f"Accuracy of Best Model on Test Set: {accuracy:.4f}")

    # Plot the best model's predictions on the test set
    plot_model_guesses(DataLoader(dataset_test, batch_size=16, shuffle=True, generator=RNG), best_model, title=f"Predictions of {best_model_name} on Test Set")



def to_one_hot(a: np.ndarray) -> np.ndarray:
    """Helper function. Converts data from categorical to one-hot encoded.

    Args:
        a (np.ndarray): Input array of integers with shape (n,).

    Returns:
        np.ndarray: Array with shape (n, c), where c is maximal element of a.
            Each element of a, has a corresponding one-hot encoded vector of length c.
    """
    r = np.zeros((len(a), 2))
    r[np.arange(len(a)), a] = 1
    return r


if __name__ == "__main__":
    main()
