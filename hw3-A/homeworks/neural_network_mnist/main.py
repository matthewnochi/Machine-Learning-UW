# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.w0 = torch.nn.Parameter(torch.rand(d, h))
        self.w1 = torch.nn.Parameter(torch.rand(h, k))

        self.b0 = torch.nn.Parameter(torch.zeros(h))
        self.b1 = torch.nn.Parameter(torch.zeros(k))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        result = x @ self.w0 + self.b0
        result = torch.nn.functional.relu(result)
        return result @ self.w1 + self.b1


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.w0 = torch.nn.Parameter(torch.rand(d, h0))
        self.w1 = torch.nn.Parameter(torch.rand(h0, h1))
        self.w2 = torch.nn.Parameter(torch.rand(h1, k))

        self.b0 = torch.nn.Parameter(torch.zeros(h0))
        self.b1 = torch.nn.Parameter(torch.zeros(h1))
        self.b2 = torch.nn.Parameter(torch.zeros(k))


    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        result = x @ self.w0 + self.b0
        result = torch.nn.functional.relu(result)

        result = result @ self.w1 + self.b1
        result = torch.nn.functional.relu(result)
        
        return result @ self.w2 + self.b2



@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    model.train()
    losses = []

    for epoch in range(1, 101):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)

        epoch_accuracy = 100 * correct / total        
        if epoch_accuracy >= 99:
            break
    
    return losses



@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    train_dataset = TensorDataset(x, y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    f1_model = F1(h=64, d=784, k=10)
    f1_optimizer = torch.optim.Adam(f1_model.parameters(), lr=0.005)

    f1_losses = train(f1_model, f1_optimizer, train_loader)

    plt.figure()
    plt.plot(f1_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss for F1')
    plt.show()

    f1_model.eval()
    correct = 0
    total = 0
    f1_test_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = f1_model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            f1_test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    f1_test_accuracy = 100 * correct / total
    f1_test_loss /= len(test_loader)

    print(f"F1 Model Test Accuracy: {f1_test_accuracy:.2f}%, Test Loss: {f1_test_loss:.4f}")

    f1_params = sum(p.numel() for p in f1_model.parameters())
    print(f"F1 Model Total Parameters: {f1_params}")

    f2_model = F2(h0=32, h1=32, d=784, k=10)
    f2_optimizer = torch.optim.Adam(f2_model.parameters(), lr=0.005)

    f2_losses = train(f2_model, f2_optimizer, train_loader)

    plt.figure()
    plt.plot(f2_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss for F2')
    plt.ylim(0, 4) # first epoch loss is is ~20
    plt.show()
    
    f2_model.eval()
    correct = 0
    total = 0
    f2_test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = f2_model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            f2_test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    f2_test_accuracy = 100 * correct / total
    f2_test_loss /= len(test_loader)

    print(f"F2 Model Test Accuracy: {f2_test_accuracy:.2f}%, Test Loss: {f2_test_loss:.4f}")

    f2_params = sum(p.numel() for p in f2_model.parameters())
    print(f"F2 Model Total Parameters: {f2_params}")

if __name__ == "__main__":
    main()
