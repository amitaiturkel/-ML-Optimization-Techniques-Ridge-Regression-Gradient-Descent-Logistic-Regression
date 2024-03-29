import random
import numpy as np
import torch
from torch import nn
from helpers import read_data_demo, cord_and_label,plot_decision_boundaries



random.seed(42)
torch.manual_seed(42)
class Ridge_Regression:

    def __init__(self, lambd):
        self.lambd = lambd
        self.weights = None

    def fit(self, X, Y):
        """
        Fit the ridge regression model to the provided data.
        :param X: The training features.
        :param Y: The training labels.
        """
        Y = 2 * (Y - 0.5)  # transform the labels to -1 and 1, instead of 0 and 1.

        N_train = X.shape[0]
        D = X.shape[1]

        XT_X = np.dot(X.T, X)
        lambda_I = self.lambd * np.identity(D) #λI

        XT_X_plus_lambda_I = XT_X/N_train + lambda_I  #  (XT*X/N_train + λI)
        XT_X_plus_lambda_I_inv = np.linalg.inv(XT_X_plus_lambda_I)  #  (XT*X + λI)-1
        X_T_Y = np.dot(X.T, Y)  # dot product of X transpose and Y
        self.weights = XT_X_plus_lambda_I_inv.dot(X_T_Y) / N_train  


    def predict(self, X):
        """
        Predict the output for the provided data.
        :param X: The data to predict. np.ndarray of shape (N, D).
        :return: The predicted output. np.ndarray of shape (N,), of 0s and 1s.
        """
        preds = np.dot(X, self.weights)

        # Classify predictions based on the sign of the dot product
        preds = np.where(preds >= 0, 1, -1)

        # Transform the labels to 0s and 1s
        preds = (preds + 1) / 2

        return preds



class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Logistic_Regression, self).__init__()


        self.linear = nn.Linear(input_dim, output_dim)

        

    def forward(self, x):
        """
        Computes the output of the linear operator.
        :param x: The input to the linear operator.
        :return: The transformed input.
        """
        # compute the output of the linear operator

        ########## YOUR CODE HERE ##########

        # return the transformed input.
        # first perform the linear operation
        # should be a single line of code.

        ####################################

        return self.linear(x)

    def predict(self, x):
        """
        THIS FUNCTION IS NOT NEEDED FOR PYTORCH. JUST FOR OUR VISUALIZATION
        """
        x = torch.from_numpy(x).float().to(self.linear.weight.data.device)
        x = self.forward(x)
        x = nn.functional.softmax(x, dim=1)
        x = x.detach().cpu().numpy()
        x = np.argmax(x, axis=1)
        return x


class DummyDataset(torch.utils.data.Dataset):
    """
    Any dataset should inherit from torch.utils.data.Dataset and override the __len__ and __getitem__ methods.
    __init__ is optional.
    __len__ should return the size of the dataset.
    __getitem__ should return a tuple (data, label) for the given index.
    """

    def __init__(self,path):
        
        x_array, y_array = cord_and_label(path)

        self.data = torch.tensor(x_array, dtype=torch.float32)
        self.labels = torch.tensor(y_array, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

