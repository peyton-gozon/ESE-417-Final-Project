import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import copy


class DataLoader:
    """ Allows for the easy wrangling of the dataset. """

    def __init__(self, source, random_state=None):
        self.data = pd.read_csv(source, sep=';')
        self.random_state = random_state if random_state is not None else np.random.RandomState(42069)

    def train_test_split(self, train_prop=0.3, random_state=None):
        rs = random_state if random_state is not None else self.random_state
        data_copy = copy.copy(self.data)

        y = data_copy['quality']  # The model output is the quality of the wine.
        del data_copy['quality']  # Remove the ground truths from the dataset.

        # the test_train_split function only works on NumPy arrays, and not pandas
        X = data_copy.to_numpy()

        # Perform a train/test split
        X_test, X_train, y_test, y_train = train_test_split(X, y, random_state=rs, test_size=train_prop)

        return X_test, X_train, y_test, y_train

    def get_all_data(self, separate_ground_truths=True):
        """ Returns the entire design matrix and ground truths. """

        data = self.data.to_numpy()

        if separate_ground_truths:
            X = data[:, :-1]
            y = data[:, -1]

            return X, y
        else:
            return data
