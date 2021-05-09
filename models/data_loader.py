import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class DataLoader:
    """ Allows for the easy and deterministic wrangling of the dataset. """

    def __init__(self, source, random_state=None):
        self.random_state = random_state if random_state is not None else np.random.RandomState(42069)

        # Read in the data from the csv file and convert it into an X and y matrices
        data = pd.read_csv(source, sep=';')

        self.y = data['quality'].to_numpy()  # The model output is the quality of the wine.
        del data['quality']  # Remove the ground truths from the dataset.

        self.X = data.to_numpy()

    def train_test_split(self, test_prop=0.3):
        # Perform a train/test split
        X_test, X_train, y_test, y_train = train_test_split(self.X, self.y, random_state=self.random_state,
                                                            test_size=test_prop)
        return X_test, X_train, y_test, y_train

    def get_all_data(self):
        """ Returns the design matrix and ground truths. """
        return self.X, self.y

    def apply_pca_to_dataset(self, n_components=8):
        """
        Apply what we've learned from the exploratory data analysis done in `exploring_data.ipynb`. The changes occur
        in place.

        1. Standardize the design matrix.
        2. Reduce the data down to 8 principal components.
        """
        standard_scaler = StandardScaler()
        X_standardized = standard_scaler.fit_transform(self.X)

        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X_standardized)

        self.X = X_pca
