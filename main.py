#%% Load the data
import numpy as np
from sklearn.decomposition import PCA
from models.data_loader import DataLoader

#%% Make the data loader
dl = DataLoader("data/winequality-red.csv")

#%% Load the data
# Notes:
# * had to fix the csv document -- the headers had an error.
# * no null values in it.
X_train, X_test, y_train, y_test = dl.train_test_split()

#%% Once a pipeline exists, modify this file further.
