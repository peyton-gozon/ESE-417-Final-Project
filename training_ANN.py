import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from models.data_loader import DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import time
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

if __name__ == '__main__':
    start_time = time.time()

    # We learned from `exploring_data.ipynb` that PCA with 8 principal components is optimal.
    n_components = 8
    train_prop = 0.8

    # Random state to allow for this to be deterministic.
    random_state = np.random.RandomState(42069)

    # Build the model pipeline
    print("Creating the model pipeline...")
    model_pipeline = Pipeline(steps=[
        ('standardization', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=random_state)),
        ('classifier', MLPClassifier(
            random_state=random_state, max_iter=2000,
        ))
    ])

    print("Loading the data...")
    dl = DataLoader('data/winequality-red.csv', random_state=random_state)
    # 20% test, 72% train, 8% validate.
    X_train, X_test, y_train, y_test = dl.train_test_split(test_prop=(1 - train_prop))

    N_train, _ = X_train.shape
    d = n_components

    # The number of hidden nodes
    N_h = N_train // 10

    print(model_pipeline.get_params().keys())

    parameter_grid = {
        # Consider 1, 2, 3, and 4-layer neural networks configurations. However, these all contain equal amounts of
        # nodes per layer.
        'classifier__hidden_layer_sizes': [
            (N_h,), (N_h // 2, N_h // 2,), (N_h // 3, N_h // 3, N_h // 3,),
        ],
        'classifier__solver': ['adam'],
        # Regularization term
        'classifier__alpha': np.logspace(-5, -1, 5),
        # Whether to use early stopping or not to combat over-training.
        'classifier__learning_rate_init': [0.01, 0.1, 1],
        'classifier__early_stopping': [True, False],
        'classifier__validation_fraction': [0.05, 0.1, 0.2],
        'classifier__activation': ['logistic', 'relu'],
        'classifier__batch_size': [50, 100, 150, 200],
        'classifier__beta_1': [0.8, 0.9]
    }

    # 5-fold validation.
    print("Defining the grid search...")
    grid_search = GridSearchCV(model_pipeline, param_grid=parameter_grid, scoring='f1_weighted', n_jobs=-2, cv=5)

    # Train the model
    print("Training the model...")
    grid_search.fit(X_train, y_train)

    print("Printing results...")
    print(f"Time taken to run the program: {time.time() - start_time}")

    print("Best Estimator:")
    print("Parameters: " + str(grid_search.best_params_))
    print("Accuracy: " + str(grid_search.best_score_))

    y_test_pred = grid_search.predict(X_test)
    print("Classification report (Test):")
    print(classification_report(y_test, y_test_pred))

    unique_labels = np.unique(y_test)

    confusion_mtrx = pd.DataFrame(
        confusion_matrix(y_test, y_test_pred, labels=unique_labels),
        index=[f'true:{i}' for i in unique_labels],
        columns=[f'pred:{i}' for i in unique_labels]
    )

    print(confusion_mtrx)
