import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from models.data_loader import DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import time


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
        ('classifier', RandomForestClassifier(random_state=random_state))
    ])

    print("Loading the data...")
    dl = DataLoader('data/winequality-red.csv', random_state=random_state)
    # 20% test, 72% train, 8% validate
    X_train, X_test, y_train, y_test = dl.train_test_split(test_prop=(1.0-train_prop))

    N_train, _ = X_train.shape
    d = n_components

    parameter_grid = {
        'classifier__n_estimators': np.linspace(50, 500, 10, dtype=int),
        'classifier__max_depth': np.append(np.linspace(d**2, d*(d+3), 9), None),
        'classifier__criterion': ['entropy', 'gini'],
        'classifier__class_weight': ['balanced_subsample', 'balanced'],
        'classifier__max_features': [2, 3, 4, 'sqrt'],
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