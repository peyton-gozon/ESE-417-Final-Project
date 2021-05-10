import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from models.data_loader import DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

if __name__ == '__main__':
    # We learned from `exploring_data.ipynb` that PCA with 8 principal components is optimal.
    n_components = 8

    # Random state to allow for this to be deterministic.
    random_state = np.random.RandomState(42069)

    # Build the model pipeline
    model_pipeline = Pipeline(steps=[
        ('standardization', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=random_state)),
        ('classifier', RandomForestClassifier(random_state=random_state))
    ])

    dl = DataLoader('../data/winequality-red.csv', random_state=random_state)
    X_train, X_test, y_train, y_test = dl.train_test_split(test_prop=0.7)  # 20% test, 72% train, 8% validate.

    N_train, d = X_train.shape

    # Following the recommendation from the slides while accounting for 10% of the points being used as a validation set
    N_h = 9 * N_train // 100

    parameter_grid = {
        'classifier__n_estimators': np.ceil(np.linspace(100, 1000, 19), dtype=int),
        'classifier__max_depth': np.append(np.linspace(d**2, d**3, 9), None),
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__class_weight': ['balanced', 'balanced_subsample'],
    }

    # 5-fold validation.
    grid_search = GridSearchCV(model_pipeline, param_grid=parameter_grid, scoring='precision', n_jobs=-1, cv=3)

    # Train the model
    grid_search.fit(X_train, y_train)

    print("Best Estimator:")
    print("Parameters: " + str(grid_search.best_params_))
    print("Accuracy: " + str(grid_search.best_score_))

    y_test_pred = grid_search.predict(X_test)
    print("Classification report:")
    print(classification_report(y_test, y_test_pred))
