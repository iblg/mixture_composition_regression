# import basic packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
import os

# Section 1: Load composition data
compositionData = pd.read_csv('/Users/Columbia_Kartik Chandran/Lignocellulose/WRF+lignocellulose/Training set/composition.csv')

# Section 2: Load IR spectra data
spectraData = pd.read_csv('/Users/Columbia_Kartik Chandran/Lignocellulose/WRF+lignocellulose/Training set/all_spectra.csv')

# Extract 'Sample' column for common identifier
commonIdentifier = compositionData.columns.tolist()

# Extract predictors (X) and response variable (y)
X = spectraData.iloc[:, 1:].values
X_transposed = np.transpose(X)
# Extract response variable (y) from 'composition.csv'
y_cellulose = compositionData.iloc[0, 1:].values
y_cellulose_transposed = np.transpose(y_cellulose)
y_hemicellulose = compositionData.iloc[1, 1:].values
y_hemicellulose_transposed = np.transpose(y_hemicellulose)
y_lignin = compositionData.iloc[2, 1:].values
y_lignin_transposed = np.transpose(y_lignin)
y_fungi = compositionData.iloc[3, 1:].values
y_fungi_transposed = np.transpose(y_fungi)


# Split the data into training, testing, and cross-validation sets
cv_ratio = 0.1  # 10% of data for cross-validation
test_ratio = 0.2  # 20% of data for testing

X_train, X_remaining, y_train, y_remaining = train_test_split(X_transposed, y_cellulose_transposed, test_size=(cv_ratio + test_ratio), random_state=123)
X_cv, X_test, y_cv, y_test = train_test_split(X_remaining, y_remaining, test_size=test_ratio / (cv_ratio + test_ratio), random_state=123)

# Define parameter grids for each model
param_grid_lr = {
    'fit_intercept': [True, False],
}

param_grid_svr = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}

param_grid_knnr = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'max_iter': [50, 100, 200]
}

param_grid_pls = {
    'n_components': [2, 5, 10]
}

param_grid_dtr = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion': ['mse', 'friedman_mse', 'mae']
}

# Train multiple models and evaluate their performance
models = []  # Placeholder for different models
mse_values = []  # Placeholder for MSE values


# Model 1: Linear Regression (LR)
grid_search_lr = GridSearchCV(LinearRegression(), param_grid_lr, cv=5)
grid_search_lr.fit(X_train, y_train)
best_lr = grid_search_lr.best_estimator_
models.append(best_lr)
mse_values.append(mean_squared_error(y_cv, best_lr.predict(X_cv)))

# Model 2: Support Vector Regression (SVR)
grid_search_svr = GridSearchCV(SVR(), param_grid_svr, cv=5)
grid_search_svr.fit(X_train, y_train)
best_svr = grid_search_svr.best_estimator_
models.append(best_svr)
mse_values.append(mean_squared_error(y_cv, best_svr.predict(X_cv)))

# Model 3: KNN Regressor (KNNR)
grid_search_knnr = GridSearchCV(KNeighborsRegressor(), param_grid_knnr, cv=5)
grid_search_knnr.fit(X_train, y_train)
best_knnr = grid_search_knnr.best_estimator_
models.append(best_knnr)
mse_values.append(mean_squared_error(y_cv, best_knnr.predict(X_cv)))

# Model 4: Multi-layer Perceptron (MLP)
grid_search_mlp = GridSearchCV(MLPRegressor(), param_grid_mlp, cv=5)
grid_search_mlp.fit(X_train, y_train)
best_mlp = grid_search_mlp.best_estimator_
models.append(best_mlp)
mse_values.append(mean_squared_error(y_cv, best_mlp.predict(X_cv)))

# Model 5: Partial Least Squares (PLS)
grid_search_pls = GridSearchCV(PLSRegression(), param_grid_pls, cv=5)
grid_search_pls.fit(X_train, y_train)
best_pls = grid_search_pls.best_estimator_
models.append(best_pls)
mse_values.append(mean_squared_error(y_cv, best_pls.predict(X_cv)))

# Model 6: Decision Tree Regressor (DTR)
grid_search_dtr = GridSearchCV(DecisionTreeRegressor(), param_grid_dtr, cv=5)
grid_search_dtr.fit(X_train, y_train)
best_dtr = grid_search_dtr.best_estimator_
models.append(best_dtr)
mse_values.append(mean_squared_error(y_cv, best_dtr.predict(X_cv)))


# Find the best model based on MSE
best_model_index = np.argmin(mse_values)
best_model = models[best_model_index]

# Make predictions on test data using the best model
predicted_values = best_model.predict(X_test)

# Evaluate the performance of the best model
mse_best_model = mean_squared_error(y_test, predicted_values)

print("Mean Squared Error of the best model:", mse_best_model)
print(best_model)
