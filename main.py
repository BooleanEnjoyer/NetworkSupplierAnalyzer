from dataReader import get_data
from dataScaler import scale_data
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

X, y = get_data()
X_scaled = scale_data(X)

models = [
    LinearRegression()
    ,DecisionTreeRegressor()
    ,Ridge()
    ,MLPRegressor()
    ,RandomForestRegressor()
]
models_names = [
    "LinearRegression"
    ,"DecisionTreeRegressor"
    ,"Ridge"
    ,"MLPRegressor"
    ,"RandomForestRegressor"
]
models_size = len(models)

metrics_array = [mean_squared_error, mean_absolute_error]
metrics_names = [
    "mean_squared_error",
    "mean_absolute_error"
]
metrics_size = len(metrics_array)

num_folds = 5
num_repeats = 3
rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=42)
elements_amount = num_folds * num_repeats

cv_scores = np.zeros((models_size, metrics_size, elements_amount))

current_value_index = 0
for model_index, model in enumerate(models):
    current_value_index = 0
    for train_index, test_index in rkf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        for metric_index, metric_function in enumerate(metrics_array):
            metric_value = metric_function(y_test, y_pred)
            cv_scores[model_index, metric_index, current_value_index] = metric_value
        current_value_index += 1


for model_index, model in enumerate(models):
    for metric_index, metric_function in enumerate(metrics_array):
        current_metric = cv_scores[model_index, metric_index]
        print(f"\n {models_names[model_index]}, metric name: {metrics_names[metric_index]}, mean: {np.mean(current_metric):.5f}, std: {np.std(current_metric):.5f}")

