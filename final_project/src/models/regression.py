import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, explained_variance_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.model_selection import RepeatedKFold


class RegressionModel:
    def __init__(self, model: BaseEstimator, param_grid: dict[str, Any]):
        self.model = model
        self.param_grid = param_grid
        self.grid_search = None

    def perform_grid_search(self, x_train: pd.DataFrame, y_train: pd.Series, cv_params: dict, scoring: str) -> None:
        
        cv = RepeatedKFold(**cv_params)
    
        self.grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=cv,
            scoring=scoring,
            error_score=0
        )
        self.grid_search.fit(x_train, y_train)
        self.model = self.grid_search.best_estimator_

    def get_best_params(self) -> dict[str, Any]:
        return self.grid_search.best_params_

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(x_test)

    def get_metrics(self, y_test: pd.DataFrame, y_pred: np.ndarray, multioutput: str) -> dict[str, float]:
        mse = mean_squared_error(y_test, y_pred, multioutput=multioutput)
        mae = mean_absolute_error(y_test, y_pred, multioutput=multioutput)
        r2 = r2_score(y_test, y_pred, multioutput=multioutput)
        rmse = root_mean_squared_error(y_test, y_pred, multioutput=multioutput)
        evs = explained_variance_score(y_test, y_pred, multioutput=multioutput)
        return {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "rmse": rmse,
            "explained_variance_score": evs
        }

    # def error_confidence_intervals(self, y_true: pd.DataFrame, y_pred: np.ndarray, confidence: float = 0.95) -> dict:
    #     from scipy.stats import t
    #     n_samples = y_true.shape[0]
    #     results = {}
    #     for i,v in enumerate(y_true.columns):
    #         residuals = y_true[v] - y_pred[:, i]
    #         mean_error = np.mean(residuals)
    #         stdev = np.sqrt(np.sum(residuals**2) / (n_samples - 1))
    #         sem = stdev / np.sqrt(n_samples)  # standard error of the mean
    #         # margin of error using t-distribution
    #         margin = t.ppf((1 + confidence) / 2, df=n_samples - 1) * sem
    #         results[f"output_{i}"] = {
    #             "mean_error": mean_error,
    #             "ci_lower": mean_error - margin,
    #             "ci_upper": mean_error + margin
    #         }
    #     return results

    # def save_figure(self, x_test: pd.DataFrame, y_test: pd.Series, y_pred: pd.Series, filename: str) -> None:
    #     plt.figure(figsize=(10, 6))
    #     plt.scatter(x_test.iloc[:, 0], y_test, color="blue", label="Actual")
    #     plt.scatter(x_test.iloc[:, 0], y_pred, color="red", label="Predicted")
    #     plt.xlabel("X Test")
    #     plt.ylabel("Y Test / Predicted")
    #     plt.legend()
    #     plt.savefig(filename)
    #     plt.close()
