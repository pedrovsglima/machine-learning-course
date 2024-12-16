import os
import numpy as np
import pandas as pd

from abc import ABC
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class FeatureSelectionMethod(ABC):
    IMPROVEMENT_THRESHOLD = 0.05

    def __init__(self, X:pd.DataFrame, y:pd.DataFrame):
        self.X = X
        self.y = y

    def _get_ranked_features(self) -> list:
        """Rank features based on a specific method."""
        raise NotImplementedError("_get_ranked_features is not implemented for this class.")

    def select_k_features(self, k:int) -> list:
        """Select the top-k features based on their ranking."""
        ranked_features = self._get_ranked_features()
        return ranked_features[:k]

    def find_best_features(self) -> list:
        """Find the best features based on some evaluation criterion."""
        ranked_features = self._get_ranked_features()

        best_score = -np.inf
        best_n_features = 0

        for n_selected in range(1, self.X.shape[1] + 1):

            selected_features = ranked_features[:n_selected]

            X_selected = self.X[selected_features]

            model = LinearRegression()
            cv_score = cross_val_score(model, X_selected, self.y, cv=5, scoring="r2")
            mean_cv_score = np.mean(cv_score)

            if mean_cv_score > best_score * (1 + self.IMPROVEMENT_THRESHOLD):
                best_score = mean_cv_score
                best_n_features = n_selected

        return ranked_features[:best_n_features]

class RFElimination(FeatureSelectionMethod):

    def select_k_features(self, k:int) -> list:

        model = LinearRegression()

        rfe = RFE(model, n_features_to_select=k)
        rfe.fit(self.X, self.y)

        return list(self.X.columns[rfe.support_])

    def find_best_features(self) -> list:
        best_score = 0
        best_features = []

        model = LinearRegression()

        for n in range(1, self.X.shape[1]+1):

            selected_features = self.select_k_features(n)

            X_selected = self.X[selected_features]

            scores = cross_val_score(model, X_selected, self.y, cv=5, scoring="r2")
            mean_score = scores.mean()

            if mean_score > best_score * (1 + self.IMPROVEMENT_THRESHOLD):
                best_score = mean_score
                best_features = selected_features

        return best_features

class MutualInformation(FeatureSelectionMethod):

    def _get_ranked_features(self) -> list:

        mi_scores = np.zeros((self.X.shape[1], self.y.shape[1]))

        for i in range(self.y.shape[1]):  # for each target
            mi_scores[:, i] = mutual_info_regression(self.X, self.y.iloc[:, i])

        # aggregate the mutual information scores across all targets
        # simple approach: sum the MI scores for each feature across all outputs
        aggregate_mi_scores = np.sum(mi_scores, axis=1)

        # sort features by MI score in descending order
        ranked_features = np.argsort(aggregate_mi_scores)[::-1]

        return list(self.X.columns[ranked_features])

class FeatureImportanceRF(FeatureSelectionMethod):

    def _get_ranked_features(self) -> list:

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(self.X, self.y)

        # sort features
        feature_importances = model.feature_importances_
        ranked_importances = np.argsort(feature_importances)[::-1]

        return list(self.X.columns[ranked_importances])


def execute(X:pd.DataFrame, y:pd.DataFrame, method:str, select_k:int=None) -> list:

    methods = {
        "rfe": RFElimination,
        "mutual information": MutualInformation,
        "random forest": FeatureImportanceRF
    }

    if method not in methods:
        raise NotImplementedError(f"Method {method} does not exist.")

    method_instance = methods[method](X, y)

    if select_k:
        return method_instance.select_k_features(select_k)
    else:
        return method_instance.find_best_features()

def save_to_excel(data:dict[str, list]) -> None:
    file_path = "results/selected_features.xlsx"
    new_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))

    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)
        combined_df = pd.concat([existing_df, new_df], axis=1)
    else:
        combined_df = new_df

    combined_df.to_excel(file_path, index=False)
