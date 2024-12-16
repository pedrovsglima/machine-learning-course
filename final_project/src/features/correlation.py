import pandas as pd

from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def calculate_mse_for_each_feature(
        x_train:pd.DataFrame, x_test:pd.DataFrame,
        y_train:pd.DataFrame, y_test:pd.DataFrame
) -> dict:
    scores = {}

    for col in x_train.columns:

        # remove rows with NaN values
        x_train_dropna = x_train[[col]].dropna()
        y_train_dropna = y_train.loc[x_train_dropna.index]
        x_test_dropna = x_test[[col]].dropna()
        y_test_dropna = y_test.loc[x_test_dropna.index]

        model = LinearRegression()  # train a simple model
        model.fit(x_train_dropna, y_train_dropna)  # using only the ith feature

        y_pred = model.predict(x_test_dropna)
        mse = mean_squared_error(y_test_dropna, y_pred, multioutput="uniform_average")

        scores[col] = mse

    return scores

def drop_redudant_features_by_importance(feature_importance:dict, correlation_threshold:float, X:pd.DataFrame) -> pd.DataFrame:

    cols_to_remove = set()

    for col_1, col_2 in combinations(X.columns, 2):
        if col_1 in cols_to_remove or col_2 in cols_to_remove:
            continue

        cols_corr = abs(X[col_1].corr(X[col_2], method="pearson"))

        if cols_corr > correlation_threshold:
            # keep the feature most informative for the prediction (i.e. lowest MSE)
            remove_feature = col_2 if feature_importance[col_1] < feature_importance[col_2] else col_1
            cols_to_remove.add(remove_feature)

    return X.drop(columns=list(cols_to_remove))

def remove_correlated_features(
        x_train: pd.DataFrame, x_test: pd.DataFrame,
        y_train: pd.DataFrame, y_test: pd.DataFrame,
        correlation_threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:

    feature_errors = calculate_mse_for_each_feature(x_train, x_test, y_train, y_test)

    x_train_corr = drop_redudant_features_by_importance(feature_errors, correlation_threshold, x_train)

    x_test_corr = x_test[x_train_corr.columns]

    return x_train_corr, x_test_corr
