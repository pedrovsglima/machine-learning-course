import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


def remove_null_rows(X:pd.DataFrame, y:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    remove_indexes = y[y.isnull().any(axis=1)].index

    y.drop(index=remove_indexes, inplace=True)
    y.reset_index(drop=True, inplace=True)

    X.drop(index=remove_indexes, inplace=True)
    X.reset_index(drop=True, inplace=True)

    return X, y

def one_hot_encoder(X:pd.DataFrame) -> pd.DataFrame:
    
    obj_cols = [col for col in X.columns if X[col].dtype == "object"]

    X[obj_cols] = X[obj_cols].astype(str)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_ohe = ohe.fit_transform(X[obj_cols])

    cols_ohe = ohe.get_feature_names_out()
    X_ohe = pd.DataFrame(X_ohe, columns=cols_ohe)

    # TODO: reaproveitar mesmo OneHotEncoder() para test set

    return X_ohe

def normalize(X:pd.DataFrame) -> pd.DataFrame:

    numerical_cols = [col for col in X.columns if X[col].dtype == np.float64]

    X_num = X[numerical_cols]

    train_mean = X_num.mean(axis=0)
    train_std = X_num.std(axis=0)

    X_num = (X_num - train_mean) / train_std

    # TODO: reaproveitar mesmos mean e std para test set

    return X_num


def execute_train(X_train:pd.DataFrame, y_train:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    # remove null values
    # because only one row is null for our target cols, we may not lose much information dropping it
    X_train, y_train = remove_null_rows(X_train, y_train)

    # TODO: decidir o que fazer com esses campos
    X_train.drop(columns=["Cell-Description", "Pre-Test-Cell-Open-Circuit-Voltage-V"], inplace=True)

    # dados categóricos e nominais
    X_train_obj = one_hot_encoder(X_train)
    # dados numericos
    X_train_num = normalize(X_train)

    X_train = X_train_num.join(X_train_obj, how="inner")

    return X_train, y_train
