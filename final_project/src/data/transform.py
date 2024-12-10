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

def one_hot_encoder(X_train:pd.DataFrame, X_test:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    obj_cols_train = [col for col in X_train.columns if X_train[col].dtype == "object"]
    obj_cols_test = [col for col in X_test.columns if X_test[col].dtype == "object"]

    X_train[obj_cols_train] = X_train[obj_cols_train].astype(str)
    X_test[obj_cols_test] = X_test[obj_cols_test].astype(str)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_ohe = ohe.fit(X_train[obj_cols_train])

    X_train_ohe = ohe.transform(X_train[obj_cols_train])
    cols_ohe = ohe.get_feature_names_out()
    X_train_ohe = pd.DataFrame(X_train_ohe, columns=cols_ohe)

    X_test_ohe = ohe.transform(X_test[obj_cols_test])
    X_test_ohe = pd.DataFrame(X_test_ohe, columns=cols_ohe)

    return X_train_ohe, X_test_ohe

def normalize(X_train:pd.DataFrame, X_test:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    numerical_cols = [col for col in X_train.columns if X_train[col].dtype == np.float64]

    X_train_num = X_train[numerical_cols]
    X_test_num = X_test[numerical_cols].reset_index(drop=True)

    train_mean = X_train_num.mean(axis=0)
    train_std = X_train_num.std(axis=0)

    X_train_num = (X_train_num - train_mean) / train_std
    X_test_num = (X_test_num - train_mean) / train_std

    return X_train_num, X_test_num

def execute(
        X_train:pd.DataFrame, X_test:pd.DataFrame,
        y_train:pd.DataFrame, y_test:pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # remove null values
    # because only one row is null for our target cols, we may not lose much information dropping it
    X_train, y_train = remove_null_rows(X_train, y_train)
    X_test, y_test = remove_null_rows(X_test, y_test)

    # TODO: decidir o que fazer com esses campos
    X_train.drop(columns=["Cell-Description", "Pre-Test-Cell-Open-Circuit-Voltage-V"], inplace=True)
    X_test.drop(columns=["Cell-Description", "Pre-Test-Cell-Open-Circuit-Voltage-V"], inplace=True)

    # dados categóricos e nominais
    X_train_obj, X_test_obj = one_hot_encoder(X_train, X_test)
    # dados numericos
    X_train_num, X_test_num = normalize(X_train, X_test)

    X_train = X_train_num.join(X_train_obj, how="inner")
    X_test = X_test_num.join(X_test_obj, how="inner")

    return X_train, X_test, y_train, y_test
