import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import OneHotEncoder


def add_calculated_columns(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    soc = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ocv = np.array([3.34, 3.53, 3.59, 3.63, 3.66, 3.70, 3.75, 3.83, 3.92, 4.03, 4.13])
    ocv_to_soc = interp1d(ocv, soc, kind="linear", fill_value="extrapolate")

    def apply_ocv_to_soc(df: pd.DataFrame) -> pd.DataFrame:
        df["State-of-Charge"] = df["Pre-Test-Cell-Open-Circuit-Voltage-V"].apply(ocv_to_soc).clip(0, 100)
        return df

    def assign_chemistry(df: pd.DataFrame) -> pd.DataFrame:
        conditions = [
            df["Cell-Description"].isin(["KULR 18650-K330", "KULR 21700-K500", "Panasonic 18650-BE", "Saft D-Cell-VES16"]),
            df["Cell-Description"].isin(["LG 21700-M50 (BV)", "MOLiCEL 18650-J"])
        ]
        choices = ["NCA/Graphite", "NMC/Graphite-SiOx"]
        df["Chemistry"] = np.select(conditions, choices, default="NMC/Graphite")
        return df

    x_train = apply_ocv_to_soc(x_train)
    x_train = assign_chemistry(x_train)

    x_test = apply_ocv_to_soc(x_test)
    x_test = assign_chemistry(x_test)

    return x_train, x_test

def remove_null_rows_on_target(X:pd.DataFrame, y:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    remove_indexes = y[y.isnull().any(axis=1)].index

    y.drop(index=remove_indexes, inplace=True)
    y.reset_index(drop=True, inplace=True)

    X.drop(index=remove_indexes, inplace=True)
    X.reset_index(drop=True, inplace=True)

    return X, y

def binary_encoder(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # edit columns to get only the material 'XXXXX-{material}'
    for col in ["Pressure-Assisted-Seal-Configuration-Positive", "Pressure-Assisted-Seal-Configuration-Negative"]:
        x_train[col] = x_train[col].str.split("-").str[1]
        x_test[col] = x_test[col].str.split("-").str[1]

    binary_cols = [col for col in x_train.columns if x_train[col].nunique() == 2]

    x_train_bin = x_train[binary_cols].copy()
    x_test_bin = x_test[binary_cols].copy().reset_index(drop=True)

    for col in binary_cols:
        unique_values = x_train_bin[col].unique()
        mapping = {val: idx for idx, val in enumerate(unique_values)}
        x_train_bin[col] = x_train_bin[col].map(mapping)
        x_test_bin[col] = x_test_bin[col].map(mapping)

    return x_train_bin, x_test_bin

def one_hot_encoder(x_train:pd.DataFrame, x_test:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    obj_cols_train = [col for col in x_train.columns if x_train[col].dtype == "object" and x_train[col].nunique() > 2]
    obj_cols_test = [col for col in x_test.columns if x_test[col].dtype == "object" and x_test[col].nunique() > 2]

    x_train[obj_cols_train] = x_train[obj_cols_train].astype(str)
    x_test[obj_cols_test] = x_test[obj_cols_test].astype(str)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x_train_ohe = ohe.fit(x_train[obj_cols_train])

    x_train_ohe = ohe.transform(x_train[obj_cols_train])
    cols_ohe = ohe.get_feature_names_out()
    x_train_ohe = pd.DataFrame(x_train_ohe, columns=cols_ohe)

    x_test_ohe = ohe.transform(x_test[obj_cols_test])
    x_test_ohe = pd.DataFrame(x_test_ohe, columns=cols_ohe)

    return x_train_ohe, x_test_ohe

def normalize(x_train:pd.DataFrame, x_test:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    numerical_cols = [col for col in x_train.columns if x_train[col].dtype == np.float64]

    x_train_num = x_train[numerical_cols]
    x_test_num = x_test[numerical_cols].reset_index(drop=True)

    train_mean = x_train_num.mean(axis=0)
    train_std = x_train_num.std(axis=0)

    x_train_num = (x_train_num - train_mean) / train_std
    x_test_num = (x_test_num - train_mean) / train_std

    return x_train_num, x_test_num

def execute(
        x_train:pd.DataFrame, x_test:pd.DataFrame,
        y_train:pd.DataFrame, y_test:pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # remove null values
    x_train, y_train = remove_null_rows_on_target(x_train, y_train)
    x_test, y_test = remove_null_rows_on_target(x_test, y_test)

    # add columns "State-of-Charge" and "Chemistry"
    x_train, x_test = add_calculated_columns(x_train, x_test)

    # TODO: decidir o que fazer com esses campos
    cols_to_drop = ["Cell-Description", "Pre-Test-Cell-Open-Circuit-Voltage-V", "State-of-Charge"]
    x_train.drop(columns=cols_to_drop, inplace=True)
    x_test.drop(columns=cols_to_drop, inplace=True)

    # dados categoricos e binários
    x_train_bin, x_test_bin = binary_encoder(x_train, x_test)
    # dados categóricos e nominais
    x_train_obj, x_test_obj = one_hot_encoder(x_train, x_test)
    # dados numericos
    x_train_num, x_test_num = normalize(x_train, x_test)

    x_train = x_train_num.join([x_train_obj, x_train_bin], how="inner")
    x_test = x_test_num.join([x_test_obj, x_test_bin], how="inner")

    return x_train, x_test, y_train, y_test
