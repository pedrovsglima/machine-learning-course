import pandas as pd

from sklearn.model_selection import train_test_split


def split_data_from_file(file_data:dict, test_size:float, random_state:int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df = pd.read_excel(
        file_data["file_path"],
        sheet_name=file_data["sheet_name"],
        na_values=file_data["na_values"]
    )

    # 'Cell-Failure-Mechanism' has a typo
    # df["Cell-Failure-Mechanism"] = df["Cell-Failure-Mechanism"].replace("Top Vent and Bottum Rupture", "Top Vent and Bottom Rupture")

    # split data into training and test sets
    X = df[file_data["input_cols"]]
    y = df[file_data["output_cols"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=X[file_data["stratify_col"]]
    )

    return X_train, X_test, y_train, y_test
