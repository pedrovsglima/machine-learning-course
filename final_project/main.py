import toml
import pandas as pd

from sklearn.model_selection import train_test_split


with open("config.toml", "r") as f:
    config = toml.load(f)


def main():

    df = pd.read_excel(
        config["dataset"]["file_path"],
        sheet_name=config["dataset"]["sheet_name"],
        na_values=config["dataset"]["na_values"]
    )

    # TODO: add colunas calculadas/estimadas
    # df['State-of-Charge'] = ...
    # df['Correlation-OCV-and-SOC'] = ...

    # 'Cell-Failure-Mechanism' has a typo
    df["Cell-Failure-Mechanism"] = df["Cell-Failure-Mechanism"].replace("Top Vent and Bottum Rupture", "Top Vent and Bottom Rupture")

    # split data into training and test sets
    X = df[config["dataset"]["input_cols"]]
    y = df[config["dataset"]["output_cols"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["features"]["test_size"],
        random_state=config["features"]["random_state"],
        stratify=X[config["dataset"]["stratify_col"]]
    )

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)


if __name__ == "__main__":
    main()
