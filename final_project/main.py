import toml

from src.data import train_test_split, transform
from src.features import selection

with open("config.toml", "r") as f:
    config = toml.load(f)


def main():

    # read files and split data into training and test data
    X_train, X_test, y_train, y_test = train_test_split.split_data_from_file(
        file_data=config["dataset"],
        test_size=config["dataset"]["test_size"],
        random_state=config["dataset"]["random_state"]
    )
    print("Raw data:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    # feature engineering
    X_train, y_train = transform.execute_train(X_train, y_train)

    print("\nData transformation:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    # feature selection
    # existing methods: ["rfe", "mutual information", "andom forest"]
    best_features_rfe = selection.execute(X_train, y_train, method="rfe")
    best_features_mi = selection.execute(X_train, y_train, method="mutual information")
    best_features_rf = selection.execute(X_train, y_train, method="random forest")

    print(f"\nMethod 'RFE' selected {len(best_features_rfe)} features:\n{best_features_rfe}")
    print(f"\nMethod 'Mutual Information' selected {len(best_features_mi)} features:\n{best_features_mi}")
    print(f"\nMethod 'Random Forest' selected {len(best_features_rf)} features:\n{best_features_rf}")


if __name__ == "__main__":
    main()
