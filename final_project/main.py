import toml
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.data import train_test_split, transform
from src.features import selection
from src.models.regression import RegressionModel

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
    X_train, X_test, y_train, y_test = transform.execute(X_train, X_test, y_train, y_test)

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

    feature_selection_data = {
        "rfe": (X_train[best_features_rfe], X_test[best_features_rfe]),
        "mutual information": (X_train[best_features_mi], X_test[best_features_mi]),
        "random forest": (X_train[best_features_rf], X_test[best_features_rf]),
        "all": (X_train, X_test)
    }

    models = [
        {"name": "random_forest", "model": RandomForestRegressor(random_state=42), "grid_params": config["random_forest"]},
        # {"name": "linear_regression", "model": LinearRegression(), "grid_params": config["linear_regression"]},
    ]

    for model_dict in models:
        model = RegressionModel(model_dict["model"], model_dict["grid_params"])
        for fs_method, (x_train, x_test) in feature_selection_data.items():
            model.perform_grid_search(x_train, y_train, cv_params=config["cross_validation"], scoring=config["grid_search"]["scoring"])
            y_pred = model.predict(x_test)
            metrics = model.get_metrics(y_test, y_pred, multioutput=config["metrics"]["multioutput_method"])
            print(model_dict["name"], fs_method)
            print(model.get_best_params())
            print(metrics)

if __name__ == "__main__":
    main()
