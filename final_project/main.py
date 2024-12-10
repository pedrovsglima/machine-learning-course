import toml
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.data import train_test_split, transform
from src.features import selection
from src.models import regression, save_model

with open("config.toml", "r") as f:
    config = toml.load(f)


def main():

    # read files and split data into training and test data
    X_train, X_test, y_train, y_test = train_test_split.split_data_from_file(
        file_data=config["dataset"],
        test_size=config["dataset"]["test_size"],
        random_state=config["dataset"]["random_state"]
    )

    # feature engineering
    X_train, X_test, y_train, y_test = transform.execute(X_train, X_test, y_train, y_test)

    # feature selection
    methods = ["rfe", "mutual information", "random forest"]
    best_features = {method: selection.execute(X_train, y_train, method=method) for method in methods}
    selection.save_to_excel(best_features)

    feature_selection_data = {
        method: (X_train[features], X_test[features]) for method, features in best_features.items()
    }
    feature_selection_data["all"] = (X_train, X_test)

    # model training and evaluation
    models = [
        {"name": "random_forest", "model": RandomForestRegressor(random_state=42), "grid_params": config["random_forest"]},
        # {"name": "linear_regression", "model": LinearRegression(), "grid_params": config["linear_regression"]},
    ]

    results = []
    for model_dict in models:
        model = regression.RegressionModel(model_dict["model"], model_dict["grid_params"])
        for fs_method, (x_train, x_test) in feature_selection_data.items():
            model.perform_grid_search(x_train, y_train, cv_params=config["cross_validation"], scoring=config["grid_search"]["scoring"])
            y_pred = model.predict(x_test)
            metrics = model.get_metrics(y_test, y_pred, multioutput=config["metrics"]["multioutput_method"])
            
            results.append({
                "model": model_dict["name"],
                "feature_selection_method": fs_method,
                "best_params": model.get_best_params(), # TODO: how to choose the best model?
                "metrics": metrics
            })

    save_model.to_excel(results)

if __name__ == "__main__":
    main()
