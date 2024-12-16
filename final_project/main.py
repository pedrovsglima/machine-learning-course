import toml
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.data import train_test_split, transform
from src.features import correlation, selection
from src.models import regression, save_model

with open("config.toml", "r") as f:
    config = toml.load(f)


def get_all_feature_selection_scenarios(x_train, x_test, y_train, methods) -> dict:

    best_features = {method: selection.execute(x_train, y_train, method) for method in methods}

    selection.save_to_excel(best_features)

    feature_selection_data = {method: (x_train[features], x_test[features]) for method, features in best_features.items()}

    feature_selection_data["all"] = (x_train, x_test)

    return feature_selection_data

def main():

    # read files and split data into training and test data
    X_train, X_test, y_train, y_test = train_test_split.split_data_from_file(
        file_data=config["dataset"],
        test_size=config["dataset"]["test_size"],
        random_state=config["dataset"]["random_state"]
    )

    # feature engineering
    X_train, X_test, y_train, y_test = transform.execute(X_train, X_test, y_train, y_test)

    # remove correlated features
    X_train_corr, X_test_corr = correlation.remove_correlated_features(
        X_train, X_test,
        y_train, y_test,
        config["feature_selection"]["corr_threshold"]
    )

    # all feature selection scenarios
    methods = ["rfe", "mutual information", "random forest"]
    feature_selection_data = get_all_feature_selection_scenarios(X_train, X_test, y_train, methods)
    feature_selection_data_corr = get_all_feature_selection_scenarios(X_train_corr, X_test_corr, y_train, methods)

    feature_selection_data.update(
        {key+"_corr": value for key, value in feature_selection_data_corr.items()}
    )

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
                "best_params": model.get_best_params(),
                "metrics": metrics
            })

    save_model.to_excel(results)

if __name__ == "__main__":
    main()
