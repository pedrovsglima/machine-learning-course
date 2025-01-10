import pandas as pd

def to_excel(results: list) -> None:
    rows = []
    for entry in results:
        mse_list = entry["metrics"]["mse"] if isinstance(entry["metrics"]["mse"], list) else [entry["metrics"]["mse"]]
        mae_list = entry["metrics"]["mae"] if isinstance(entry["metrics"]["mae"], list) else [entry["metrics"]["mae"]]
        r2_list = entry["metrics"]["r2"] if isinstance(entry["metrics"]["r2"], list) else [entry["metrics"]["r2"]]
        rmse_list = entry["metrics"]["rmse"] if isinstance(entry["metrics"]["rmse"], list) else [entry["metrics"]["rmse"]]
        evs_list = entry["metrics"]["explained_variance_score"] if isinstance(entry["metrics"]["explained_variance_score"], list) else [entry["metrics"]["explained_variance_score"]]

        row = {
            "model": entry["model"],
            "feature_selection_method": entry["feature_selection_method"],
            "best_params": entry["best_params"],
            **{f"mse_{i+1}": mse for i, mse in enumerate(mse_list)},
            **{f"mae_{i+1}": mae for i, mae in enumerate(mae_list)},
            **{f"r2_{i+1}": r2 for i, r2 in enumerate(r2_list)},
            **{f"rmse_{i+1}": rmse for i, rmse in enumerate(rmse_list)},
            **{f"explained_variance_score_{i+1}": evs for i, evs in enumerate(evs_list)}
        }
        rows.append(row)

    pd.DataFrame(rows).to_excel("results/metrics.xlsx", index=False)
