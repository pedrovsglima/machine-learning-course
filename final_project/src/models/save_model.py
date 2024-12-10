import pandas as pd

def to_excel(results: list) -> None:
    rows = [
        {
            "model": entry["model"],
            "feature_selection_method": entry["feature_selection_method"],
            "best_params": entry["best_params"],
            **{f"mse_{i+1}": mse for i, mse in enumerate(entry["metrics"]["mse"])},
            **{f"mae_{i+1}": mae for i, mae in enumerate(entry["metrics"]["mae"])},
            **{f"r2_{i+1}": r2 for i, r2 in enumerate(entry["metrics"]["r2"])},
            **{f"rmse_{i+1}": rmse for i, rmse in enumerate(entry["metrics"]["rmse"])},
            **{f"explained_variance_score_{i+1}": evs for i, evs in enumerate(entry["metrics"]["explained_variance_score"])}
        }
        for entry in results
    ]

    pd.DataFrame(rows).to_excel("results/metrics.xlsx", index=False)
