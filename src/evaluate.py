import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:

    y_pred = model.predict(X_test)

    return regression_metrics(y_test, y_pred)