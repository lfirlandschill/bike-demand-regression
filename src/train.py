import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer

from src.evaluate import evaluate_model
from src.models import build_ridge_model
from src.models import build_tree_model
from src.models import build_xgboost_model
from src.plots import *

def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    target_col = "cnt"
    leakage_cols = ["casual", "registered"]
    not_using_cols = ["atemp", "instant", "dteday"]
    drop_cols = [target_col] + leakage_cols + not_using_cols
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    return X, y


def unscale_weather_units(X):
    X = X.copy()

    X["temp"] = X["temp"] * 47 - 8
    # X["atemp"] = X["atemp"] * 66 - 16
    X["hum"] = X["hum"] * 100
    X["windspeed"] = X["windspeed"] * 67

    return X

def build_preprocessors(X: pd.DataFrame) -> tuple[ColumnTransformer, ColumnTransformer]:

    # one-hot encode categorical features
    # scale numeric features (ridge only)
    # unscale weather for interpretability (tree only)

    categorical_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
    weather_cols = ["temp", "hum", "windspeed"]

    ridge_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    tree_numeric_pipeline = Pipeline(
        steps=[
            ("unit_conversion",
             FunctionTransformer(unscale_weather_units, feature_names_out="one-to-one")),
        ]
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", tree_numeric_pipeline, weather_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return ridge_preprocessor, tree_preprocessor

def train_val_test_split(X, y):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42
    )

    return (X_train, X_val, X_test,
            y_train, y_val, y_test)

def select_ridge_model(preprocessor, ridge_alphas, X_train, y_train, X_val, y_val):
    rmses = []
    best_alpha = None
    best_rmse = float("inf")
    for alpha in ridge_alphas:
        model = build_ridge_model(preprocessor, alpha)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_val, y_val)
        rmses.append(metrics["rmse"])
        if (metrics["rmse"] < best_rmse):
            best_rmse = metrics["rmse"]
            best_alpha = alpha
    return (best_alpha, rmses)


def select_tree_model(preprocessor, tree_depths, X_train, y_train, X_val, y_val):
    rmses = []
    best_depth = None
    best_rmse = float("inf")
    for depth in tree_depths:
        model = build_tree_model(preprocessor, depth)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_val, y_val)
        rmses.append(metrics["rmse"])
        if (metrics["rmse"] < best_rmse):
            best_rmse = metrics["rmse"]
            best_depth = depth
    return (best_depth, rmses)

def select_xgboost_model(preprocessor, xgb_depths, X_train, y_train, X_val, y_val):
    rmses = []
    best_depth = None
    best_rmse = float("inf")
    for depth in xgb_depths:
        model = build_xgboost_model(preprocessor, depth)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_val, y_val)
        rmses.append(metrics["rmse"])
        if (metrics["rmse"] < best_rmse):
            best_rmse = metrics["rmse"]
            best_depth = depth
    return (best_depth, rmses)


def train_and_evaluate(df: pd.DataFrame) -> pd.DataFrame:

    X, y = split_features_target(df)

    (X_train, X_val, X_test, y_train, y_val, y_test) = train_val_test_split(X, y)

    ridge_preprocessor, tree_preprocessor = build_preprocessors(X_train)

    ridge_alphas = np.logspace(-2, 2, 20)
    best_alpha, ridge_rmses = select_ridge_model(ridge_preprocessor, ridge_alphas, X_train, y_train, X_val, y_val)

    tree_depths = np.linspace(3, 50, 10, dtype=int)
    best_depth, tree_rmses = select_tree_model(tree_preprocessor, tree_depths, X_train, y_train, X_val, y_val)

    xgb_depths = np.linspace(2, 10, 9, dtype=int)
    best_xgb_depth, xgb_rmses = select_xgboost_model(tree_preprocessor, xgb_depths, X_train, y_train, X_val, y_val)

    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    best_ridge = build_ridge_model(ridge_preprocessor, alpha=best_alpha)
    best_ridge.fit(X_trainval, y_trainval)

    best_tree = build_tree_model(tree_preprocessor, max_depth=best_depth)
    best_tree.fit(X_trainval, y_trainval)

    best_xgb = build_xgboost_model(tree_preprocessor, max_depth=best_xgb_depth)
    best_xgb.fit(X_trainval, y_trainval)

    ridge_test_metrics = evaluate_model(best_ridge, X_test, y_test)
    tree_test_metrics = evaluate_model(best_tree, X_test, y_test)
    xgb_test_metrics = evaluate_model(best_xgb, X_test, y_test)

    importances = best_tree.named_steps["model"].feature_importances_

    plot_ridge_validation(ridge_alphas, ridge_rmses)
    plot_tree_validation(tree_depths, tree_rmses)
    plot_xgboost_validation(xgb_depths, xgb_rmses)
    plot_feature_importance(best_tree)
    plot_linear_coefficients(best_ridge)
    visualize_tree(best_tree)
    plot_actual_vs_predicted(best_ridge, X_test, y_test, "ridge")
    plot_actual_vs_predicted(best_tree, X_test, y_test, "tree")
    plot_actual_vs_predicted(best_xgb, X_test, y_test, "xgboost")

    results = pd.DataFrame([
        {
            "model": "ridge",
            "test_rmse": ridge_test_metrics["rmse"],
            "alpha": best_alpha,
        },
        {
            "model": "tree",
            "test_rmse": tree_test_metrics["rmse"],
            "max_depth": best_depth,
        },
        {
            "model": "xgboost",
            "test_rmse": xgb_test_metrics["rmse"],
            "max_depth": best_xgb_depth,
        },
    ])

    return results