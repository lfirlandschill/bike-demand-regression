import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import plot_tree


def plot_ridge_validation(
    alphas: list[float],
    rmses: list[float],
):

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, rmses, marker="o")
    plt.xscale("log")
    plt.xlabel("Alpha")
    plt.ylabel("Validation RMSE")
    plt.title("Ridge Validation Curve")
    plt.grid(True)

    plt.savefig("figures/results/ridge_validation.png")


def plot_tree_validation(
    depths: list,
    rmses: list[float],
):

    depth_labels = [str(d) for d in depths]
    plt.figure(figsize=(8, 5))
    plt.plot(depth_labels, rmses, marker="o")
    plt.xlabel("Max Depth")
    plt.ylabel("Validation RMSE")
    plt.title("Decision Tree Validation Curve")
    plt.grid(True)

    plt.savefig("figures/results/tree_validation.png")

def plot_xgboost_validation(
    depths: list,
    rmses: list[float],
):

    depth_labels = [str(d) for d in depths]
    plt.figure(figsize=(8, 5))
    plt.plot(depth_labels, rmses, marker="o")
    plt.xlabel("Max Depth")
    plt.ylabel("Validation RMSE")
    plt.title("XGBoost Validation Curve")
    plt.grid(True)

    plt.savefig("figures/results/xgboost_validation.png")

def clean_feature_name(feature: str) -> str:
    # Remove transformer prefixes
    feature = feature.replace("num__", "").replace("cat__", "")

    if feature.startswith("hr_"):
        return "hour"
    if feature.startswith("mnth_"):
        return "month"
    if feature.startswith("season_"):
        return "season"
    if feature.startswith("weekday_"):
        return "weekday"
    if feature.startswith("weathersit_"):
        return "weather"

    return feature

def plot_feature_importance(tree_model):
    """
    Plot aggregated Decision Tree feature importances.
    """

    model = tree_model.named_steps["model"]
    preprocessor = tree_model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    # Group one-hot encoded features into original feature groups
    importance_df["feature_group"] = importance_df["feature"].apply(clean_feature_name)

    importance_df = (
        importance_df
        .groupby("feature_group", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["feature_group"],
        importance_df["importance"],
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Aggregated Decision Tree Feature Importances")

    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig("figures/results/tree_feature_importance.png")

def plot_linear_coefficients(ridge_model):

    model = ridge_model.named_steps["model"]

    preprocessor = ridge_model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_,
    })

    coef_df["abs_coefficient"] = (
        coef_df["coefficient"].abs()
    )

    coef_df = coef_df.sort_values(
        "abs_coefficient",
        ascending=False,
    ).head(30)

    plt.figure(figsize=(10, 6))

    plt.barh(
        coef_df["feature"],
        coef_df["coefficient"],
    )

    plt.xlabel("Coefficient")
    plt.ylabel("Feature")

    plt.title("Ridge Regression Coefficients")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig("figures/results/ridge_linear_coefficients.png")

def visualize_tree(tree_model, max_depth: int = 2):
    
    model = tree_model.named_steps["model"]
    preprocessor = tree_model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    plt.figure(figsize=(20, 10))

    plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=10,
    )

    plt.title("Decision Tree Visualization")

    plt.savefig("figures/results/tree_visualization.png")

def plot_actual_vs_predicted(model, X, y, name):
    """
    Scatterplot of actual vs predicted values.
    """

    y_pred = model.predict(X)

    plt.figure(figsize=(7, 7))

    plt.scatter(
        y,
        y_pred,
        alpha=0.5,
    )

    # Perfect prediction line
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
    )

    plt.xlabel("Actual Demand")
    plt.ylabel("Predicted Demand")

    plt.title("Actual vs Predicted")

    plt.tight_layout()

    plt.savefig("figures/results/" + name + "_scatterplot.png")