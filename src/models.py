from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

def build_ridge_model(
    preprocessor: ColumnTransformer,
    alpha: float = 0.0,
) -> Pipeline:

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=alpha)),
        ]
    )

    return model


def build_tree_model(
    preprocessor: ColumnTransformer,
    max_depth: int | None = 5,
) -> Pipeline:

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeRegressor(
                max_depth=max_depth,
                random_state=42,
            )),
        ]
    )

    return model

def build_xgboost_model(
    preprocessor,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    n_estimators: int = 300,
):

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBRegressor(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    random_state=42,
                ),
            ),
        ]
    )

    return model