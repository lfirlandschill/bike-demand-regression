from pathlib import Path
import pandas as pd


def load_bike_data(path: str | Path) -> pd.DataFrame:
    """
    Expected file: hour.csv from the UCI Bike Sharing Dataset.
    """

    path = Path(path)
    df = pd.read_csv(path)

    return df


def clean_bike_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # convert bool cols
    bool_cols = ["yr", "holiday", "weekday", "workingday"]
    for col in bool_cols:
        df[col] = df[col].astype(bool)

    # convert categorical columns
    categorical_cols = ["season", "weathersit", "hr", "mnth"]
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    return df


def load_and_clean_bike_data(path: str | Path) -> pd.DataFrame:

    df = load_bike_data(path)
    df = clean_bike_data(df)

    return df