import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    
    df["windspeed_sq"] = df["windspeed"] ** 2

    return df