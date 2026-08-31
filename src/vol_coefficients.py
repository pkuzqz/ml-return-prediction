from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/model_dataset.csv")

VOL_FEATURES = [
    "vol_5d",
    "vol_20d",
    "vol_63d",
]

TARGET = "target_21d"

TEST_YEARS = [2022, 2023, 2024, 2025]
PURGE_DAYS = 21


def make_fold(df, year):
    test_start = pd.Timestamp(f"{year}-01-01")

    pre_test_dates = np.sort(
        df.loc[
            df["date"] < test_start,
            "date"
        ].unique()
    )

    purge_start = pd.Timestamp(
        pre_test_dates[-PURGE_DAYS]
    )

    train = df[
        df["date"] < purge_start
    ].copy()

    return train


if __name__ == "__main__":

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    ).sort_values(
        ["date", "ticker"]
    )

    for year in TEST_YEARS:

        train = make_fold(
            df,
            year
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            train[VOL_FEATURES]
        )

        model = Ridge(alpha=1.0)

        model.fit(
            X_train,
            train[TARGET]
        )

        coefficients = pd.Series(
            model.coef_,
            index=VOL_FEATURES
        )

        print()
        print("=" * 55)
        print(f"MODEL USED TO PREDICT {year}")
        print("=" * 55)

        print(
            coefficients.round(6)
        )