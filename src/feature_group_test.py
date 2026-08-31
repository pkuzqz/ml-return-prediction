from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/model_dataset.csv")
TARGET = "target_21d"

PURGE_DAYS = 21
TEST_YEARS = [2022, 2023, 2024, 2025]


FEATURE_GROUPS = {
    "return_momentum": [
        "ret_1d",
        "ret_5d",
        "ret_21d",
        "ret_63d",
        "ret_126d",
        "momentum_126_5",
    ],

    "volatility": [
        "vol_5d",
        "vol_20d",
        "vol_63d",
    ],

    "trend_position": [
        "drawdown_63d",
        "ma_ratio_5_21",
        "ma_ratio_21_63",
    ],
}


ALL_FEATURES = sum(
    FEATURE_GROUPS.values(),
    []
)


def make_fold(df, year):
    test_start = pd.Timestamp(
        f"{year}-01-01"
    )

    test_end = pd.Timestamp(
        f"{year + 1}-01-01"
    )

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

    test = df[
        (df["date"] >= test_start)
        & (df["date"] < test_end)
    ].copy()

    return train, test


def run_ridge(
    train,
    test,
    features
):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train[features]
    )

    X_test = scaler.transform(
        test[features]
    )

    model = Ridge(alpha=1.0)

    model.fit(
        X_train,
        train[TARGET]
    )

    return model.predict(
        X_test
    )


def mean_rank_ic(
    test,
    predictions
):
    result = test[
        ["date", TARGET]
    ].copy()

    result["prediction"] = predictions

    daily_ic = result.groupby("date").apply(
        lambda g:
        g["prediction"].rank().corr(
            g[TARGET].rank()
        ),
        include_groups=False,
    )

    return daily_ic


if __name__ == "__main__":

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    ).sort_values(
        ["date", "ticker"]
    )

    feature_sets = {
        "vol_5d": [
            "vol_5d",
        ],

        "vol_20d": [
            "vol_20d",
        ],

        "vol_63d": [
            "vol_63d",
        ],

        "all_volatility": [
            "vol_5d",
            "vol_20d",
            "vol_63d",
        ],

        "vol_plus_trend": [
            "vol_5d",
            "vol_20d",
            "vol_63d",
            "drawdown_63d",
            "ma_ratio_5_21",
            "ma_ratio_21_63",
        ],

        "all_features": ALL_FEATURES,
    }

    overall = {}

    for name, features in feature_sets.items():

        all_ic = []

        print()
        print("=" * 60)
        print(name)
        print("=" * 60)

        for year in TEST_YEARS:

            train, test = make_fold(
                df,
                year
            )

            prediction = run_ridge(
                train,
                test,
                features
            )

            ic = mean_rank_ic(
                test,
                prediction
            )

            all_ic.append(ic)

            print(
                f"{year}: {ic.mean():.4f}"
            )

        all_ic = pd.concat(
            all_ic
        )

        overall[name] = all_ic.mean()

        print(
            f"Overall: {all_ic.mean():.4f}"
        )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, value in sorted(
        overall.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(
            f"{name:20s}: {value:.4f}"
        )