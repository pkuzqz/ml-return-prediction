from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor


DATA_PATH = Path("data/model_dataset.csv")

FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "ret_126d",
    "momentum_126_5",
    "vol_5d",
    "vol_20d",
    "vol_63d",
    "drawdown_63d",
    "ma_ratio_5_21",
    "ma_ratio_21_63",
]

TARGET = "target_21d"
PURGE_DAYS = 21

TEST_YEARS = [
    2022,
    2023,
    2024,
    2025,
]


def daily_rank_ic(df):
    return df.groupby("date").apply(
        lambda g: g["prediction"].rank().corr(
            g[TARGET].rank()
        ),
        include_groups=False,
    )


def make_fold(df, test_year):
    test_start = pd.Timestamp(
        f"{test_year}-01-01"
    )

    test_end = pd.Timestamp(
        f"{test_year + 1}-01-01"
    )

    # All trading dates before this test year
    pre_test_dates = np.sort(
        df.loc[
            df["date"] < test_start,
            "date"
        ].unique()
    )

    # Remove final 21 trading dates
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

    return train, test, purge_start


def run_ridge(train, test):
    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = Ridge(alpha=1.0)

    model.fit(
        X_train_scaled,
        y_train
    )

    return model.predict(
        X_test_scaled
    )


def run_hgb(train, test):
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(
        train[FEATURES],
        train[TARGET],
    )

    return model.predict(
        test[FEATURES]
    )


if __name__ == "__main__":

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    ).sort_values(
        ["date", "ticker"]
    )

    all_results = []

    for year in TEST_YEARS:

        train, test, purge_start = make_fold(
            df,
            year
        )

        print()
        print("=" * 65)
        print(f"WALK-FORWARD YEAR: {year}")
        print("=" * 65)

        print(
            "Train:",
            train["date"].min(),
            "to",
            train["date"].max(),
        )

        print(
            "Purge starts:",
            purge_start
        )

        print(
            "Test:",
            test["date"].min(),
            "to",
            test["date"].max(),
        )

        # -------------------------
        # Ridge
        # -------------------------

        ridge_pred = run_ridge(
            train,
            test
        )

        ridge_result = test[
            ["date", "ticker", TARGET]
        ].copy()

        ridge_result[
            "prediction"
        ] = ridge_pred

        ridge_result[
            "model"
        ] = "Ridge"

        all_results.append(
            ridge_result
        )

        ridge_ic = daily_rank_ic(
            ridge_result
        )

        print(
            f"Ridge Mean IC: "
            f"{ridge_ic.mean():.4f}"
        )

        # -------------------------
        # HGB
        # -------------------------

        hgb_pred = run_hgb(
            train,
            test
        )

        hgb_result = test[
            ["date", "ticker", TARGET]
        ].copy()

        hgb_result[
            "prediction"
        ] = hgb_pred

        hgb_result[
            "model"
        ] = "HistGradientBoosting"

        all_results.append(
            hgb_result
        )

        hgb_ic = daily_rank_ic(
            hgb_result
        )

        print(
            f"HGB Mean IC: "
            f"{hgb_ic.mean():.4f}"
        )

    # =====================================================
    # Overall walk-forward comparison
    # =====================================================

    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    print()
    print("=" * 65)
    print("OVERALL WALK-FORWARD RESULTS")
    print("=" * 65)

    for model_name, group in combined.groupby(
        "model"
    ):
        ic = daily_rank_ic(
            group
        )

        print()
        print(model_name)

        print(
            f"Mean Rank IC: "
            f"{ic.mean():.4f}"
        )

        print(
            f"Median Rank IC: "
            f"{ic.median():.4f}"
        )

        print(
            f"Positive IC Ratio: "
            f"{(ic > 0).mean():.2%}"
        )