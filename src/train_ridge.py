from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


DATA_PATH = Path("data/model_dataset.csv")

FEATURES = [
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "vol_20d",
]

TARGET = "target_21d"

TEST_START = pd.Timestamp("2022-01-03")
PURGE_DAYS = 21

RIDGE_ALPHA = 1.0


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=["date"]
    )

    df = df.sort_values(
        ["date", "ticker"]
    )

    return df


def time_split_with_purge(
    df: pd.DataFrame,
    test_start: pd.Timestamp,
    purge_days: int
):
    # All trading dates strictly before the test period.
    pre_test_dates = np.sort(
        df.loc[
            df["date"] < test_start,
            "date"
        ].unique()
    )

    # Remove the final 21 training dates so their
    # future-return labels do not overlap the test period.
    purge_dates = pre_test_dates[
        -purge_days:
    ]

    purge_start = pd.Timestamp(
        purge_dates[0]
    )

    train = df[
        df["date"] < purge_start
    ].copy()

    test = df[
        df["date"] >= test_start
    ].copy()

    return train, test, purge_start


def compute_daily_rank_ic(
    df: pd.DataFrame
) -> pd.Series:
    records = {}

    for date, group in df.groupby("date"):
        if len(group) < 5:
            continue

        pred_rank = group[
            "prediction"
        ].rank()

        target_rank = group[
            TARGET
        ].rank()

        ic = pred_rank.corr(
            target_rank
        )

        records[date] = ic

    return pd.Series(
        records,
        name="rank_ic"
    )


if __name__ == "__main__":
    # --------------------------------
    # 1. Load data
    # --------------------------------

    df = load_dataset(
        DATA_PATH
    )

    # --------------------------------
    # 2. Time-based train/test split
    #    with a 21-day purge gap
    # --------------------------------

    train, test, purge_start = (
        time_split_with_purge(
            df,
            TEST_START,
            PURGE_DAYS
        )
    )

    print("=" * 60)
    print("DATA SPLIT")
    print("=" * 60)

    print(
        "Train:",
        train["date"].min(),
        "to",
        train["date"].max()
    )

    print(
        "Purge starts:",
        purge_start
    )

    print(
        "Test:",
        test["date"].min(),
        "to",
        test["date"].max()
    )

    print(
        "Train samples:",
        len(train)
    )

    print(
        "Test samples:",
        len(test)
    )

    # --------------------------------
    # 3. Build X and y
    # --------------------------------

    X_train = train[
        FEATURES
    ]

    y_train = train[
        TARGET
    ]

    X_test = test[
        FEATURES
    ]

    y_test = test[
        TARGET
    ]

    # --------------------------------
    # 4. Standardize features
    # --------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    # --------------------------------
    # 5. Train Ridge
    # --------------------------------

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # --------------------------------
    # 6. Predict on unseen test data
    # --------------------------------

    predictions = model.predict(
        X_test_scaled
    )

    result = test[
        ["date", "ticker", TARGET]
    ].copy()

    result["prediction"] = (
        predictions
    )

    # --------------------------------
    # 7. Standard regression metrics
    # --------------------------------

    mse = mean_squared_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # --------------------------------
    # 8. Quant metric: daily Rank IC
    # --------------------------------

    daily_ic = compute_daily_rank_ic(
        result
    )
    yearly_ic = (
        daily_ic
        .groupby(daily_ic.index.year)
        .agg(["mean", "median", "count"])
    )

    print()
    print("=" * 60)
    print("YEARLY RANK IC")
    print("=" * 60)

    print(
        yearly_ic.round(4)
    )

    print()
    print("=" * 60)
    print("RIDGE OUT-OF-SAMPLE RESULTS")
    print("=" * 60)

    print(
        f"MSE: {mse:.6f}"
    )

    print(
        f"R^2: {r2:.6f}"
    )

    print(
        f"Mean Rank IC: "
        f"{daily_ic.mean():.4f}"
    )

    print(
        f"Median Rank IC: "
        f"{daily_ic.median():.4f}"
    )

    print(
        f"Positive IC Ratio: "
        f"{(daily_ic > 0).mean():.2%}"
    )

    # --------------------------------
    # 9. Inspect learned coefficients
    # --------------------------------

    coefficients = pd.Series(
        model.coef_,
        index=FEATURES
    ).sort_values(
        key=np.abs,
        ascending=False
    )

    print()
    print("=" * 60)
    print("RIDGE COEFFICIENTS")
    print("=" * 60)

    print(
        coefficients.round(6)
    )
    print()
    print("=" * 60)
    print("FEATURE ABLATION")
    print("=" * 60)

    feature_sets = {
        "ret_5d": ["ret_5d"],
        "ret_21d": ["ret_21d"],
        "ret_63d": ["ret_63d"],
        "vol_20d": ["vol_20d"],
        "all_features": FEATURES,
    }

    for name, feature_list in feature_sets.items():

        X_train_ab = train[feature_list]
        X_test_ab = test[feature_list]

        scaler_ab = StandardScaler()

        X_train_ab_scaled = scaler_ab.fit_transform(
            X_train_ab
        )

        X_test_ab_scaled = scaler_ab.transform(
            X_test_ab
        )

        model_ab = Ridge(
            alpha=RIDGE_ALPHA
        )

        model_ab.fit(
            X_train_ab_scaled,
            y_train
        )

        pred_ab = model_ab.predict(
            X_test_ab_scaled
        )

        result_ab = test[
            ["date", "ticker", TARGET]
        ].copy()

        result_ab["prediction"] = pred_ab

        daily_ic_ab = compute_daily_rank_ic(
            result_ab
        )

        print(
            f"{name:15s} "
            f"Mean IC = {daily_ic_ab.mean(): .4f} | "
            f"Median IC = {daily_ic_ab.median(): .4f} | "
            f"Positive = {(daily_ic_ab > 0).mean():.2%}"
        )