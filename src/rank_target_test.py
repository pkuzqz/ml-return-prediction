from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/model_dataset.csv")

FEATURES = [
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "vol_20d",
]

RAW_TARGET = "target_21d"
RANK_TARGET = "target_rank"

TEST_YEARS = [2022, 2023, 2024, 2025]
PURGE_DAYS = 21


def make_fold(df, test_year):
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year + 1}-01-01")

    pre_test_dates = np.sort(
        df.loc[df["date"] < test_start, "date"].unique()
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


def train_predict(train, test, target):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train[FEATURES]
    )

    X_test = scaler.transform(
        test[FEATURES]
    )

    model = Ridge(alpha=1.0)

    model.fit(
        X_train,
        train[target]
    )

    return model.predict(X_test)


def daily_rank_ic(test, prediction):
    result = test[
        ["date", "ticker", RAW_TARGET]
    ].copy()

    result["prediction"] = prediction

    records = []

    for date, group in result.groupby("date"):

        ic = (
            group["prediction"]
            .rank()
            .corr(
                group[RAW_TARGET].rank()
            )
        )

        records.append(
            {
                "date": date,
                "rank_ic": ic,
            }
        )

    return pd.Series(
        {
            x["date"]: x["rank_ic"]
            for x in records
        }
    )


if __name__ == "__main__":

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    ).sort_values(
        ["date", "ticker"]
    )

    # -------------------------------------------------
    # Future-return percentile rank within each date
    # -------------------------------------------------

    df[RANK_TARGET] = (
        df.groupby("date")[RAW_TARGET]
        .rank(pct=True)
    )

    raw_all = []
    rank_all = []

    for year in TEST_YEARS:

        train, test = make_fold(
            df,
            year
        )

        # =============================================
        # Model A: predict raw future return
        # =============================================

        raw_pred = train_predict(
            train,
            test,
            RAW_TARGET,
        )

        raw_ic = daily_rank_ic(
            test,
            raw_pred,
        )

        raw_all.append(raw_ic)

        # =============================================
        # Model B: predict future return rank
        # =============================================

        rank_pred = train_predict(
            train,
            test,
            RANK_TARGET,
        )

        rank_ic = daily_rank_ic(
            test,
            rank_pred,
        )

        rank_all.append(rank_ic)

        print()
        print("=" * 60)
        print(year)
        print("=" * 60)

        print(
            f"Raw-return target Mean IC: "
            f"{raw_ic.mean():.4f}"
        )

        print(
            f"Rank target Mean IC:       "
            f"{rank_ic.mean():.4f}"
        )

    raw_all = pd.concat(
        raw_all
    )

    rank_all = pd.concat(
        rank_all
    )

    print()
    print("=" * 60)
    print("OVERALL WALK-FORWARD")
    print("=" * 60)

    print()
    print("Raw-return target:")
    print(
        f"Mean IC:   {raw_all.mean():.4f}"
    )
    print(
        f"Median IC: {raw_all.median():.4f}"
    )
    print(
        f"Positive:  {(raw_all > 0).mean():.2%}"
    )

    print()
    print("Rank target:")
    print(
        f"Mean IC:   {rank_all.mean():.4f}"
    )
    print(
        f"Median IC: {rank_all.median():.4f}"
    )
    print(
        f"Positive:  {(rank_all > 0).mean():.2%}"
    )