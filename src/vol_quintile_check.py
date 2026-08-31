from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/model_dataset.csv")

VOL_FEATURE = "vol_63d"
TARGET = "target_21d"

TEST_START = pd.Timestamp("2022-01-01")


if __name__ == "__main__":

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    )

    # Only inspect the out-of-sample period.
    test = df[
        df["date"] >= TEST_START
    ].copy()

    # Cross-sectional volatility rank
    # within each date.
    test["vol_rank"] = (
        test.groupby("date")[VOL_FEATURE]
        .rank(
            pct=True,
            method="first"
        )
    )

    # Convert percentile rank into quintiles 1,...,5.
    test["quintile"] = np.ceil(
        test["vol_rank"] * 5
    ).astype(int)

    test["quintile"] = (
        test["quintile"]
        .clip(1, 5)
    )

    # First average stock returns within
    # each date/quintile.
    daily_quintile_returns = (
        test.groupby(
            ["date", "quintile"]
        )[TARGET]
        .mean()
        .unstack()
    )

    # Then average across dates.
    overall = (
        daily_quintile_returns
        .mean()
    )

    print("=" * 60)
    print("VOLATILITY QUINTILE TEST")
    print("=" * 60)

    for q in range(1, 6):
        print(
            f"Q{q}: "
            f"{overall[q]:.4%}"
        )

    spread = (
        daily_quintile_returns[5]
        - daily_quintile_returns[1]
    )

    print()
    print(
        f"Q5 - Q1 average spread: "
        f"{spread.mean():.4%}"
    )

    print(
        f"Positive spread ratio: "
        f"{(spread > 0).mean():.2%}"
    )

    print()
    print("=" * 60)
    print("YEARLY Q5 - Q1 SPREAD")
    print("=" * 60)

    yearly = spread.groupby(
        spread.index.year
    ).mean()

    for year, value in yearly.items():
        print(
            f"{year}: {value:.4%}"
        )