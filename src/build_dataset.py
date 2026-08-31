from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/prices.csv")
OUTPUT_PATH = Path("data/model_dataset.csv")


def load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=["date"]
    )

    prices = df.pivot(
        index="date",
        columns="ticker",
        values="close"
    )

    return prices.sort_index()


def build_dataset(prices: pd.DataFrame) -> pd.DataFrame:

    # =====================================================
    # 1. Return / momentum features
    # =====================================================

    ret_1d = prices / prices.shift(1) - 1

    ret_5d = prices / prices.shift(5) - 1

    ret_21d = prices / prices.shift(21) - 1

    ret_63d = prices / prices.shift(63) - 1

    ret_126d = prices / prices.shift(126) - 1

    # 6-month momentum, but skip the most recent 5 days
    momentum_126_5 = (
        prices.shift(5)
        / prices.shift(126)
        - 1
    )

    # =====================================================
    # 2. Volatility features
    # =====================================================

    daily_returns = prices.pct_change(
        fill_method=None
    )

    vol_5d = (
        daily_returns
        .rolling(5)
        .std()
    )

    vol_20d = (
        daily_returns
        .rolling(20)
        .std()
    )

    vol_63d = (
        daily_returns
        .rolling(63)
        .std()
    )

    # =====================================================
    # 3. Trend / position features
    # =====================================================

    # Distance from 63-day high.
    rolling_high_63 = (
        prices
        .rolling(63)
        .max()
    )

    drawdown_63d = (
        prices
        / rolling_high_63
        - 1
    )

    # Moving averages
    ma_5 = prices.rolling(5).mean()
    ma_21 = prices.rolling(21).mean()
    ma_63 = prices.rolling(63).mean()

    ma_ratio_5_21 = (
        ma_5 / ma_21 - 1
    )

    ma_ratio_21_63 = (
        ma_21 / ma_63 - 1
    )

    # =====================================================
    # 4. Future 21-day return target
    # =====================================================

    target_21d = (
        prices.shift(-21)
        / prices
        - 1
    )

    # =====================================================
    # 5. Convert wide price matrices into panel dataset
    # =====================================================

    dataset = pd.concat(
        [
            ret_1d.stack().rename("ret_1d"),
            ret_5d.stack().rename("ret_5d"),
            ret_21d.stack().rename("ret_21d"),
            ret_63d.stack().rename("ret_63d"),
            ret_126d.stack().rename("ret_126d"),

            momentum_126_5.stack().rename(
                "momentum_126_5"
            ),

            vol_5d.stack().rename("vol_5d"),
            vol_20d.stack().rename("vol_20d"),
            vol_63d.stack().rename("vol_63d"),

            drawdown_63d.stack().rename(
                "drawdown_63d"
            ),

            ma_ratio_5_21.stack().rename(
                "ma_ratio_5_21"
            ),

            ma_ratio_21_63.stack().rename(
                "ma_ratio_21_63"
            ),

            target_21d.stack().rename(
                "target_21d"
            ),
        ],
        axis=1
    )

    dataset.index.names = [
        "date",
        "ticker"
    ]

    dataset = (
        dataset
        .dropna()
        .reset_index()
    )

    return dataset


if __name__ == "__main__":

    prices = load_prices(
        DATA_PATH
    )

    dataset = build_dataset(
        prices
    )

    print(dataset.head())

    print()
    print(
        "Shape:",
        dataset.shape
    )

    print()
    print("Date range:")
    print(
        dataset["date"].min(),
        "to",
        dataset["date"].max()
    )

    print()
    print(
        "Number of stocks:",
        dataset["ticker"].nunique()
    )

    print()
    print("Features:")

    for column in dataset.columns:
        if column not in [
            "date",
            "ticker",
            "target_21d",
        ]:
            print(
                " -",
                column
            )

    dataset.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print(
        f"Saved dataset to {OUTPUT_PATH}"
    )