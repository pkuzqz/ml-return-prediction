from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


DATASET_PATH = Path("data/model_dataset.csv")
PRICE_PATH = Path("data/prices.csv")

FEATURES = [
    "vol_5d",
    "vol_20d",
    "vol_63d",
]

TARGET = "target_21d"

TEST_YEARS = [2022, 2023, 2024, 2025]

PURGE_DAYS = 21
REBALANCE_DAYS = 21

TOP_FRAC = 0.10
BOTTOM_FRAC = 0.10

LONG_GROSS = 1.00
SHORT_GROSS = 0.00

COST_BPS = 5


def make_fold(df, year):
    test_start = pd.Timestamp(f"{year}-01-01")
    test_end = pd.Timestamp(f"{year + 1}-01-01")

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


def make_predictions(df):
    all_predictions = []

    for year in TEST_YEARS:
        train, test = make_fold(
            df,
            year
        )

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
            train[TARGET]
        )

        prediction = model.predict(
            X_test
        )

        out = test[
            ["date", "ticker"]
        ].copy()

        out["prediction"] = prediction

        all_predictions.append(out)

    return pd.concat(
        all_predictions,
        ignore_index=True
    )


def build_weights(group):
    group = group.sort_values(
        "prediction"
    )

    n = len(group)

    n_long = max(
        1,
        int(n * TOP_FRAC)
    )

    n_short = max(
        1,
        int(n * BOTTOM_FRAC)
    )

    short_names = (
        group.head(n_short)["ticker"]
        .tolist()
    )

    long_names = (
        group.tail(n_long)["ticker"]
        .tolist()
    )

    weights = {}

    for ticker in long_names:
        weights[ticker] = (
            LONG_GROSS / n_long
        )

    for ticker in short_names:
        weights[ticker] = (
            -SHORT_GROSS / n_short
        )

    return weights


def portfolio_turnover(
    old_weights,
    new_weights
):
    tickers = (
        set(old_weights)
        | set(new_weights)
    )

    turnover = sum(
        abs(
            new_weights.get(t, 0.0)
            - old_weights.get(t, 0.0)
        )
        for t in tickers
    )

    return turnover


def run_backtest(
    predictions,
    prices
):
    prediction_dates = np.array(
        sorted(
            predictions["date"].unique()
        )
    )

    # One signal every 21 trading days.
    signal_dates = prediction_dates[
        ::REBALANCE_DAYS
    ]

    price_dates = prices.index

    previous_weights = {}
    records = []

    for signal_date in signal_dates:

        signal_date = pd.Timestamp(
            signal_date
        )

        signal_idx = price_dates.searchsorted(
            signal_date
        )

        if signal_idx >= len(price_dates):
            continue

        # Observe signal at day t,
        # enter on next trading day.
        entry_idx = signal_idx + 1

        # Hold for 21 trading days.
        exit_idx = (
            entry_idx
            + REBALANCE_DAYS
        )

        if exit_idx >= len(price_dates):
            break

        entry_date = price_dates[
            entry_idx
        ]

        exit_date = price_dates[
            exit_idx
        ]

        day_predictions = predictions[
            predictions["date"]
            == signal_date
        ].copy()

        if len(day_predictions) == 0:
            continue

        weights = build_weights(
            day_predictions
        )

        # -----------------------------------------
        # Calculate ONE portfolio-period return
        # -----------------------------------------

        gross_return = 0.0
        long_contribution = 0.0
        short_contribution = 0.0

        valid_weights = {}

        for ticker, weight in weights.items():

            if ticker not in prices.columns:
                continue

            entry_price = prices.at[
                entry_date,
                ticker
            ]

            exit_price = prices.at[
                exit_date,
                ticker
            ]

            if (
                pd.isna(entry_price)
                or pd.isna(exit_price)
                or entry_price <= 0
            ):
                continue

            stock_return = (
                exit_price
                / entry_price
                - 1
            )

            contribution = (
                weight
                * stock_return
            )

            gross_return += contribution

            if weight > 0:
                long_contribution += (
                    contribution
                )

            elif weight < 0:
                short_contribution += (
                    contribution
                )

            valid_weights[ticker] = weight

        # -----------------------------------------
        # Only AFTER all stocks have been processed
        # calculate portfolio turnover and cost
        # -----------------------------------------

        turnover = portfolio_turnover(
            previous_weights,
            valid_weights
        )

        cost = (
            turnover
            * COST_BPS
            / 10000
        )

        net_return = (
            gross_return
            - cost
        )

        # ONE record = ONE rebalance period
        records.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "long_contribution": long_contribution,
                "short_contribution": short_contribution,
                "gross_return": gross_return,
                "turnover": turnover,
                "cost": cost,
                "net_return": net_return,
            }
        )

        previous_weights = (
            valid_weights
        )

    return pd.DataFrame(
        records
    )

def performance_stats(backtest):
    returns = backtest[
        "net_return"
    ]

    periods_per_year = (
        252 / REBALANCE_DAYS
    )

    equity = (
        1 + returns
    ).cumprod()

    years = (
        len(returns)
        / periods_per_year
    )

    cagr = (
        equity.iloc[-1] ** (1 / years)
        - 1
    )

    annual_vol = (
        returns.std()
        * np.sqrt(periods_per_year)
    )

    sharpe = (
        returns.mean()
        / returns.std()
        * np.sqrt(periods_per_year)
    )

    running_max = equity.cummax()

    drawdown = (
        equity / running_max - 1
    )

    max_drawdown = (
        drawdown.min()
    )

    return {
        "CAGR": cagr,
        "Annual Vol": annual_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
        "Average Turnover": backtest[
            "turnover"
        ].mean(),
        "Total Return": (
            equity.iloc[-1] - 1
        ),
    }


if __name__ == "__main__":

    df = pd.read_csv(
        DATASET_PATH,
        parse_dates=["date"]
    ).sort_values(
        ["date", "ticker"]
    )

    price_df = pd.read_csv(
        PRICE_PATH,
        parse_dates=["date"]
    )

    prices = (
        price_df
        .pivot(
            index="date",
            columns="ticker",
            values="close"
        )
        .sort_index()
    )

    predictions = make_predictions(
        df
    )

    backtest = run_backtest(
        predictions,
        prices
    )

    stats = performance_stats(
        backtest
    )

    print("=" * 65)
    print("RIDGE LONG-SHORT PORTFOLIO")
    print("=" * 65)

    print(
        "Periods:",
        len(backtest)
    )

    for name, value in stats.items():

        if name == "Average Turnover":
            print(
                f"{name}: {value:.4f}"
            )

        elif name == "Sharpe":
            print(
                f"{name}: {value:.4f}"
            )

        else:
            print(
                f"{name}: {value:.2%}"
            )

    print()
    print("First 5 periods:")
    print(
        backtest.head()
    )
    print()
    print("=" * 65)
    print("LONG / SHORT DECOMPOSITION")
    print("=" * 65)

    print(
        "Average long contribution:",
        f"{backtest['long_contribution'].mean():.2%}"
    )

    print(
        "Average short contribution:",
        f"{backtest['short_contribution'].mean():.2%}"
    )

    print(
        "Average gross return:",
        f"{backtest['gross_return'].mean():.2%}"
    )


    print()
    print("=" * 65)
    print("YEARLY PERFORMANCE")
    print("=" * 65)

    backtest["year"] = (
        backtest["entry_date"].dt.year
    )

    for year, group in backtest.groupby("year"):

        yearly_return = (
            (1 + group["net_return"])
            .prod()
            - 1
        )

        long_total = group[
            "long_contribution"
        ].sum()

        short_total = group[
            "short_contribution"
        ].sum()

        print()
        print(year)

        print(
            f"Net return: "
            f"{yearly_return:.2%}"
        )

        print(
            f"Long contribution sum: "
            f"{long_total:.2%}"
        )

        print(
            f"Short contribution sum: "
            f"{short_total:.2%}"
        )
    print()
    print("=" * 65)
    print("TRANSACTION COST SENSITIVITY")
    print("=" * 65)

    for cost_bps in [0, 5, 10, 20]:

        temp = backtest.copy()

        temp["net_return_test"] = (
            temp["gross_return"]
            - temp["turnover"]
            * cost_bps
            / 10000
        )

        returns = temp["net_return_test"]

        periods_per_year = (
            252 / REBALANCE_DAYS
        )

        equity = (
            1 + returns
        ).cumprod()

        years = (
            len(returns)
            / periods_per_year
        )

        cagr = (
            equity.iloc[-1] ** (1 / years)
            - 1
        )

        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(periods_per_year)
        )

        print(
            f"{cost_bps:2d} bps | "
            f"CAGR = {cagr:.2%} | "
            f"Sharpe = {sharpe:.2f}"
        )