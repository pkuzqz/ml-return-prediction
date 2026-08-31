import numpy as np
import pandas as pd

from portfolio_backtest import (
    DATASET_PATH,
    PRICE_PATH,
    REBALANCE_DAYS,
    COST_BPS,
    make_predictions,
    run_backtest,
    performance_stats,
    portfolio_turnover,
)


def run_equal_weight_benchmark(
    predictions,
    prices,
):
    """
    Buy all available stocks with equal weights.
    Same signal dates, entry dates, holding period,
    and transaction-cost assumption as the Ridge strategy.
    """

    prediction_dates = np.array(
        sorted(
            predictions["date"].unique()
        )
    )

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

        entry_idx = signal_idx + 1
        exit_idx = entry_idx + REBALANCE_DAYS

        if exit_idx >= len(price_dates):
            break

        entry_date = price_dates[entry_idx]
        exit_date = price_dates[exit_idx]

        tickers = (
            predictions.loc[
                predictions["date"] == signal_date,
                "ticker",
            ]
            .tolist()
        )

        # Keep only stocks with usable prices.
        valid_tickers = []

        for ticker in tickers:

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

            valid_tickers.append(
                ticker
            )

        if len(valid_tickers) == 0:
            continue

        # 100% long, equal weight.
        weight = (
            1.0 / len(valid_tickers)
        )

        weights = {
            ticker: weight
            for ticker in valid_tickers
        }

        gross_return = 0.0

        for ticker in valid_tickers:

            stock_return = (
                prices.at[
                    exit_date,
                    ticker
                ]
                / prices.at[
                    entry_date,
                    ticker
                ]
                - 1
            )

            gross_return += (
                weight
                * stock_return
            )

        turnover = portfolio_turnover(
            previous_weights,
            weights
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

        records.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "gross_return": gross_return,
                "turnover": turnover,
                "cost": cost,
                "net_return": net_return,
            }
        )

        previous_weights = weights

    return pd.DataFrame(
        records
    )


if __name__ == "__main__":

    df = pd.read_csv(
        DATASET_PATH,
        parse_dates=["date"],
    ).sort_values(
        ["date", "ticker"]
    )

    price_df = pd.read_csv(
        PRICE_PATH,
        parse_dates=["date"],
    )

    prices = (
        price_df
        .pivot(
            index="date",
            columns="ticker",
            values="close",
        )
        .sort_index()
    )

    predictions = make_predictions(
        df
    )

    # --------------------------------
    # Ridge top-10% long portfolio
    # --------------------------------

    ridge_bt = run_backtest(
        predictions,
        prices
    )

    ridge_stats = performance_stats(
        ridge_bt
    )

    # --------------------------------
    # Equal-weight universe benchmark
    # --------------------------------

    benchmark_bt = (
        run_equal_weight_benchmark(
            predictions,
            prices
        )
    )

    benchmark_stats = performance_stats(
        benchmark_bt
    )

    # --------------------------------
    # Compare
    # --------------------------------

    print()
    print("=" * 70)
    print("RIDGE TOP-10% LONG VS EQUAL-WEIGHT BENCHMARK")
    print("=" * 70)

    metrics = [
        "CAGR",
        "Annual Vol",
        "Sharpe",
        "Max Drawdown",
        "Average Turnover",
        "Total Return",
    ]

    for metric in metrics:

        ridge_value = ridge_stats[
            metric
        ]

        bench_value = benchmark_stats[
            metric
        ]

        if metric == "Sharpe":
            print(
                f"{metric:18s} | "
                f"Ridge {ridge_value:7.3f} | "
                f"Benchmark {bench_value:7.3f}"
            )

        elif metric == "Average Turnover":
            print(
                f"{metric:18s} | "
                f"Ridge {ridge_value:7.3f} | "
                f"Benchmark {bench_value:7.3f}"
            )

        else:
            print(
                f"{metric:18s} | "
                f"Ridge {ridge_value:7.2%} | "
                f"Benchmark {bench_value:7.2%}"
            )

    # --------------------------------
    # Risk-adjusted comparison
    # --------------------------------

    comparison = pd.merge(
        ridge_bt[
            ["entry_date", "net_return"]
        ],
        benchmark_bt[
            ["entry_date", "net_return"]
        ],
        on="entry_date",
        suffixes=(
            "_ridge",
            "_benchmark",
        ),
    )

    ridge_r = comparison[
        "net_return_ridge"
    ]

    bench_r = comparison[
        "net_return_benchmark"
    ]

    periods_per_year = (
        252 / REBALANCE_DAYS
    )

    beta = (
        np.cov(
            ridge_r,
            bench_r,
            ddof=1,
        )[0, 1]
        / np.var(
            bench_r,
            ddof=1,
        )
    )

    alpha_period = (
        ridge_r
        - beta * bench_r
    ).mean()

    annualized_alpha = (
        (1 + alpha_period)
        ** periods_per_year
        - 1
    )

    correlation = ridge_r.corr(
        bench_r
    )

    r_squared = (
        correlation ** 2
    )

    excess = (
        ridge_r - bench_r
    )

    information_ratio = (
        excess.mean()
        / excess.std()
        * np.sqrt(periods_per_year)
    )

    print()
    print("=" * 70)
    print("RISK-ADJUSTED ANALYSIS")
    print("=" * 70)

    print(
        f"Beta vs benchmark:     "
        f"{beta:.3f}"
    )

    print(
        f"Annualized alpha:      "
        f"{annualized_alpha:.2%}"
    )

    print(
        f"Correlation:           "
        f"{correlation:.3f}"
    )

    print(
        f"R^2:                   "
        f"{r_squared:.3f}"
    )

    print(
        f"Information Ratio:     "
        f"{information_ratio:.3f}"
    )

    print(
        f"Positive excess ratio: "
        f"{(excess > 0).mean():.2%}"
    )