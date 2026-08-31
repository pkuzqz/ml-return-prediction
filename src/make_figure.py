from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from portfolio_backtest import (
    DATASET_PATH,
    PRICE_PATH,
    make_predictions,
    run_backtest,
)

from benchmark_compare import (
    run_equal_weight_benchmark,
)

from feature_group_test import (
    make_fold,
    run_ridge,
    mean_rank_ic,
)


FIGURE_DIR = Path("figures")

VOL_FEATURES = [
    "vol_5d",
    "vol_20d",
    "vol_63d",
]

TEST_YEARS = [
    2022,
    2023,
    2024,
    2025,
]


def load_data():

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

    return df, prices


def make_equity_curve(
    df,
    prices,
):

    # =====================================================
    # 1. Generate Ridge predictions
    # =====================================================

    predictions = make_predictions(
        df
    )

    # =====================================================
    # 2. Run Ridge Top-10% long portfolio
    # =====================================================

    ridge_bt = run_backtest(
        predictions,
        prices,
    )

    # =====================================================
    # 3. Run equal-weight benchmark
    # =====================================================

    benchmark_bt = (
        run_equal_weight_benchmark(
            predictions,
            prices,
        )
    )

    # =====================================================
    # 4. Calculate cumulative portfolio values
    # =====================================================

    ridge_equity = (
        1 + ridge_bt["net_return"]
    ).cumprod()

    benchmark_equity = (
        1 + benchmark_bt["net_return"]
    ).cumprod()

    # Add starting point:
    # $1 at the first entry date.

    initial_date = (
        ridge_bt["entry_date"]
        .iloc[0]
    )

    ridge_dates = pd.concat(
        [
            pd.Series(
                [initial_date]
            ),
            ridge_bt[
                "exit_date"
            ].reset_index(
                drop=True
            ),
        ],
        ignore_index=True,
    )

    benchmark_dates = pd.concat(
        [
            pd.Series(
                [initial_date]
            ),
            benchmark_bt[
                "exit_date"
            ].reset_index(
                drop=True
            ),
        ],
        ignore_index=True,
    )

    ridge_equity = pd.concat(
        [
            pd.Series([1.0]),
            ridge_equity.reset_index(
                drop=True
            ),
        ],
        ignore_index=True,
    )

    benchmark_equity = pd.concat(
        [
            pd.Series([1.0]),
            benchmark_equity.reset_index(
                drop=True
            ),
        ],
        ignore_index=True,
    )

    # =====================================================
    # 5. Plot
    # =====================================================

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        ridge_dates,
        ridge_equity,
        label="Ridge Top-10% Long",
    )

    plt.plot(
        benchmark_dates,
        benchmark_equity,
        label="Equal-Weight Benchmark",
    )

    plt.axhline(
        1.0,
        linewidth=1,
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Growth of $1"
    )

    plt.title(
        "Cumulative Net Value: "
        "Ridge Top-10% vs Equal-Weight Benchmark"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output = (
        FIGURE_DIR
        / "equity_curve.png"
    )

    plt.savefig(
        output,
        dpi=180,
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


def make_yearly_ic(
    df,
):

    yearly_ic = {}

    # =====================================================
    # Purged expanding-window evaluation
    # =====================================================

    for year in TEST_YEARS:

        train, test = make_fold(
            df,
            year,
        )

        predictions = run_ridge(
            train,
            test,
            VOL_FEATURES,
        )

        ic = mean_rank_ic(
            test,
            predictions,
        )

        yearly_ic[
            year
        ] = ic.mean()

    years = list(
        yearly_ic.keys()
    )

    values = list(
        yearly_ic.values()
    )

    # =====================================================
    # Plot yearly Mean Rank IC
    # =====================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        years,
        values,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Test Year"
    )

    plt.ylabel(
        "Mean Rank IC"
    )

    plt.title(
        "Purged Walk-Forward Rank IC by Year"
    )

    plt.xticks(
        years
    )

    # Show exact IC values on bars.

    for year, value in zip(
        years,
        values,
    ):

        plt.text(
            year,
            value,
            f"{value:.3f}",
            ha="center",
            va=(
                "bottom"
                if value >= 0
                else "top"
            ),
        )

    plt.tight_layout()

    output = (
        FIGURE_DIR
        / "yearly_rank_ic.png"
    )

    plt.savefig(
        output,
        dpi=180,
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":

    FIGURE_DIR.mkdir(
        exist_ok=True
    )

    df, prices = load_data()

    make_equity_curve(
        df,
        prices,
    )

    make_yearly_ic(
        df,
    )