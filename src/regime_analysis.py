from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


DATA_PATH = Path("data/model_dataset.csv")

FEATURES = [
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "vol_20d",
]

TARGET = "target_21d"

TEST_YEARS = [2022, 2023, 2024, 2025]

PURGE_DAYS = 21
N_REGIMES = 3


def make_fold(df, test_year):
    test_start = pd.Timestamp(
        f"{test_year}-01-01"
    )

    test_end = pd.Timestamp(
        f"{test_year + 1}-01-01"
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


def build_market_features(df):
    """
    Convert stock-level rows into one market-level row per date.
    """

    daily = df.groupby("date").agg(
        market_ret_21d=(
            "ret_21d",
            "mean"
        ),

        market_vol_20d=(
            "vol_20d",
            "mean"
        ),

        dispersion_21d=(
            "ret_21d",
            "std"
        ),
    )

    return daily.dropna()


def fit_regime_model(
    train_market,
    test_market
):
    regime_features = [
        "market_ret_21d",
        "market_vol_20d",
        "dispersion_21d",
    ]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train_market[regime_features]
    )

    X_test = scaler.transform(
        test_market[regime_features]
    )

    gmm = GaussianMixture(
        n_components=N_REGIMES,
        covariance_type="full",
        random_state=42,
    )

    gmm.fit(X_train)

    test_cluster = gmm.predict(
        X_test
    )

    # Convert cluster centers back to original units
    centers = pd.DataFrame(
        scaler.inverse_transform(
            gmm.means_
        ),
        columns=regime_features,
    )

    # GMM cluster numbers are arbitrary.
    # Sort them by volatility so labels become interpretable.
    ordered_clusters = (
        centers["market_vol_20d"]
        .sort_values()
        .index
        .tolist()
    )

    regime_names = [
        "low_vol",
        "mid_vol",
        "high_vol",
    ]

    cluster_to_name = {
        cluster: name
        for cluster, name
        in zip(
            ordered_clusters,
            regime_names
        )
    }

    test_regimes = pd.Series(
        test_cluster,
        index=test_market.index,
    ).map(cluster_to_name)

    centers["regime"] = centers.index.map(
        cluster_to_name
    )

    centers = (
        centers
        .set_index("regime")
        .loc[regime_names]
    )

    return test_regimes, centers


def run_ridge(train, test):
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

    return model.predict(
        X_test
    )


def compute_daily_ic(test, predictions):
    result = test[
        ["date", "ticker", TARGET]
    ].copy()

    result["prediction"] = predictions

    records = []

    for date, group in result.groupby("date"):
        ic = (
            group["prediction"]
            .rank()
            .corr(
                group[TARGET].rank()
            )
        )

        records.append(
            {
                "date": date,
                "rank_ic": ic,
            }
        )

    return pd.DataFrame(
        records
    ).set_index("date")


if __name__ == "__main__":

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    ).sort_values(
        ["date", "ticker"]
    )

    all_ic = []

    for year in TEST_YEARS:

        train, test = make_fold(
            df,
            year
        )

        # ----------------------------
        # 1. Build market state data
        # ----------------------------

        train_market = build_market_features(
            train
        )

        test_market = build_market_features(
            test
        )

        # ----------------------------
        # 2. Fit GMM on TRAIN only
        # ----------------------------

        regimes, centers = fit_regime_model(
            train_market,
            test_market
        )

        # ----------------------------
        # 3. Fit Ridge and predict test
        # ----------------------------

        predictions = run_ridge(
            train,
            test
        )

        daily_ic = compute_daily_ic(
            test,
            predictions
        )

        daily_ic["regime"] = regimes

        daily_ic["year"] = year

        all_ic.append(
            daily_ic
        )

        print()
        print("=" * 70)
        print(f"{year} GMM REGIME CENTERS")
        print("=" * 70)

        print(
            centers.round(4)
        )

        print()
        print("Ridge Rank IC by regime:")

        print(
            daily_ic
            .groupby("regime")["rank_ic"]
            .agg(
                count="count",
                mean="mean",
                median="median",
            )
            .round(4)
        )

    # =====================================================
    # Overall regime analysis
    # =====================================================

    combined = pd.concat(
        all_ic
    )

    summary = (
        combined
        .groupby("regime")["rank_ic"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            positive_ratio=lambda x: (
                x > 0
            ).mean(),
        )
    )

    print()
    print("=" * 70)
    print("OVERALL RIDGE PERFORMANCE BY REGIME")
    print("=" * 70)

    print(
        summary.round(4)
    )