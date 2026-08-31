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

REGIME_FEATURES = [
    "market_ret_21d",
    "market_vol_20d",
    "dispersion_21d",
]

TARGET = "target_21d"

TEST_YEARS = [
    2022,
    2023,
    2024,
    2025,
]

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

    return (
        df.groupby("date")
        .agg(
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
        .dropna()
    )


def assign_regimes(
    train_market,
    test_market
):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train_market[REGIME_FEATURES]
    )

    X_test = scaler.transform(
        test_market[REGIME_FEATURES]
    )

    gmm = GaussianMixture(
        n_components=N_REGIMES,
        covariance_type="full",
        random_state=42,
    )

    gmm.fit(X_train)

    train_cluster = gmm.predict(
        X_train
    )

    test_cluster = gmm.predict(
        X_test
    )

    # Convert cluster centers back
    # to original units.
    centers = pd.DataFrame(
        scaler.inverse_transform(
            gmm.means_
        ),
        columns=REGIME_FEATURES,
    )

    # GMM label 0/1/2 itself has
    # no meaning, so sort by volatility.
    ordered_clusters = (
        centers["market_vol_20d"]
        .sort_values()
        .index
        .tolist()
    )

    names = [
        "low_vol",
        "mid_vol",
        "high_vol",
    ]

    mapping = dict(
        zip(
            ordered_clusters,
            names
        )
    )

    train_regime = pd.Series(
        train_cluster,
        index=train_market.index,
    ).map(mapping)

    test_regime = pd.Series(
        test_cluster,
        index=test_market.index,
    ).map(mapping)

    return train_regime, test_regime


def attach_regime(
    df,
    regime_series
):

    result = df.copy()

    result["regime"] = (
        result["date"]
        .map(regime_series)
    )

    return result.dropna(
        subset=["regime"]
    )


def make_regime_design(
    train,
    test
):
    """
    Build:
        base features
        regime indicators
        feature x regime interactions
    """

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train[FEATURES]
    )

    X_test = scaler.transform(
        test[FEATURES]
    )

    train_mid = (
        train["regime"]
        .eq("mid_vol")
        .astype(float)
        .to_numpy()
        .reshape(-1, 1)
    )

    train_high = (
        train["regime"]
        .eq("high_vol")
        .astype(float)
        .to_numpy()
        .reshape(-1, 1)
    )

    test_mid = (
        test["regime"]
        .eq("mid_vol")
        .astype(float)
        .to_numpy()
        .reshape(-1, 1)
    )

    test_high = (
        test["regime"]
        .eq("high_vol")
        .astype(float)
        .to_numpy()
        .reshape(-1, 1)
    )

    X_train_regime = np.hstack(
        [
            X_train,

            train_mid,
            train_high,

            X_train * train_mid,
            X_train * train_high,
        ]
    )

    X_test_regime = np.hstack(
        [
            X_test,

            test_mid,
            test_high,

            X_test * test_mid,
            X_test * test_high,
        ]
    )

    return (
        X_train_regime,
        X_test_regime,
        scaler,
    )


def daily_rank_ic(
    df
):

    return df.groupby("date").apply(
        lambda g:
        g["prediction"]
        .rank()
        .corr(
            g[TARGET].rank()
        ),
        include_groups=False,
    )


def evaluate(
    test,
    predictions
):

    result = test[
        [
            "date",
            "ticker",
            TARGET,
            "regime",
        ]
    ].copy()

    result["prediction"] = (
        predictions
    )

    return daily_rank_ic(
        result
    )


if __name__ == "__main__":

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    ).sort_values(
        ["date", "ticker"]
    )

    baseline_all = []
    regime_all = []

    for year in TEST_YEARS:

        train, test = make_fold(
            df,
            year
        )

        # =================================================
        # 1. GMM regime detection
        # =================================================

        train_market = (
            build_market_features(train)
        )

        test_market = (
            build_market_features(test)
        )

        train_regime, test_regime = (
            assign_regimes(
                train_market,
                test_market,
            )
        )

        train = attach_regime(
            train,
            train_regime,
        )

        test = attach_regime(
            test,
            test_regime,
        )

        # =================================================
        # 2. Baseline Ridge
        # =================================================

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            train[FEATURES]
        )

        X_test = scaler.transform(
            test[FEATURES]
        )

        baseline = Ridge(
            alpha=1.0
        )

        baseline.fit(
            X_train,
            train[TARGET]
        )

        baseline_pred = baseline.predict(
            X_test
        )

        baseline_ic = evaluate(
            test,
            baseline_pred
        )

        baseline_all.append(
            baseline_ic
        )

        # =================================================
        # 3. Regime-aware Ridge
        # =================================================

        (
            X_train_regime,
            X_test_regime,
            _
        ) = make_regime_design(
            train,
            test
        )

        regime_model = Ridge(
            alpha=1.0
        )

        regime_model.fit(
            X_train_regime,
            train[TARGET]
        )

        regime_pred = regime_model.predict(
            X_test_regime
        )

        regime_ic = evaluate(
            test,
            regime_pred
        )

        regime_all.append(
            regime_ic
        )

        print()
        print("=" * 65)
        print(year)
        print("=" * 65)

        print(
            f"Baseline Ridge Mean IC: "
            f"{baseline_ic.mean():.4f}"
        )

        print(
            f"Regime-aware Mean IC:   "
            f"{regime_ic.mean():.4f}"
        )

    # =====================================================
    # Overall
    # =====================================================

    baseline_all = pd.concat(
        baseline_all
    )

    regime_all = pd.concat(
        regime_all
    )

    print()
    print("=" * 65)
    print("OVERALL")
    print("=" * 65)

    print(
        "Baseline Ridge"
    )

    print(
        f"Mean IC:   "
        f"{baseline_all.mean():.4f}"
    )

    print(
        f"Median IC: "
        f"{baseline_all.median():.4f}"
    )

    print(
        f"Positive:  "
        f"{(baseline_all > 0).mean():.2%}"
    )

    print()

    print(
        "Regime-aware Ridge"
    )

    print(
        f"Mean IC:   "
        f"{regime_all.mean():.4f}"
    )

    print(
        f"Median IC: "
        f"{regime_all.median():.4f}"
    )

    print(
        f"Positive:  "
        f"{(regime_all > 0).mean():.2%}"
    )