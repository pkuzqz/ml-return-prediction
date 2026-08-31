from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


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

TEST_START = pd.Timestamp("2022-01-03")
PURGE_DAYS = 21


def load_dataset():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    )

    return df.sort_values(
        ["date", "ticker"]
    )


def split_data(df):
    pre_test_dates = np.sort(
        df.loc[
            df["date"] < TEST_START,
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
        df["date"] >= TEST_START
    ].copy()

    return train, test


def daily_rank_ic(result):
    return result.groupby("date").apply(
        lambda g: g["prediction"].rank().corr(
            g[TARGET].rank()
        ),
        include_groups=False,
    )


def evaluate(
    name,
    predictions,
    test,
):
    result = test[
        ["date", "ticker", TARGET]
    ].copy()

    result["prediction"] = predictions

    ic = daily_rank_ic(result)

    mse = mean_squared_error(
        result[TARGET],
        predictions,
    )

    r2 = r2_score(
        result[TARGET],
        predictions,
    )

    yearly = (
        ic.groupby(ic.index.year)
        .agg(["mean", "median"])
    )

    print()
    print("=" * 65)
    print(name)
    print("=" * 65)

    print(f"MSE: {mse:.6f}")
    print(f"R^2: {r2:.6f}")
    print(f"Mean Rank IC: {ic.mean():.4f}")
    print(f"Median Rank IC: {ic.median():.4f}")
    print(
        f"Positive IC Ratio: "
        f"{(ic > 0).mean():.2%}"
    )

    print()
    print("Yearly Rank IC:")
    print(yearly.round(4))

    return {
        "model": name,
        "mse": mse,
        "r2": r2,
        "mean_ic": ic.mean(),
        "median_ic": ic.median(),
        "positive_ic": (ic > 0).mean(),
    }


if __name__ == "__main__":

    df = load_dataset()

    train, test = split_data(df)

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    results = []

    # =====================================================
    # 1. Ridge baseline
    # =====================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    ridge = Ridge(
        alpha=1.0
    )

    ridge.fit(
        X_train_scaled,
        y_train
    )

    ridge_pred = ridge.predict(
        X_test_scaled
    )

    results.append(
        evaluate(
            "Ridge",
            ridge_pred,
            test,
        )
    )

    # =====================================================
    # 2. Nonlinear model
    # =====================================================

    hgb = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )

    hgb.fit(
        X_train,
        y_train
    )

    hgb_pred = hgb.predict(
        X_test
    )

    results.append(
        evaluate(
            "HistGradientBoosting",
            hgb_pred,
            test,
        )
    )

    # =====================================================
    # Comparison
    # =====================================================

    summary = pd.DataFrame(
        results
    ).set_index("model")

    print()
    print("=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)

    print(
        summary.round(4)
    )