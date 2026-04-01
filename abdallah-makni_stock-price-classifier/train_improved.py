import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
PREDICTION_HORIZON_DAYS = 5


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Target"] = (
        out["SP500"].shift(-PREDICTION_HORIZON_DAYS) > out["SP500"]
    ).astype(int)

    out["ret_1"] = out["SP500"].pct_change(1, fill_method=None)
    out["ret_2"] = out["SP500"].pct_change(2, fill_method=None)
    out["ret_5"] = out["SP500"].pct_change(5, fill_method=None)

    out["ma_5"] = out["SP500"].rolling(5).mean()
    out["ma_10"] = out["SP500"].rolling(10).mean()
    out["ma_20"] = out["SP500"].rolling(20).mean()

    out["vol_5"] = out["ret_1"].rolling(5).std()
    out["momentum_5"] = out["SP500"] / out["SP500"].shift(5) - 1

    out["ma_ratio_5"] = out["SP500"] / out["ma_5"] - 1
    out["ma_ratio_10"] = out["SP500"] / out["ma_10"] - 1
    out["ma_ratio_20"] = out["SP500"] / out["ma_20"] - 1

    out["day_of_week"] = out["observation_date"].dt.dayofweek

    return out.dropna().reset_index(drop=True)


def main() -> None:
    np.random.seed(RANDOM_STATE)

    df = pd.read_csv("data/SP500.csv")
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    if date_col != "observation_date":
        df = df.rename(columns={date_col: "observation_date"})

    model_df = build_features(df)

    feature_cols = [
        "ret_1",
        "ret_2",
        "ret_5",
        "vol_5",
        "momentum_5",
        "ma_ratio_5",
        "ma_ratio_10",
        "ma_ratio_20",
        "day_of_week",
    ]

    X = model_df[feature_cols]
    y = model_df["Target"]

    split = int(len(model_df) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    accuracy_pct = int(accuracy * 100)

    print(f"Prediction Horizon (days): {PREDICTION_HORIZON_DAYS}")
    print(f"Validation Accuracy: {accuracy_pct}%")
    print(f"Validation Accuracy (raw): {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, digits=4))


if __name__ == "__main__":
    main()
