

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingRegressor,
    IsolationForest,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_DIR,
    MODEL_DIR,
    METRICS_FILE,
    EVAL_FILE,
    MODEL_FILE,
    PREDICTION_HORIZON_MINUTES,
    ROLLING_WINDOW,
    CONGESTION_UTILIZATION_PCT,
    CONGESTION_LATENCY_MS,
    CONGESTION_PACKET_LOSS_PCT,
    CONGESTION_JITTER_MS,
    MIN_TRAIN_ROWS,
    RANDOM_STATE,
)

warnings.filterwarnings("ignore")




BASE_FEATURES = [
    "bandwidth_utilization_pct",
    "latency_ms",
    "packet_loss_pct",
    "jitter_ms",
]

ROLLING_FEATURES = []

TARGET_METRICS = [
    "bandwidth_utilization_pct",
    "latency_ms",
    "packet_loss_pct",
    "jitter_ms",
]




def load_data():

    print("\n" + "=" * 70)
    print("LOADING NETWORK METRICS")
    print("=" * 70)

    if not Path(METRICS_FILE).exists():

        raise FileNotFoundError(
            f"\nCould not find network metrics file:\n"
            f"{METRICS_FILE}\n\n"
            f"Run collector.py first."
        )

    df = pd.read_csv(METRICS_FILE)

    print(f"File: {METRICS_FILE}")
    print(f"Rows loaded: {len(df)}")
    print(f"Columns found: {list(df.columns)}")

    if len(df) == 0:

        raise ValueError(
            "\nnetwork_metrics.csv is empty.\n"
            "Run collector.py and allow it to collect data."
        )

    return df




def clean_data(df):

    print("\n" + "=" * 70)
    print("CLEANING DATA")
    print("=" * 70)

    df = df.copy()


    timestamp_candidates = [
        "timestamp",
        "time",
        "datetime",
        "date",
    ]

    timestamp_column = None

    for column in timestamp_candidates:

        if column in df.columns:

            timestamp_column = column
            break

    if timestamp_column is None:

        raise ValueError(
            "\nNo timestamp column was found.\n"
            "Expected one of:\n"
            "timestamp, time, datetime, date"
        )

    df["timestamp"] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce"
    )

    before = len(df)

    df = df.dropna(subset=["timestamp"])

    removed = before - len(df)

    if removed > 0:

        print(
            f"Removed {removed} rows with invalid timestamps."
        )

   

    missing_columns = [
        column
        for column in BASE_FEATURES
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nThe following required network metrics are missing:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )


    for column in BASE_FEATURES:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    df = df.sort_values("timestamp")


    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    )

    df = df.reset_index(drop=True)


    print(f"Rows after cleaning: {len(df)}")

    print("\nMetric validity:")

    for column in BASE_FEATURES:

        valid = df[column].notna().sum()
        missing = df[column].isna().sum()

        print(
            f"  {column:<35} "
            f"valid={valid:<6} "
            f"missing={missing}"
        )

 

    before = len(df)

    df = df.dropna(
        subset=BASE_FEATURES,
        how="all"
    )

    removed = before - len(df)

    if removed:

        print(
            f"\nRemoved {removed} rows where all metrics were missing."
        )

    df = df.reset_index(drop=True)

    if len(df) < 2:

        raise ValueError(
            "\nNot enough valid network observations."
        )

    return df




def build_features(df):

    print("\n" + "=" * 70)
    print("BUILDING FEATURES")
    print("=" * 70)

    df = df.copy()

    window = max(
        2,
        int(ROLLING_WINDOW)
    )

    for column in BASE_FEATURES:

      
        mean_column = f"{column}_rolling_mean"

        df[mean_column] = (
            df[column]
            .rolling(window=window, min_periods=1)
            .mean()
        )

        ROLLING_FEATURES.append(
            mean_column
        )

  
        std_column = f"{column}_rolling_std"

        df[std_column] = (
            df[column]
            .rolling(window=window, min_periods=2)
            .std()
            .fillna(0)
        )

        ROLLING_FEATURES.append(
            std_column
        )

       
        delta_column = f"{column}_delta"

        df[delta_column] = (
            df[column]
            .diff()
            .fillna(0)
        )

        ROLLING_FEATURES.append(
            delta_column
        )

    print(
        f"Rolling window: {window}"
    )

    print(
        f"Additional features created: "
        f"{len(ROLLING_FEATURES)}"
    )

    return df




def estimate_sample_interval(df):

    timestamps = (
        pd.to_datetime(df["timestamp"])
        .sort_values()
    )

    differences = (
        timestamps
        .diff()
        .dt.total_seconds()
    )

    differences = differences[
        differences > 0
    ]

    if differences.empty:

        print(
            "\nCould not determine sampling interval."
        )

        return 5.0

    median_seconds = differences.median()

    print(
        f"\nEstimated sampling interval: "
        f"{median_seconds:.2f} seconds"
    )

    return float(median_seconds)



def calculate_safe_future_steps(df):

    print("\n" + "=" * 70)
    print("CALCULATING FUTURE PREDICTION HORIZON")
    print("=" * 70)

    requested_minutes = float(
        PREDICTION_HORIZON_MINUTES
    )

    sample_seconds = estimate_sample_interval(df)

    if sample_seconds <= 0:

        sample_seconds = 5.0

    requested_steps = int(
        round(
            (requested_minutes * 60)
            / sample_seconds
        )
    )

    requested_steps = max(
        1,
        requested_steps
    )

    total_rows = len(df)


    maximum_safe_steps = max(
        1,
        int(total_rows * 0.20)
    )

    steps = min(
        requested_steps,
        maximum_safe_steps
    )


    if steps >= total_rows:

        steps = max(
            1,
            total_rows - 2
        )

    print(
        f"Configured prediction horizon: "
        f"{requested_minutes:.2f} minutes"
    )

    print(
        f"Requested future steps: "
        f"{requested_steps}"
    )

    print(
        f"Maximum safe future steps: "
        f"{maximum_safe_steps}"
    )

    print(
        f"ACTUAL future steps used: "
        f"{steps}"
    )

    if steps != requested_steps:

        print(
            "\nWARNING:"
            "\nThe requested prediction horizon is too large "
            "\nfor the amount of data currently collected."
            "\nThe training system automatically reduced it."
        )

    return steps




def calculate_adaptive_thresholds(df):

    print("\n" + "=" * 70)
    print("DEGRADATION THRESHOLDS")
    print("=" * 70)

    absolute_thresholds = {

        "bandwidth_utilization_pct":
            CONGESTION_UTILIZATION_PCT,

        "latency_ms":
            CONGESTION_LATENCY_MS,

        "packet_loss_pct":
            CONGESTION_PACKET_LOSS_PCT,

        "jitter_ms":
            CONGESTION_JITTER_MS,
    }

    print(
        f"Absolute utilization threshold: "
        f"{CONGESTION_UTILIZATION_PCT}%"
    )

    print(
        f"Absolute latency threshold: "
        f"{CONGESTION_LATENCY_MS} ms"
    )

    print(
        f"Absolute packet loss threshold: "
        f"{CONGESTION_PACKET_LOSS_PCT}%"
    )

    print(
        f"Absolute jitter threshold: "
        f"{CONGESTION_JITTER_MS} ms"
    )

    adaptive = {}

    for column in TARGET_METRICS:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(values) < 20:

            adaptive[column] = np.nan

            print(
                f"Adaptive {column}: disabled "
                f"(only {len(values)} valid observations)"
            )

            continue

        unique_values = values.nunique()

        if unique_values < 5:

            adaptive[column] = np.nan

            print(
                f"Adaptive {column}: disabled "
                f"(only {unique_values} unique values)"
            )

            continue

        threshold = values.quantile(0.80)

        if not np.isfinite(threshold):

            adaptive[column] = np.nan

            print(
                f"Adaptive {column}: disabled "
                f"(invalid percentile)"
            )

            continue

        # Never allow a zero adaptive threshold
        if threshold <= 0:

            adaptive[column] = np.nan

            print(
                f"Adaptive {column}: disabled "
                f"(threshold={threshold:.3f})"
            )

            continue

        adaptive[column] = float(
            threshold
        )

        print(
            f"Adaptive {column}: "
            f"{threshold:.3f}"
        )

    return absolute_thresholds, adaptive



def create_future_values(df, steps):

    print("\n" + "=" * 70)
    print("CREATING FUTURE TARGET VALUES")
    print("=" * 70)

    df = df.copy()

    for column in TARGET_METRICS:

        future_column = (
            f"future_{column}"
        )

        df[future_column] = (
            df[column]
            .shift(-steps)
        )

    print(
        f"Future horizon steps: {steps}"
    )

    for column in TARGET_METRICS:

        future_column = (
            f"future_{column}"
        )

        valid = (
            df[future_column]
            .notna()
            .sum()
        )

        print(
            f"{future_column:<45} "
            f"valid={valid}"
        )

    return df




def make_future_target(
    df,
    absolute_thresholds,
    adaptive_thresholds
):

    print("\n" + "=" * 70)
    print("CREATING FUTURE DEGRADATION TARGET")
    print("=" * 70)

    df = df.copy()

    future_columns = [
        f"future_{column}"
        for column in TARGET_METRICS
    ]

   

    before = len(df)

    df = df.dropna(
        subset=future_columns,
        how="all"
    ).reset_index(drop=True)

    removed = before - len(df)

    print(
        f"Rows removed because all future target "
        f"metrics were unavailable: {removed}"
    )

    print(
        f"Rows remaining: {len(df)}"
    )

    if len(df) == 0:

        raise ValueError(
            "\nNo future observations are available.\n\n"
            "The collector must collect more data."
        )


    conditions = []

    for column in TARGET_METRICS:

        future_column = (
            f"future_{column}"
        )

        future_values = df[
            future_column
        ]

        # Absolute threshold
        absolute_threshold = (
            absolute_thresholds[column]
        )

        if column == "bandwidth_utilization_pct":

            absolute_condition = (
                future_values
                >= absolute_threshold
            )

        else:

            absolute_condition = (
                future_values
                >= absolute_threshold
            )

        conditions.append(
            absolute_condition
        )

        
        adaptive_threshold = (
            adaptive_thresholds.get(
                column,
                np.nan
            )
        )

        if np.isfinite(
            adaptive_threshold
        ):

            adaptive_condition = (
                future_values
                >= adaptive_threshold
            )

            conditions.append(
                adaptive_condition
            )


    target_matrix = pd.concat(
        [
            condition.fillna(False)
            .astype(int)
            for condition in conditions
        ],
        axis=1
    )

    df["congestion_target"] = (
        target_matrix
        .max(axis=1)
        .astype(int)
    )

  

    degraded_count = int(
        df["congestion_target"].sum()
    )

    normal_count = int(
        len(df) - degraded_count
    )

    print("\nInitial target distribution:")

    print(
        f"Normal observations:     "
        f"{normal_count}"
    )

    print(
        f"Degraded observations:   "
        f"{degraded_count}"
    )

  

    if degraded_count == 0:

        print(
            "\nNo observations exceeded the configured "
            "degradation thresholds."
        )

        print(
            "Creating a data-driven fallback target "
            "using the worst future network conditions."
        )

        scores = pd.Series(
            0.0,
            index=df.index
        )

        weights = {

            "future_bandwidth_utilization_pct":
                1.0,

            "future_latency_ms":
                1.0,

            "future_packet_loss_pct":
                1.0,

            "future_jitter_ms":
                1.0,
        }

        for column, weight in weights.items():

            values = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            if values.notna().sum() > 1:

                minimum = values.min()
                maximum = values.max()

                if maximum > minimum:

                    normalized = (
                        values - minimum
                    ) / (
                        maximum - minimum
                    )

                    scores += (
                        normalized
                        .fillna(0)
                        * weight
                    )

        # Top 20% worst observations
        cutoff = scores.quantile(0.80)

        df["congestion_target"] = (
            scores >= cutoff
        ).astype(int)

        degraded_count = int(
            df["congestion_target"].sum()
        )

        normal_count = (
            len(df) - degraded_count
        )

        print(
            "\nFallback target distribution:"
        )

        print(
            f"Normal observations:     "
            f"{normal_count}"
        )

        print(
            f"Degraded observations:   "
            f"{degraded_count}"
        )

    return df




def get_feature_columns():

    return BASE_FEATURES + ROLLING_FEATURES


def train_regressor(X_train, y_train):

    model = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "model",
                GradientBoostingRegressor(
                    random_state=RANDOM_STATE,
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                )
            ),
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    return model



def train(
    df,
    absolute_thresholds,
    adaptive_thresholds
):

    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)

    feature_columns = get_feature_columns()



    df = df.dropna(
        subset=["congestion_target"]
    ).reset_index(drop=True)

    if len(df) < 10:

        raise ValueError(
            "\nNot enough observations available "
            "after target creation.\n\n"
            f"Available rows: {len(df)}\n"
            f"Required minimum: 10"
        )

    # ------------------------------------------------------------
    # Prepare X and y
    # ------------------------------------------------------------

    X = df[
        feature_columns
    ].copy()

    y = df[
        "congestion_target"
    ].astype(int)

 

    if y.nunique() < 2:

        print(
            "\nOnly one target class exists."
        )

        print(
            "Creating a fallback split using "
            "future network condition scores."
        )

        scores = pd.Series(
            0.0,
            index=df.index
        )

        for column in [
            "future_bandwidth_utilization_pct",
            "future_latency_ms",
            "future_packet_loss_pct",
            "future_jitter_ms",
        ]:

            if column in df.columns:

                values = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

                if values.notna().sum() > 1:

                    rank = (
                        values
                        .rank(pct=True)
                        .fillna(0)
                    )

                    scores += rank

        y = (
            scores
            >= scores.quantile(0.80)
        ).astype(int)

   

    split_index = int(
        len(df) * 0.80
    )

    split_index = max(
        1,
        min(
            split_index,
            len(df) - 1
        )
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    print(
        f"\nTraining rows: {len(X_train)}"
    )

    print(
        f"Testing rows:  {len(X_test)}"
    )

    print(
        "\nTraining target distribution:"
    )

    print(
        y_train.value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nTesting target distribution:"
    )

    print(
        y_test.value_counts()
        .sort_index()
        .to_string()
    )

  

    classifier = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                )
            ),
        ]
    )

    classifier.fit(
        X_train,
        y_train
    )


    predictions = classifier.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = None

    if y_test.nunique() >= 2:

        probabilities = classifier.predict_proba(
            X_test
        )[:, 1]

        auc = roc_auc_score(
            y_test,
            probabilities
        )



    regression_models = {}

    regression_errors = {}

    for metric in TARGET_METRICS:

        future_column = (
            f"future_{metric}"
        )

        if future_column not in df.columns:

            continue

        valid_train = (
            df.iloc[:split_index]
            [future_column]
            .notna()
        )

        valid_test = (
            df.iloc[split_index:]
            [future_column]
            .notna()
        )

        if valid_train.sum() < 5:

            print(
                f"\nSkipping regression for {metric}: "
                f"not enough training observations."
            )

            continue

        X_reg_train = X_train.loc[
            valid_train.values
        ]

        y_reg_train = (
            df.iloc[:split_index]
            .loc[
                valid_train.values,
                future_column
            ]
        )

        model = train_regressor(
            X_reg_train,
            y_reg_train
        )

        regression_models[
            metric
        ] = model

        if valid_test.sum() > 0:

            X_reg_test = X_test.loc[
                valid_test.values
            ]

            y_reg_test = (
                df.iloc[split_index:]
                .loc[
                    valid_test.values,
                    future_column
                ]
            )

            reg_predictions = model.predict(
                X_reg_test
            )

            regression_errors[
                metric
            ] = mean_absolute_error(
                y_reg_test,
                reg_predictions
            )

   
    anomaly_model = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                IsolationForest(
                    n_estimators=300,
                    contamination="auto",
                    random_state=RANDOM_STATE,
                )
            ),
        ]
    )

    anomaly_model.fit(
        X_train
    )


    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_package = {

        "classifier":
            classifier,

        "regression_models":
            regression_models,

        "anomaly_model":
            anomaly_model,

        "feature_columns":
            feature_columns,

        "target_metrics":
            TARGET_METRICS,

        "absolute_thresholds":
            absolute_thresholds,

        "adaptive_thresholds":
            adaptive_thresholds,

        "prediction_horizon_minutes":
            PREDICTION_HORIZON_MINUTES,

        "random_state":
            RANDOM_STATE,
    }

    joblib.dump(
        model_package,
        MODEL_FILE
    )



    evaluation = {

        "rows_total":
            int(len(df)),

        "training_rows":
            int(len(X_train)),

        "testing_rows":
            int(len(X_test)),

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "roc_auc":
            None
            if auc is None
            else float(auc),

        "regression_mae":
            {
                key: float(value)
                for key, value
                in regression_errors.items()
            },

        "normal_observations":
            int((y == 0).sum()),

        "degraded_observations":
            int((y == 1).sum()),
    }

    with open(
        EVAL_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation,
            file,
            indent=4
        )

   

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Accuracy:   {accuracy:.4f}"
    )

    print(
        f"Precision:  {precision:.4f}"
    )

    print(
        f"Recall:     {recall:.4f}"
    )

    print(
        f"F1 Score:   {f1:.4f}"
    )

    if auc is not None:

        print(
            f"ROC AUC:    {auc:.4f}"
        )

    print(
        "\nRegression MAE:"
    )

    for metric, error in regression_errors.items():

        print(
            f"  {metric}: {error:.4f}"
        )

    print(
        "\nModel saved to:"
    )

    print(
        MODEL_FILE
    )

    print(
        "\nEvaluation saved to:"
    )

    print(
        EVAL_FILE
    )

    print(
        "\nSystem is ready for prediction."
    )



def main():

    print("\n")
    print("=" * 70)
    print("PREDICTIVE NETWORK MANAGEMENT SYSTEM")
    print("MODEL TRAINING")
    print("=" * 70)



    df = load_data()



    df = clean_data(
        df
    )

    

    df = build_features(
        df
    )


    (
        absolute_thresholds,
        adaptive_thresholds
    ) = calculate_adaptive_thresholds(
        df
    )


    steps = calculate_safe_future_steps(
        df
    )



    df = create_future_values(
        df,
        steps
    )



    df = make_future_target(
        df,
        absolute_thresholds,
        adaptive_thresholds
    )

  

    train(
    df,
    absolute_thresholds,
    adaptive_thresholds
)

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)




if __name__ == "__main__":

    main()