
import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# =====================================================
# CONFIGURATION
# =====================================================

DATASET_PATH = (
    r"ALLFLOWMETER_HIKARI2021.csv\ALLFLOWMETER_HIKARI2021.csv"
)

MODEL_DIR = "models"

BINARY_TARGET = "Label"
MULTICLASS_TARGET = "traffic_category"

RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================================
# LOAD DATASET
# =====================================================

def load_dataset():

    print("\nLoading HIKARI dataset...")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH,
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    print("Original dataset shape:", df.shape)

    df = df.drop_duplicates().copy()

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    for target in [
        BINARY_TARGET,
        MULTICLASS_TARGET
    ]:

        if target in df.columns:

            df[target] = (
                df[target]
                .astype("string")
                .str.strip()
            )

            df[target] = df[target].replace(
                ["", "nan", "None", "<NA>"],
                pd.NA
            )

    print("Cleaned dataset shape:", df.shape)

    return df


# =====================================================
# FEATURE SELECTION
# =====================================================


def get_features(df):

    excluded = {
        BINARY_TARGET,
        MULTICLASS_TARGET
    }

    for col in df.columns:

        normalized = (
            col.lower()
            .replace("_", "")
            .replace("-", "")
            .strip()
        )

        # Remove automatically generated CSV indexes
        if normalized.startswith("unnamed"):
            excluded.add(col)

        # Remove identifiers
        if normalized in {
            "uid",
            "flowid",
            "originh",
            "responh",
            "srcip",
            "dstip",
            "sourceip",
            "destinationip",
            "srcmac",
            "dstmac",
            "timestamp"
        }:
            excluded.add(col)

    features = [
        col for col in df.columns
        if col not in excluded
        and df[col].notna().any()
    ]

    return features


# =====================================================
# PREPROCESSING
# =====================================================

def create_preprocessor(X):

    numeric_features = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = [
        col for col in X.columns
        if col not in numeric_features
    ]

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    return preprocessor


# =====================================================
# MODEL DEFINITIONS
# =====================================================

def get_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            class_weight=None,
            random_state=RANDOM_STATE
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=25,
            min_samples_leaf=2,
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=20,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    return models


# =====================================================
# DATA SPLITTING
# =====================================================

def split_data(X, y):

    counts = y.value_counts()

    if len(counts) < 2 or counts.min() < 3:
        raise ValueError(
            "Each class must have at least 3 records."
        )

    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            stratify=y,
            random_state=RANDOM_STATE
        )
    )

    X_val, X_test, y_val, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            stratify=y_temp,
            random_state=RANDOM_STATE
        )
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# =====================================================
# METRIC CALCULATION
# =====================================================

def calculate_metrics(model, X, y):

    predictions = model.predict(X)

    return {

        "Accuracy": accuracy_score(
            y,
            predictions
        ),

        "Precision": precision_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        ),

        "Recall": recall_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        ),

        "F1-score": f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        )
    }


# =====================================================
# TRAIN AND SELECT MODEL
# =====================================================

def train_and_select(X, y, task_name):

    print("\n" + "=" * 60)
    print(f"TRAINING {task_name.upper()} MODEL")
    print("=" * 60)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(X, y)

    print("\nTraining records:", len(X_train))
    print("Validation records:", len(X_val))
    print("Testing records:", len(X_test))

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    models = get_models()

    best_model = None
    best_score = -1

    comparison = []

    for name, classifier in models.items():

        print(f"\nTraining {name}...", flush=True)

        preprocessor = create_preprocessor(
            X_train
        )

        pipeline = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ])

        pipeline.fit(
            X_train,
            y_train
        )

        validation_metrics = calculate_metrics(
            pipeline,
            X_val,
            y_val
        )

        score = validation_metrics["F1-score"]

        print(
            f"Validation Accuracy: "
            f"{validation_metrics['Accuracy']:.4f}"
        )

        print(
            f"Validation Macro F1: {score:.4f}"
        )

        comparison.append({
            "Algorithm": name,
            **validation_metrics
        })

        if score > best_score:

            best_score = score
            best_model = pipeline

    # -------------------------------------------------
    # TEST SELECTED MODEL
    # -------------------------------------------------

    print("\nSelected model:")
    print(best_model.named_steps["classifier"])

    print(
        f"\nBest Validation Macro F1: "
        f"{best_score:.4f}"
    )

    test_metrics = calculate_metrics(
        best_model,
        X_test,
        y_test
    )

    print("\nIndependent Test Results:")

    for metric, value in test_metrics.items():

        print(f"{metric}: {value:.4f}")

    predictions = best_model.predict(X_test)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # -------------------------------------------------
    # CONFUSION MATRIX
    # -------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=best_model.classes_
    )

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=best_model.classes_,
        yticklabels=best_model.classes_
    )

    plt.title(
        f"{task_name} Confusion Matrix"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            MODEL_DIR,
            f"{task_name.lower()}_confusion_matrix.png"
        ),
        dpi=150
    )

    plt.close()

    # -------------------------------------------------
    # SAVE COMPARISON
    # -------------------------------------------------

    comparison_df = pd.DataFrame(comparison)

    comparison_df.to_csv(
        os.path.join(
            MODEL_DIR,
            f"{task_name.lower()}_comparison.csv"
        ),
        index=False
    )

    return {

        "model": best_model,

        "classes": [
            str(c) for c in best_model.classes_
        ],

        "test_metrics": {
            k: float(v)
            for k, v in test_metrics.items()
        },

        "comparison": comparison
    }


# =====================================================
# DEMO METADATA
# =====================================================

def generate_demo_metadata(df, features):

    X = df[features].copy()

    numeric_features = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = [
        col for col in features
        if col not in numeric_features
    ]

    metadata = {

        "numeric_features": numeric_features,

        "categorical_features": categorical_features,

        "defaults": {},

        "options": {}
    }

    # NUMERIC FEATURES

    for col in numeric_features:

        values = pd.to_numeric(
            X[col],
            errors="coerce"
        )

        values = values.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if values.empty:
            continue

        metadata["defaults"][col] = float(
            values.median()
        )

        quantiles = values.quantile(
            [0.10, 0.25, 0.50, 0.75, 0.90]
        )

        metadata["options"][col] = {

            "Low": float(quantiles.loc[0.10]),

            "Medium-Low": float(quantiles.loc[0.25]),

            "Medium": float(quantiles.loc[0.50]),

            "Medium-High": float(quantiles.loc[0.75]),

            "High": float(quantiles.loc[0.90])
        }

    # CATEGORICAL FEATURES

    for col in categorical_features:

        values = X[col].dropna().astype(str)

        mode = values.mode()

        metadata["defaults"][col] = (
            str(mode.iloc[0])
            if not mode.empty
            else "Unknown"
        )

        metadata["options"][col] = (
            values.value_counts()
            .head(20)
            .index.tolist()
        )

    return metadata


# =====================================================
# MAIN
# =====================================================

def main():

    df = load_dataset()

    required_columns = [
        BINARY_TARGET,
        MULTICLASS_TARGET
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing target columns: {missing}"
        )

    features = get_features(df)

    print("\nSelected features:", len(features))

    # -------------------------------------------------
    # BINARY CLASSIFICATION
    # -------------------------------------------------

    binary_df = df.dropna(
        subset=[BINARY_TARGET]
    ).copy()

    X_binary = binary_df[features].copy()

    y_binary = (
        binary_df[BINARY_TARGET]
        .astype(str)
        .str.strip()
    )

    binary_result = train_and_select(
        X_binary,
        y_binary,
        "Binary"
    )

    joblib.dump(
        binary_result["model"],
        os.path.join(
            MODEL_DIR,
            "binary_model.pkl"
        )
    )

    print("\nBinary model saved successfully.")

    # -------------------------------------------------
    # MULTICLASS CLASSIFICATION
    # -------------------------------------------------

    multi_df = df.dropna(
        subset=[MULTICLASS_TARGET]
    ).copy()

    X_multi = multi_df[features].copy()

    y_multi = (
        multi_df[MULTICLASS_TARGET]
        .astype(str)
        .str.strip()
    )

    multiclass_result = train_and_select(
        X_multi,
        y_multi,
        "Multiclass"
    )

    joblib.dump(
        multiclass_result["model"],
        os.path.join(
            MODEL_DIR,
            "multiclass_model.pkl"
        )
    )

    print("\nMulticlass model saved successfully.")

    # -------------------------------------------------
    # GENERATE DEMO METADATA
    # -------------------------------------------------

    demo_metadata = generate_demo_metadata(
        binary_df,
        features
    )

    metadata = {

        "feature_columns": features,

        "binary_classes": binary_result["classes"],

        "multiclass_classes": multiclass_result["classes"],

        "binary_metrics": binary_result["test_metrics"],

        "multiclass_metrics": multiclass_result["test_metrics"],

        "binary_comparison": binary_result["comparison"],

        "multiclass_comparison": multiclass_result["comparison"],

        "demo": demo_metadata
    }

    # -------------------------------------------------
    # SAVE METADATA
    # -------------------------------------------------

    with open(
        os.path.join(
            MODEL_DIR,
            "model_metadata.json"
        ),
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # -------------------------------------------------
    # COMPLETION
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated files:")

    print("models/binary_model.pkl")
    print("models/multiclass_model.pkl")
    print("models/model_metadata.json")
    print("models/binary_comparison.csv")
    print("models/multiclass_comparison.csv")
    print("models/binary_confusion_matrix.png")
    print("models/multiclass_confusion_matrix.png")


# =====================================================
# EXECUTION
# =====================================================

if __name__ == "__main__":
    main()
