import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report


def build_and_train(csv_path, target_col="loan_status", output_path="model/rf_pipeline.joblib", random_state=42):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Detect column types
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocessing for numeric data
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = None
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print("Accuracy:", acc)
    print("Confusion matrix:\n", cm)
    if roc_auc is not None:
        print("ROC AUC:", roc_auc)
    print("Classification report:\n", classification_report(y_test, y_pred))

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    joblib.dump(clf, output_path)
    print(f"Saved trained pipeline to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train credit risk RandomForest pipeline")
    parser.add_argument("--csv", type=str, default="data/credit_risk_dataset.csv", help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default="loan_status", help="Target column name")
    parser.add_argument("--out", type=str, default="model/rf_pipeline.joblib", help="Output path for saved pipeline")
    args = parser.parse_args()

    build_and_train(args.csv, target_col=args.target, output_path=args.out)


if __name__ == "__main__":
    main()
