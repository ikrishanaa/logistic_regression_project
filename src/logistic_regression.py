import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve


def load_default_dataset():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


def load_csv_dataset(path, target):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")
    return df


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_and_evaluate(df, target_col, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)

    df = df.drop(columns=[col for col in df.columns if col.lower() in ["id", "unnamed: 32"]], errors="ignore")

    if df[target_col].dtype == object:
        df[target_col] = df[target_col].astype("category").cat.codes

    df = df.fillna(df.mean(numeric_only=True))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {metrics['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))
    plt.close()

    z = np.linspace(-10, 10, 100)
    plt.plot(z, sigmoid(z))
    plt.title("Sigmoid Function")
    plt.xlabel("z")
    plt.ylabel("sigmoid(z)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "sigmoid_function.png"))
    plt.close()

    print("Metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default="target", help="Target column name")
    args = parser.parse_args()

    if args.csv:
        df = load_csv_dataset(args.csv, args.target)
    else:
        df = load_default_dataset()

    train_and_evaluate(df, args.target)
