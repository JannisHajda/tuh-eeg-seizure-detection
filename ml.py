# cores: 64, mem: 256G, node: Kottos
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from config import get_ml_config
import os
import json

N_SPLITS = 10
OUTPUT_PATH = "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/results/"

random_state = 42


def tn_scorer(y_true, y_pred):
    tn, _, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn


def fp_scorer(y_true, y_pred):
    _, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return fp


def fn_scorer(y_true, y_pred):
    _, _, fn, _ = confusion_matrix(y_true, y_pred).ravel()
    return fn


def tp_scorer(y_true, y_pred):
    _, _, _, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp


cv = StratifiedGroupKFold(
    n_splits=N_SPLITS, shuffle=True, random_state=random_state)
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score),
    'tn': make_scorer(tn_scorer),
    'fp': make_scorer(fp_scorer),
    'fn': make_scorer(fn_scorer),
    'tp': make_scorer(tp_scorer),
}


def remove_highly_correlated_features(X, threshold=0.95):
    # Calculate the correlation matrix
    corr_matrix = X.corr().abs()

    # Select the upper triangle of the correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(
        upper[column] > threshold)]

    # Drop highly correlated features
    X_reduced = X.drop(columns=to_drop)

    return X_reduced, to_drop


def cross_validate_pipeline(pipeline, X, y, groups, cv, scoring):
    results = cross_validate(pipeline, X, y, groups=groups,
                             cv=cv, scoring=scoring, return_train_score=False)

    # Initialize variables to accumulate confusion matrix components
    sum_tn, sum_fp, sum_fn, sum_tp = 0, 0, 0, 0

    # iterate over test_tn, test_fp, test_fn, test_tp
    for i in range(N_SPLITS):
        sum_tn += results['test_tn'][i]
        sum_fp += results['test_fp'][i]
        sum_fn += results['test_fn'][i]
        sum_tp += results['test_tp'][i]

    # Store the sum of confusion matrix components
    avg_results = {
        'sum_tn': int(sum_tn),
        'sum_fp': int(sum_fp),
        'sum_fn': int(sum_fn),
        'sum_tp': int(sum_tp),
    }

    # Add other average metrics if needed
    for metric, values in results.items():
        if metric not in ['test_tn', 'test_fp', 'test_fn', 'test_tp']:
            avg_results[metric] = float(np.mean(values))

    return results, avg_results


if __name__ == "__main__":
    input_files = [
        # "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/features/features_30.csv",
        # "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/features/features_25.csv",
        # "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/features/features_20.csv",
        # "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/features/features_15.csv",
        # "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/features/features_10.csv",
        # "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/features/features_5.csv",
        # "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/features/features_2.csv",
        "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/features/features_1.csv",
    ]

    knn_pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=50))
    ])

    rf_pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100))
    ])

    svm_pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1, gamma=0.001, probability=True))
    ])

    pipelines = {
        "knn": knn_pipeline,
        "rf": rf_pipeline,
    }

    for input_file in input_files:
        print(f"Loading {input_file}...")

        window_length = input_file.split("_")[-1].split(".")[0]

        feature_windows = pd.read_csv(input_file)
        print("Loaded features.")

        feature_windows = feature_windows.dropna()

        X = feature_windows.drop(columns=["label", "patient_id"])
        y = feature_windows["label"]
        groups = feature_windows["patient_id"]

        # Remove highly correlated features
        print("Removing highly correlated features...")
        X, _ = remove_highly_correlated_features(X, threshold=0.95)
        print("Removed highly correlated features.")

        for classifier, pipeline in pipelines.items():
            print(
                f"Training {classifier} with window length {window_length}...")
            split_results, avg_results = cross_validate_pipeline(
                pipeline, X, y, groups, cv, scoring)
            print("Finished training.")

            results = {
                "splits": [
                    {
                        "accuracy": float(split_results['test_accuracy'][i]),
                        "precision": float(split_results['test_precision'][i]),
                        "recall": float(split_results['test_recall'][i]),
                        "f1": float(split_results['test_f1'][i]),
                        "roc_auc": float(split_results['test_roc_auc'][i]),
                        "tn": int(split_results['test_tn'][i]),
                        "fp": int(split_results['test_fp'][i]),
                        "fn": int(split_results['test_fn'][i]),
                        "tp": int(split_results['test_tp'][i]),
                        "score_time": float(split_results['score_time'][i]),
                        "fit_time": float(split_results['fit_time'][i])

                    } for i in range(N_SPLITS)
                ],
                "avg": avg_results
            }

            print("Saving results...")
            results_output = os.path.join(
                OUTPUT_PATH, f"{classifier}_{window_length}.json")

            with open(results_output, "w") as f:
                json.dump(results, f)

            print(f"Results saved to {results_output}.")
