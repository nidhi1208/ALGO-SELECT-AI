"""
Machine Learning Training Pipeline for Algorithm Selection.
Trains Random Forest, Gradient Boosting, MLP Neural Network, SVM, and k-NN models.
Computes evaluation metrics, confusion matrix, feature importances, and baseline speedups.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .data_generator import build_sorting_dataset
from ..features.sorting_features import FEATURE_NAMES

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_evaluate_models(dataset: pd.DataFrame = None) -> Dict[str, Any]:
    if dataset is None or len(dataset) < 100:
        dataset = build_sorting_dataset(samples_per_distribution=40)

    X = dataset[FEATURE_NAMES].fillna(0)
    y_raw = dataset["best_algorithm"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Candidate Models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        "MLP_NeuralNet": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42),
        "SVM_RBF": SVC(kernel="rbf", probability=True, C=10.0, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance")
    }

    results = {}
    best_model_name = None
    best_accuracy = -1.0
    best_model_obj = None

    for name, clf in models.items():
        # Train
        if name in ["MLP_NeuralNet", "SVM_RBF", "KNN"]:
            clf.fit(X_train_scaled, y_train)
            train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
            test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
            cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
        else:
            clf.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, clf.predict(X_train))
            test_acc = accuracy_score(y_test, clf.predict(X_test))
            cv_scores = cross_val_score(clf, X_train, y_train, cv=5)

        results[name] = {
            "train_accuracy": round(float(train_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4)
        }

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_name = name
            best_model_obj = clf

    # Extract Feature Importance (from Random Forest)
    rf_model = models["RandomForest"]
    feature_importances = {
        FEATURE_NAMES[i]: round(float(rf_model.feature_importances_[i]), 4)
        for i in range(len(FEATURE_NAMES))
    }
    sorted_importance = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))

    # Evaluate speedup vs default general-purpose algorithm ("QuickSort")
    speedup_ratios = []
    time_saved_pct = []
    
    test_indices = X_test.index
    test_dataset = dataset.loc[test_indices]
    
    y_pred_best = best_model_obj.predict(X_test if best_model_name in ["RandomForest", "GradientBoosting"] else X_test_scaled)
    pred_labels = label_encoder.inverse_transform(y_pred_best)

    for idx, pred_algo in zip(test_indices, pred_labels):
        row = dataset.loc[idx]
        default_time = row.get("time_QuickSort", 1.0)
        selected_time = row.get(f"time_{pred_algo}", default_time)
        
        if selected_time > 0 and default_time > 0:
            speedup = default_time / selected_time
            speedup_ratios.append(min(100.0, speedup))
            saved = max(0.0, (default_time - selected_time) / default_time * 100.0)
            time_saved_pct.append(saved)

    avg_speedup = float(np.mean(speedup_ratios)) if speedup_ratios else 1.0
    avg_time_saved = float(np.mean(time_saved_pct)) if time_saved_pct else 0.0

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_best)
    class_names = list(label_encoder.classes_)

    metadata = {
        "best_model_name": best_model_name,
        "best_accuracy": round(best_accuracy, 4),
        "model_comparison": results,
        "feature_importances": sorted_importance,
        "classes": class_names,
        "avg_speedup_vs_default": round(avg_speedup, 2),
        "avg_time_saved_pct": round(avg_time_saved, 1),
        "confusion_matrix": {
            "classes": class_names,
            "matrix": cm.tolist()
        },
        "dataset_size": len(dataset)
    }

    # Save artifacts
    joblib.dump(best_model_obj, os.path.join(MODEL_DIR, "best_model.joblib"))
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "random_forest.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTraining Complete! Best model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    print(f"Average speedup over default (Always QuickSort): {avg_speedup:.2f}x")
    print(f"Artifacts saved successfully in models directory.")

    return metadata

if __name__ == "__main__":
    train_and_evaluate_models()
