"""
step2_train_model.py
Train Random Forest + SVM on 16 PPG features with StandardScaler.
Outputs: model.pkl (includes scaler)

Usage:
    python3 step2_train_model.py --data features_dataset.csv --output model.pkl
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

FEATURE_COLS = [
    'hr_mean', 'hr_std',
    'ibi_mean', 'ibi_std',
    'rmssd', 'sdnn', 'pnn50',
    'lf_power', 'hf_power', 'lf_hf_ratio',
    'peak_count', 'mean_peak_amp',
    'sig_std', 'sig_range',
    'sig_skewness', 'sig_kurtosis',
]

def load_data(path):
    df = pd.read_csv(path)
    available = [c for c in FEATURE_COLS if c in df.columns]
    print(f"\nLoaded: {len(df)} samples  |  {len(available)} features")
    for i, f in enumerate(available, 1):
        print(f"  {i:2d}. {f}")
    X = df[available].values
    y = df['label'].values
    return X, y, available

def train_and_evaluate(X, y, feature_names):
    print(f"\nClass balance — Low(0): {(y==0).sum()}  High(1): {(y==1).sum()}")

    # ── Fit scaler on ALL training data ───────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("\nStandardScaler fitted on full dataset.")
    print(f"  Feature means (first 4): {scaler.mean_[:4].round(3)}")
    print(f"  Feature stds  (first 4): {scaler.scale_[:4].round(3)}")

    # ── Models ────────────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        min_samples_leaf=4, random_state=42, n_jobs=-1
    )
    svm = SVC(kernel='rbf', C=10, gamma='scale',
              probability=True, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Cross-validation on scaled data ───────────────────────────────────────
    print("\n── 5-Fold Cross-Validation (on scaled features) ──────")
    for name, model in [("Random Forest", rf), ("SVM", svm)]:
        acc = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        f1  = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
        print(f"  {name:15s}  Accuracy: {acc.mean():.3f} ± {acc.std():.3f}  "
              f"F1: {f1.mean():.3f} ± {f1.std():.3f}")

    # ── Train/test split for final evaluation ─────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    print("\n── Classification Report (held-out test set 20%) ────")
    print(classification_report(y_te, y_pred, target_names=['Low MWL', 'High MWL']))

    # Feature importance
    # Refit on full scaled data for final model
    rf.fit(X_scaled, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("── Feature Importance (ranked) ──────────────────────")
    for i in indices:
        bar = '█' * int(importances[i] * 100)
        print(f"  {feature_names[i]:20s}: {importances[i]:.4f}  {bar}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Model Evaluation — PPG Attention State Detection", fontsize=13)

    cm = confusion_matrix(y_te, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Low MWL', 'High MWL']).plot(
        ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion Matrix (test set 20%)")

    axes[1].barh([feature_names[i] for i in indices],
                 importances[indices], color='steelblue')
    axes[1].set_title("Feature Importance (16 features)")
    axes[1].set_xlabel("Importance score")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig("model_evaluation.png", dpi=150)
    print("\nPlot saved: model_evaluation.png")
    plt.show()

    return rf, scaler

def main(data_path, output_path):
    X, y, feature_names = load_data(data_path)

    mask = ~np.isnan(X).any(axis=1)
    removed = len(X) - mask.sum()
    X, y = X[mask], y[mask]
    if removed > 0:
        print(f"Removed {removed} rows with NaN values")

    rf, scaler = train_and_evaluate(X, y, feature_names)

    # Save model + scaler + feature names together
    joblib.dump({
        'model':    rf,
        'scaler':   scaler,
        'features': feature_names
    }, output_path)
    print(f"\n Model + Scaler saved: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   default='features_dataset.csv')
    parser.add_argument('--output', default='model.pkl')
    args = parser.parse_args()
    main(args.data, args.output)