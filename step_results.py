"""
step_results.py
Run this to get all evaluation metrics for your poster/report.

Usage:
    python3 step_results.py --data features_dataset.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, accuracy_score)
from sklearn.pipeline import Pipeline

FEATURE_COLS = [
    'hr_mean', 'hr_std', 'ibi_mean', 'ibi_std',
    'rmssd', 'sdnn', 'pnn50',
    'lf_power', 'hf_power', 'lf_hf_ratio',
    'peak_count', 'mean_peak_amp',
    'sig_std', 'sig_range', 'sig_skewness', 'sig_kurtosis',
]

def main(data_path):
    df = pd.read_csv(data_path)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df['label'].values
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]

    # ── Dataset info ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("  DATASET INFO")
    print("="*60)
    print(f"  Total windows:     {len(y)}")
    print(f"  Low MWL  (label 0): {(y==0).sum()}")
    print(f"  High MWL (label 1): {(y==1).sum()}")
    if 'source_file' in df.columns:
        participants = df[mask]['source_file'].str.extract(r'(p\d+)')[0].nunique()
        print(f"  Participants:      {participants}")
    print(f"  Features:          {len(available)}")
    print(f"  Window size:       30s @ 256Hz = 7680 samples")
    print(f"  Step:              15s (50% overlap)")

    # ── Models ────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        min_samples_leaf=4, random_state=42, n_jobs=-1
    )
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=10, gamma='scale',
                    probability=True, random_state=42))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── 5-Fold CV ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  5-FOLD CROSS-VALIDATION RESULTS")
    print("="*60)
    print(f"  {'Model':<20} {'Accuracy':>12}  {'F1':>12}")
    print(f"  {'-'*20} {'-'*12}  {'-'*12}")

    for name, model in [("Random Forest", rf), ("SVM (RBF)", svm)]:
        acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        f1  = cross_val_score(model, X, y, cv=cv, scoring='f1')
        print(f"  {name:<20} {acc.mean():.4f}±{acc.std():.4f}  "
              f"{f1.mean():.4f}±{f1.std():.4f}")

    # ── Detailed RF report + confusion matrix ─────────────────────
    print("\n" + "="*60)
    print("  RANDOM FOREST — DETAILED REPORT (CV predictions)")
    print("="*60)
    rf_preds = cross_val_predict(rf, X, y, cv=cv)
    print(classification_report(y, rf_preds, target_names=['Low MWL', 'High MWL']))

    cm_rf = confusion_matrix(y, rf_preds)
    print("  Confusion Matrix (RF):")
    print(f"  {'':15s} Pred Low  Pred High")
    print(f"  {'True Low':15s} {cm_rf[0][0]:8d}  {cm_rf[0][1]:9d}")
    print(f"  {'True High':15s} {cm_rf[1][0]:8d}  {cm_rf[1][1]:9d}")

    # ── Detailed SVM report ───────────────────────────────────────
    print("\n" + "="*60)
    print("  SVM — DETAILED REPORT (CV predictions)")
    print("="*60)
    svm_preds = cross_val_predict(svm, X, y, cv=cv)
    print(classification_report(y, svm_preds, target_names=['Low MWL', 'High MWL']))

    cm_svm = confusion_matrix(y, svm_preds)
    print("  Confusion Matrix (SVM):")
    print(f"  {'':15s} Pred Low  Pred High")
    print(f"  {'True Low':15s} {cm_svm[0][0]:8d}  {cm_svm[0][1]:9d}")
    print(f"  {'True High':15s} {cm_svm[1][0]:8d}  {cm_svm[1][1]:9d}")

    # ── Feature importance ────────────────────────────────────────
    print("\n" + "="*60)
    print("  FEATURE IMPORTANCE (Random Forest, trained on full data)")
    print("="*60)
    rf.fit(X, y)
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1]
    for rank, i in enumerate(idx, 1):
        bar = '█' * int(imp[i] * 200)
        print(f"  {rank:2d}. {available[i]:20s}: {imp[i]:.4f}  {bar}")

    # ── Plots ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Model Evaluation — PPG Attention Detection", fontsize=14)

    # Confusion matrix RF
    ConfusionMatrixDisplay(cm_rf, display_labels=['Low MWL', 'High MWL']).plot(
        ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion Matrix — Random Forest")

    # Confusion matrix SVM
    ConfusionMatrixDisplay(cm_svm, display_labels=['Low MWL', 'High MWL']).plot(
        ax=axes[1], colorbar=False)
    axes[1].set_title("Confusion Matrix — SVM")

    # Feature importance
    axes[2].barh([available[i] for i in idx], imp[idx], color='steelblue')
    axes[2].set_title("Feature Importance (Random Forest)")
    axes[2].set_xlabel("Importance score")
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig("results.png", dpi=150, bbox_inches='tight')
    print("\n  Plot saved: results.png")
    plt.show()
    print("\nDone!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='features_dataset.csv')
    args = parser.parse_args()
    main(args.data)