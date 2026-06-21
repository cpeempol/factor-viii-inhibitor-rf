# factor_viii_inhibitor_analysis.py

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    roc_auc_score,
    roc_curve,
    r2_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import pearsonr


# ============================================================
# Configuration
# ============================================================

TARGET_COL = "factor_viii_inhibitor_bu"

FEATURE_COLS = [
    "sex",
    "age",
    "patient_aptt",
    "immediate_mix_aptt",
    "incubated_mix_aptt",
    "normal_pooled_plasma_aptt",
]

CATEGORICAL_FEATURES = ["sex"]

NUMERIC_FEATURES = [
    "age",
    "patient_aptt",
    "immediate_mix_aptt",
    "incubated_mix_aptt",
    "normal_pooled_plasma_aptt",
]

POS_CUTOFF = 0.6
ROSNER_CUTOFF = 15
PCTCORR_CUTOFF = 70

N_SPLITS = 5
RANDOM_STATE = 42
N_BOOT = 5000


# ============================================================
# Helper functions
# ============================================================

def make_preprocess():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )


def safe_pearson(x, y_pred):
    x = np.asarray(x, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if x.size < 3:
        return np.nan, np.nan

    if np.isclose(np.std(x), 0) or np.isclose(np.std(y_pred), 0):
        return np.nan, np.nan

    r, p = pearsonr(x, y_pred)
    return float(r), float(p)


def compute_class_metrics(y_true_bin, y_pred_bin):
    tn, fp, fn, tp = confusion_matrix(
        y_true_bin,
        y_pred_bin,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
    }


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan

    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(
        (p * (1 - p) / n) + (z**2 / (4 * n**2))
    ) / denom

    return center - half, center + half


def bootstrap_auc_ci(y_true_bin, scores, n_boot=N_BOOT, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    y_true_bin = np.asarray(y_true_bin).astype(int)
    scores = np.asarray(scores).astype(float)

    n = len(y_true_bin)
    aucs = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)

        if len(np.unique(y_true_bin[idx])) < 2:
            continue

        aucs.append(roc_auc_score(y_true_bin[idx], scores[idx]))

    aucs = np.asarray(aucs)

    return {
        "auc": roc_auc_score(y_true_bin, scores),
        "auc_ci_low": np.percentile(aucs, 2.5),
        "auc_ci_high": np.percentile(aucs, 97.5),
    }


def diagnostic_metrics_with_ci(y_true_bin, y_pred_bin, scores, method_name):
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_pred_bin = np.asarray(y_pred_bin).astype(int)
    scores = np.asarray(scores).astype(float)

    tn, fp, fn, tp = confusion_matrix(
        y_true_bin,
        y_pred_bin,
        labels=[0, 1],
    ).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    sens_low, sens_high = wilson_ci(tp, tp + fn)
    spec_low, spec_high = wilson_ci(tn, tn + fp)
    ppv_low, ppv_high = wilson_ci(tp, tp + fp)
    npv_low, npv_high = wilson_ci(tn, tn + fn)

    auc_info = bootstrap_auc_ci(y_true_bin, scores)

    return {
        "method": method_name,
        "n": len(y_true_bin),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "AUC": auc_info["auc"],
        "AUC_95CI_low": auc_info["auc_ci_low"],
        "AUC_95CI_high": auc_info["auc_ci_high"],
        "Sensitivity": sens,
        "Sensitivity_95CI_low": sens_low,
        "Sensitivity_95CI_high": sens_high,
        "Specificity": spec,
        "Specificity_95CI_low": spec_low,
        "Specificity_95CI_high": spec_high,
        "PPV": ppv,
        "PPV_95CI_low": ppv_low,
        "PPV_95CI_high": ppv_high,
        "NPV": npv,
        "NPV_95CI_low": npv_low,
        "NPV_95CI_high": npv_high,
    }


def paired_bootstrap_auc_difference(
    y_true_bin,
    score_a,
    score_b,
    n_boot=N_BOOT,
    seed=RANDOM_STATE,
):
    rng = np.random.default_rng(seed)

    y_true_bin = np.asarray(y_true_bin).astype(int)
    score_a = np.asarray(score_a).astype(float)
    score_b = np.asarray(score_b).astype(float)

    n = len(y_true_bin)
    diffs = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)

        if len(np.unique(y_true_bin[idx])) < 2:
            continue

        auc_a = roc_auc_score(y_true_bin[idx], score_a[idx])
        auc_b = roc_auc_score(y_true_bin[idx], score_b[idx])
        diffs.append(auc_a - auc_b)

    diffs = np.asarray(diffs)

    p_low = (np.sum(diffs <= 0) + 1) / (len(diffs) + 1)
    p_high = (np.sum(diffs >= 0) + 1) / (len(diffs) + 1)
    p_value = min(1.0, 2 * min(p_low, p_high))

    return {
        "AUC_difference": np.mean(diffs),
        "AUC_difference_95CI_low": np.percentile(diffs, 2.5),
        "AUC_difference_95CI_high": np.percentile(diffs, 97.5),
        "bootstrap_p_value": p_value,
    }


def format_ci(value, low, high, decimals=3):
    return f"{value:.{decimals}f} ({low:.{decimals}f}-{high:.{decimals}f})"


# ============================================================
# Main analysis
# ============================================================

def run_analysis(data_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    data = df[FEATURE_COLS + [TARGET_COL]].copy()
    data = data.replace(
        ["-", " - ", "", "NA", "N/A", "na", "n/a", "null", "None"],
        np.nan,
    )

    for col in NUMERIC_FEATURES + [TARGET_COL]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["sex"] = data["sex"].astype(str).str.strip().str.upper()
    data.loc[data["sex"].isin(["NAN", "NA", "NONE"]), "sex"] = np.nan

    data = data.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)

    X = data[FEATURE_COLS].copy()
    y = data[TARGET_COL].copy()

    y_bin = (y > POS_CUTOFF).astype(int)
    y_eval = y.where(y > POS_CUTOFF, 0.0).astype(float)

    print("Rows used:", len(data))
    print("Positive cases:", int(y_bin.sum()))
    print("Negative cases:", int((1 - y_bin).sum()))
    print("Positive percentage:", y_bin.mean() * 100)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_prob = np.zeros(len(data), dtype=float)
    oof_pred_bu = np.zeros(len(data), dtype=float)
    oof_pred_class = np.zeros(len(data), dtype=int)
    oof_threshold = np.zeros(len(data), dtype=float)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bin), start=1):
        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()

        y_train = y.iloc[train_idx].copy()
        y_val = y.iloc[val_idx].copy()

        y_train_bin = (y_train > POS_CUTOFF).astype(int)
        y_val_bin = (y_val > POS_CUTOFF).astype(int)
        y_val_eval = y_val.where(y_val > POS_CUTOFF, 0.0).astype(float)

        clf_pipe = Pipeline([
            ("preprocess", make_preprocess()),
            ("model", RandomForestClassifier(
                n_estimators=1200,
                max_depth=None,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            )),
        ])

        clf_pipe.fit(X_train, y_train_bin)

        pos_mask = y_train > POS_CUTOFF

        reg_pipe = Pipeline([
            ("preprocess", make_preprocess()),
            ("model", RandomForestRegressor(
                n_estimators=2000,
                max_depth=None,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])

        reg_pipe.fit(X_train.loc[pos_mask], y_train.loc[pos_mask])

        p_pos = clf_pipe.predict_proba(X_val)[:, 1]

        def predict_with_threshold(threshold):
            is_positive = p_pos >= threshold
            y_pred = np.zeros(len(X_val), dtype=float)

            if is_positive.any():
                pred_positive = reg_pipe.predict(X_val[is_positive])
                pred_positive = np.clip(pred_positive, 0, None)
                pred_positive = np.maximum(pred_positive, POS_CUTOFF)
                y_pred[is_positive] = pred_positive

            return y_pred

        best_threshold = None
        best_mae = np.inf
        best_pred_bu = None
        best_pred_class = None

        for threshold in np.arange(0.10, 0.91, 0.05):
            pred_bu = predict_with_threshold(threshold)
            mae = mean_absolute_error(y_val_eval, pred_bu)

            if mae < best_mae:
                best_mae = mae
                best_threshold = float(threshold)
                best_pred_bu = pred_bu
                best_pred_class = (p_pos >= threshold).astype(int)

        auc = roc_auc_score(y_val_bin, p_pos)
        cls_metrics = compute_class_metrics(y_val_bin.values, best_pred_class)

        oof_prob[val_idx] = p_pos
        oof_pred_bu[val_idx] = best_pred_bu
        oof_pred_class[val_idx] = best_pred_class
        oof_threshold[val_idx] = best_threshold

        fold_results.append({
            "fold": fold,
            "n_validation": len(X_val),
            "positive_validation_percent": y_val_bin.mean() * 100,
            "auc": auc,
            "best_threshold": best_threshold,
            "mae_overall": best_mae,
            "tn": cls_metrics["tn"],
            "fp": cls_metrics["fp"],
            "fn": cls_metrics["fn"],
            "tp": cls_metrics["tp"],
            "sensitivity": cls_metrics["sensitivity"],
            "specificity": cls_metrics["specificity"],
            "ppv": cls_metrics["ppv"],
            "npv": cls_metrics["npv"],
        })

        print(
            f"Fold {fold}: AUC={auc:.4f}, "
            f"threshold={best_threshold:.2f}, MAE={best_mae:.4f}"
        )

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(os.path.join(output_dir, "fold_level_results.csv"), index=False)

    # Conventional indices
    patient_aptt = data["patient_aptt"].astype(float).values
    mix2 = data["incubated_mix_aptt"].astype(float).values
    npp = data["normal_pooled_plasma_aptt"].astype(float).values

    rosner_index = ((mix2 - npp) / patient_aptt) * 100
    percent_correction = ((patient_aptt - mix2) / (patient_aptt - npp)) * 100

    rosner_score = rosner_index
    percent_correction_score = -percent_correction

    rosner_pred = (rosner_index > ROSNER_CUTOFF).astype(int)
    percent_correction_pred = (percent_correction < PCTCORR_CUTOFF).astype(int)

    valid_mask = (
        np.isfinite(rosner_index)
        & np.isfinite(percent_correction)
        & np.isfinite(oof_prob)
    )

    diagnostic_rows = [
        diagnostic_metrics_with_ci(
            y_bin.values,
            oof_pred_class,
            oof_prob,
            "Random Forest",
        ),
        diagnostic_metrics_with_ci(
            y_bin.values[valid_mask],
            rosner_pred[valid_mask],
            rosner_score[valid_mask],
            "Rosner Index",
        ),
        diagnostic_metrics_with_ci(
            y_bin.values[valid_mask],
            percent_correction_pred[valid_mask],
            percent_correction_score[valid_mask],
            "Percent correction",
        ),
    ]

    diagnostic_df = pd.DataFrame(diagnostic_rows)
    diagnostic_df.to_csv(
        os.path.join(output_dir, "diagnostic_metrics_with_95CI.csv"),
        index=False,
    )

    diagnostic_table = pd.DataFrame({
        "Method": diagnostic_df["method"],
        "AUC (95% CI)": [
            format_ci(row.AUC, row.AUC_95CI_low, row.AUC_95CI_high)
            for row in diagnostic_df.itertuples()
        ],
        "Sensitivity (95% CI)": [
            format_ci(row.Sensitivity, row.Sensitivity_95CI_low, row.Sensitivity_95CI_high)
            for row in diagnostic_df.itertuples()
        ],
        "Specificity (95% CI)": [
            format_ci(row.Specificity, row.Specificity_95CI_low, row.Specificity_95CI_high)
            for row in diagnostic_df.itertuples()
        ],
        "PPV (95% CI)": [
            format_ci(row.PPV, row.PPV_95CI_low, row.PPV_95CI_high)
            for row in diagnostic_df.itertuples()
        ],
        "NPV (95% CI)": [
            format_ci(row.NPV, row.NPV_95CI_low, row.NPV_95CI_high)
            for row in diagnostic_df.itertuples()
        ],
        "TP": diagnostic_df["TP"],
        "FP": diagnostic_df["FP"],
        "TN": diagnostic_df["TN"],
        "FN": diagnostic_df["FN"],
    })

    diagnostic_table.to_csv(
        os.path.join(output_dir, "table3_diagnostic_metrics.csv"),
        index=False,
    )

    # Regression performance
    mae_all = mean_absolute_error(y_eval.values, oof_pred_bu)
    r2_all = r2_score(y_eval.values, oof_pred_bu)
    r_all, p_all = safe_pearson(y_eval.values, oof_pred_bu)

    positive_mask = y.values > POS_CUTOFF

    mae_pos = mean_absolute_error(y.values[positive_mask], oof_pred_bu[positive_mask])
    r2_pos = r2_score(y.values[positive_mask], oof_pred_bu[positive_mask])
    r_pos, p_pos = safe_pearson(y.values[positive_mask], oof_pred_bu[positive_mask])

    regression_df = pd.DataFrame([
        {
            "analysis": "All records; inhibitor-negative values set to 0",
            "MAE_BU": mae_all,
            "R2": r2_all,
            "Pearson_r": r_all,
            "Pearson_p": p_all,
        },
        {
            "analysis": "Inhibitor-positive records only",
            "MAE_BU": mae_pos,
            "R2": r2_pos,
            "Pearson_r": r_pos,
            "Pearson_p": p_pos,
        },
    ])

    regression_df.to_csv(
        os.path.join(output_dir, "regression_performance_summary.csv"),
        index=False,
    )

    # ROC curve
    fpr_ml, tpr_ml, _ = roc_curve(y_bin.values[valid_mask], oof_prob[valid_mask])
    fpr_rosner, tpr_rosner, _ = roc_curve(
        y_bin.values[valid_mask],
        rosner_score[valid_mask],
    )
    fpr_pct, tpr_pct, _ = roc_curve(
        y_bin.values[valid_mask],
        percent_correction_score[valid_mask],
    )

    auc_ml = roc_auc_score(y_bin.values[valid_mask], oof_prob[valid_mask])
    auc_rosner = roc_auc_score(y_bin.values[valid_mask], rosner_score[valid_mask])
    auc_pct = roc_auc_score(y_bin.values[valid_mask], percent_correction_score[valid_mask])

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_ml, tpr_ml, label=f"Random Forest, AUC={auc_ml:.3f}")
    plt.plot(fpr_rosner, tpr_rosner, label=f"Rosner Index, AUC={auc_rosner:.3f}")
    plt.plot(fpr_pct, tpr_pct, label=f"Percent correction, AUC={auc_pct:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Receiver operating characteristic curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure2_roc_curve.png"), dpi=300)
    plt.close()

    # Log-transformed observed vs predicted plot
    plt.figure(figsize=(6, 6))
    plt.scatter(np.log1p(y_eval.values), np.log1p(oof_pred_bu), alpha=0.7)

    max_log = max(np.max(np.log1p(y_eval.values)), np.max(np.log1p(oof_pred_bu)))
    plt.plot([0, max_log], [0, max_log], linestyle="--", linewidth=1, color="gray")

    plt.xlabel("Observed inhibitor titer, log1p(BU)")
    plt.ylabel("Predicted inhibitor titer, log1p(BU)")
    plt.title("Log-transformed observed versus predicted inhibitor titer")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "figure1_observed_vs_predicted_log1p.png"),
        dpi=300,
    )
    plt.close()

    # Paired bootstrap AUC comparison
    auc_diff_rosner = paired_bootstrap_auc_difference(
        y_true_bin=y_bin.values[valid_mask],
        score_a=oof_prob[valid_mask],
        score_b=rosner_score[valid_mask],
        seed=RANDOM_STATE,
    )

    auc_diff_pct = paired_bootstrap_auc_difference(
        y_true_bin=y_bin.values[valid_mask],
        score_a=oof_prob[valid_mask],
        score_b=percent_correction_score[valid_mask],
        seed=RANDOM_STATE + 1,
    )

    auc_comparison_df = pd.DataFrame([
        {
            "comparison": "Random Forest - Rosner Index",
            **auc_diff_rosner,
        },
        {
            "comparison": "Random Forest - Percent correction",
            **auc_diff_pct,
        },
    ])

    auc_comparison_df.to_csv(
        os.path.join(output_dir, "paired_bootstrap_auc_comparison.csv"),
        index=False,
    )

    # Important: patient-level out-of-fold predictions are not saved.
    # This avoids sharing de-identified but potentially sensitive laboratory records.

    print("\nSaved aggregate outputs to:", output_dir)
    print("- fold_level_results.csv")
    print("- diagnostic_metrics_with_95CI.csv")
    print("- table3_diagnostic_metrics.csv")
    print("- regression_performance_summary.csv")
    print("- paired_bootstrap_auc_comparison.csv")
    print("- figure1_observed_vs_predicted_log1p.png")
    print("- figure2_roc_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Random Forest analysis for factor VIII inhibitor prediction."
    )
    parser.add_argument(
        "--data",
        default="data/example_input.csv",
        help="Path to anonymized or synthetic CSV input file.",
    )
    parser.add_argument(
        "--out",
        default="outputs",
        help="Output directory for aggregate results and figures.",
    )

    args = parser.parse_args()
    run_analysis(args.data, args.out)
