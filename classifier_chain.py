import numpy as np
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score, f1_score, hamming_loss, jaccard_score
)
from xgboost import XGBClassifier

# ── CONFIGURATION ────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
DATA_SPLIT_NPZ = "./models_splits/dataset_splits.npz"
BEHAVIORS      = ["standing", "eating", "moving", "resting", "interacting"]
CONSENSUS_RULE = 0.5

# ── 1. LOAD EXACT SPLITS & DYNAMIC BINARIZATION ──────────────────────────────
print("Loading frozen data splits from deep learning pipeline...")
data = np.load(DATA_SPLIT_NPZ)

X_train, Y_train_soft = data["X_train"], data["Y_train"]
X_cal, Y_cal_soft     = data["X_cal"], data["Y_cal"]
X_test, Y_test_soft   = data["X_test"], data["Y_test"]

# Convert all continuous targets to discrete binary targets using the consensus rule
Y_train_binary = (Y_train_soft >= CONSENSUS_RULE).astype(int)
Y_cal_binary   = (Y_cal_soft >= CONSENSUS_RULE).astype(int)
Y_test_binary  = (Y_test_soft >= CONSENSUS_RULE).astype(int)

print(f"Verified Setup -> Train: {len(X_train)} | Calibration: {len(X_cal)} | Test: {len(X_test)}")

# ── 2. CALCULATE SCALE POS WEIGHT ────────────────────────────────────────────
ratio = (Y_train_binary == 0).sum() / (Y_train_binary == 1).sum()

# ── 3. INITIALIZE XGBOOST & CLASSIFIER CHAIN ─────────────────────────────────
base_xgb = XGBClassifier(
    scale_pos_weight=ratio,
    random_state=RANDOM_SEED,
    eval_metric="logloss",
    subsample=0.8,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.3,
    min_child_weight=0.3,
    colsample_bytree=1,
    n_jobs=-1  # Multi-threaded execution
)

chain = ClassifierChain(
    base_xgb,
    order=[0, 1, 2, 3, 4],
    random_state=RANDOM_SEED
)

# ── 4. TRAINING EXECUTION ────────────────────────────────────────────────────
print("\nTraining XGBoost ClassifierChain...")
chain.fit(X_train, Y_train_binary)

# ── 5. PHASE 1: THRESHOLD CALIBRATION (CALIBRATION SPLIT) ────────────────────
print("\n=== Phase 1: Grid Search Threshold Optimization (Calibration Split) ===")
cal_probs = chain.predict_proba(X_cal)
cal_thresholds = np.zeros(5)

threshold_grid = np.linspace(0.01, 0.99, 100)
for i, name in enumerate(BEHAVIORS):
    best_f1, best_th = -1, 0.5
    for th in threshold_grid:
        current_preds = (cal_probs[:, i] >= th).astype(int)
        score = f1_score(Y_cal_binary[:, i], current_preds, zero_division=0)
        if score > best_f1:
            best_f1, best_th = score, th
    cal_thresholds[i] = best_th
    print(f"  {name:<12} -> Optimal Threshold: {best_th:.2f} (Calibration F1: {best_f1:.3f})")

# ── 6. PHASE 2: EVALUATING UNSEEN TEST SET WITH CALIBRATED THRESHOLDS ────────
print("\n=== Phase 2: Final Evaluation with Calibrated Thresholds (Test Split) ===")
test_probs = chain.predict_proba(X_test)

# Vectorized threshold evaluation via broadcasting
test_preds = (test_probs >= cal_thresholds).astype(int)

print("\n── Structural Global Multi-Label Metrics ──")
print(f"  Exact Match Ratio (Subset Accuracy): {accuracy_score(Y_test_binary, test_preds) * 100:.2f}%")
print(f"  Jaccard Score (Sample Average):     {jaccard_score(Y_test_binary, test_preds, average='samples', zero_division=0):.4f}")
print(f"  Hamming Loss (lower is better):     {hamming_loss(Y_test_binary, test_preds):.4f}")
print(f"  Macro F1-Score:                     {f1_score(Y_test_binary, test_preds, average='macro'):.4f}")

print("\n── Detailed Per-Class Classification Profiles ──")
for i, name in enumerate(BEHAVIORS):
    auc = roc_auc_score(Y_test_binary[:, i], test_probs[:, i])
    print(f"\nBehavior Subtype: {name.upper()} (Area Under ROC: {auc:.3f} | Threshold Applied: {cal_thresholds[i]:.2f})")
    print(classification_report(Y_test_binary[:, i], test_preds[:, i], target_names=["no", "yes"], zero_division=0))

all_zero = (test_preds.sum(axis=1) == 0).sum()
print(f"\nAll-zero predictions: {all_zero} / {len(X_test)} ({100*all_zero/len(X_test):.1f}%)")