import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, f1_score, hamming_loss, jaccard_score
)

# ── CONFIGURATION & REPRODUCIBILITY ──────────────────────────────────────────
RANDOM_SEED    = 42
BEHAVIORS      = ["standing", "eating", "moving", "resting", "interacting"]
MODEL_PATH     = "./models_splits/multi_label_mlp.pth"
DATA_SPLIT_NPZ = "./models_splits/dataset_splits.npz"
BATCH_SIZE     = 64
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
CONSENSUS_RULE = 0.5  

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(RANDOM_SEED)

# ── 1. DATA LOADING & LOCAL BINARIZATION ─────────────────────────────────────
print("Loading frozen data splits...")
data = np.load(DATA_SPLIT_NPZ)
X_cal, Y_cal_soft   = data["X_cal"], data["Y_cal"]
X_test, Y_test_soft = data["X_test"], data["Y_test"]

# Dynamically apply threshold to create hard evaluation targets
Y_cal_binary  = (Y_cal_soft >= CONSENSUS_RULE).astype(int)
Y_test_binary = (Y_test_soft >= CONSENSUS_RULE).astype(int)

class InferenceDataset(Dataset):
    def __init__(self, embeddings):
        self.X = torch.tensor(embeddings, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx]

class MultiLabelMLP(nn.Module):
    def __init__(self, input_dim=1024, num_classes=5):
        super(MultiLabelMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),       nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.LayerNorm(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.network(x)

cal_loader  = DataLoader(InferenceDataset(X_cal), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(InferenceDataset(X_test), batch_size=BATCH_SIZE, shuffle=False)

# Load Model
model = MultiLabelMLP(input_dim=X_cal.shape[1], num_classes=5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def extract_probabilities(dataloader):
    all_probs = []
    with torch.no_grad():
        for batch_X in dataloader:
            batch_X = batch_X.to(DEVICE)
            probs   = torch.sigmoid(model(batch_X))
            all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)

# ── 2. CALIBRATE DECISION THRESHOLDS ─────────────────────────────────────────
print("\n=== Phase 1: Grid Search Threshold Optimization (Calibration Split) ===")
cal_probs = extract_probabilities(cal_loader)
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

# ── 3. EVALUATE ON UNSEEN TEST SET ───────────────────────────────────────────
print("\n=== Phase 2: Final Evaluation (Test Split) ===")
test_probs = extract_probabilities(test_loader)
test_preds = (test_probs >= cal_thresholds).astype(int)

print("\n── Structural Global Multi-Label Metrics ──")
print(f"  Exact Match Ratio (Subset Accuracy): {accuracy_score(Y_test_binary, test_preds) * 100:.2f}%")
print(f"  Jaccard Score (Sample Average):     {jaccard_score(Y_test_binary, test_preds, average='samples', zero_division=0):.4f}")
print(f"  Hamming Loss (lower is better):     {hamming_loss(Y_test_binary, test_preds):.4f}")
print(f"  Macro F1-Score:                     {f1_score(Y_test_binary, test_preds, average='macro'):.4f}")

print("\n── Detailed Per-Class Classification Profiles ──")
for i, name in enumerate(BEHAVIORS):
    auc = roc_auc_score(Y_test_binary[:, i], test_probs[:, i])
    print(f"\nBehavior Subtype: {name.upper()} (Area Under ROC: {auc:.3f})")
    print(classification_report(Y_test_binary[:, i], test_preds[:, i], target_names=["no", "yes"], zero_division=0))