import h5py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score

# ── CONFIGURATION & GLOBAL REPRODUCIBILITY SEED ──────────────────────────────
RANDOM_SEED   = 42
HDF5_PATH     = "../embeddings/ser_data_2_embeddings.h5"
MODEL_PATH    = "./models_splits/multi_label_mlp.pth"
DATA_SPLIT_NPZ = "./models_splits/dataset_splits.npz"

BATCH_SIZE    = 64
EPOCHS        = 2000       
VAL_EVERY     = 100        
LEARNING_RATE = 1e-3
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CONSENSUS_RULE = 0.5      

# ── SEED INITIALIZATION FUNCTION ─────────────────────────────────────────────
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(RANDOM_SEED)

# ── 1. DATA PREPARATION (SOFT TARGET COMPATIBLE) ──────────────────────────────
print("Loading embeddings and soft consensus scores...")
with h5py.File(HDF5_PATH, "r") as hf:
    X      = hf["embeddings"][:]
    labels = hf["labels"][:]

# Remove invalid all-zero feature frames
valid  = X.sum(axis=1) != 0
X      = X[valid]
labels = labels[valid]

# Retain raw continuous consensus scores (float32) for the first 5 behaviors
Y_soft = labels[:, :5].astype(np.float32)

# Filter out frames that have completely negligible target activity
has_signal = Y_soft.sum(axis=1) > 1e-3
X_filtered = X[has_signal]
Y_filtered = Y_soft[has_signal]

# Split Strategy: 60% Train, 20% Calibration (Validation), 20% Test
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X_filtered, Y_filtered, test_size=0.4, random_state=RANDOM_SEED
)
X_cal, X_test, Y_cal, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=RANDOM_SEED
)

# Archive ONLY the raw feature arrays and continuous consensus labels
np.savez(
    DATA_SPLIT_NPZ, 
    X_train=X_train, Y_train=Y_train,
    X_cal=X_cal,     Y_cal=Y_cal,
    X_test=X_test,   Y_test=Y_test
)
print(f"Dataset splits archived to {DATA_SPLIT_NPZ} (Soft targets only)")
print(f"  Sizes -> Train: {len(X_train)} | Cal: {len(X_cal)} | Test: {len(X_test)}")

# ── 2. EXPECTED FREQUENCY IMBALANCE SCALING ──────────────────────────────────
expected_positives = Y_train.sum(axis=0)
total_samples      = Y_train.shape[0]
expected_negatives = total_samples - expected_positives
pos_weight         = expected_negatives / (expected_positives + 1e-6)

pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)

# ── 3. PYTORCH COMPONENTS ────────────────────────────────────────────────────
class SoftLabelDataset(Dataset):
    def __init__(self, embeddings, soft_labels):
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.Y = torch.tensor(soft_labels, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class MultiLabelMLP(nn.Module):
    def __init__(self, input_dim=1024, num_classes=5):
        super(MultiLabelMLP, self).__init__()
        if input_dim != 1024:
            print(f"[Warning] Embedded input feature dimension detected is {input_dim}, adapting entry layer...")

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),       nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(RANDOM_SEED)

train_loader = DataLoader(
    SoftLabelDataset(X_train, Y_train), 
    batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=seed_worker, generator=g
)
val_loader = DataLoader(SoftLabelDataset(X_cal, Y_cal), batch_size=BATCH_SIZE, shuffle=False)

model     = MultiLabelMLP(input_dim=X_train.shape[1], num_classes=5).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# Calculate discrete validation targets once locally for training metrics
Y_cal_binary = (Y_cal >= CONSENSUS_RULE).astype(int)

# ── 4. TRAINING & INTERMITTENT VALIDATION ENGINE ─────────────────────────────
print(f"\nTraining joint network on {DEVICE}...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(DEVICE), batch_Y.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(batch_X)
        loss   = criterion(logits, batch_Y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_X.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    current_epoch = epoch + 1
    if current_epoch % VAL_EVERY == 0 or epoch == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n[Epoch {current_epoch:04d}/{EPOCHS}] Train Loss: {epoch_loss:.4f} | Current LR: {current_lr:.6f}")
        
        model.eval()
        val_probs = []
        
        with torch.no_grad():
            for val_batch_X, _ in val_loader:
                val_batch_X = val_batch_X.to(DEVICE)
                val_logits  = model(val_batch_X)
                probs       = torch.sigmoid(val_logits)
                val_probs.append(probs.cpu().numpy())
        
        val_probs = np.vstack(val_probs)
        val_preds = (val_probs >= CONSENSUS_RULE).astype(int)
        
        exact_match  = accuracy_score(Y_cal_binary, val_preds)  
        h_loss       = hamming_loss(Y_cal_binary, val_preds)
        jaccard_samp = jaccard_score(Y_cal_binary, val_preds, average="samples", zero_division=0)
        
        print(" └── Validation Metrics (Baseline 0.5 Threshold):")
        print(f"      Exact Match Ratio (Subset Accuracy): {exact_match * 100:.2f}%")
        print(f"      Jaccard Score (Sample Average):     {jaccard_samp:.4f}")
        print(f"      Hamming Loss (Lower is better):     {h_loss:.4f}")
        
    scheduler.step()

torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel tracking complete. Target weights saved to {MODEL_PATH}")