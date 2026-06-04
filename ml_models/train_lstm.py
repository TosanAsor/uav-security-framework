import pandas as pd
import numpy as np
import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

# ── CONFIGURATION ──
PROCESSED_DIR = '../data/processed/'
RESULTS_DIR   = '../results/'
WINDOW_SIZE   = 20
BATCH_SIZE    = 64
EPOCHS        = 30
LEARNING_RATE = 0.001
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── LOAD DATA ──
print("Loading processed data...")
all_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, '*.csv')))
print(f"Found {len(all_files)} files\n")

dfs = []
for f in all_files:
    df = pd.read_csv(f)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(data)}")
print(f"Label distribution:\n{data['label'].value_counts()}\n")

# ── FEATURES ──
features = [
    'residual_lat',
    'residual_lon',
    'residual_alt',
    'residual_mag',
    'residual_mag_mean',
    'residual_mag_std',
    'gps_fix_type',
    'gps_satellites',
    'imu_acc_mag',
    'baro_pressure',
]

X_raw = data[features].fillna(0).values
y_raw = data['label'].values

# ── SCALE ──
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ── CREATE SEQUENCES ──
print(f"Creating sequences with window size {WINDOW_SIZE}...")

def create_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i + window])
        ys.append(y[i + window])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_raw, WINDOW_SIZE)
print(f"Sequences created: {X_seq.shape}")
print(f"Label distribution in sequences:\n{pd.Series(y_seq).value_counts()}\n")

# ── TRAIN / TEST SPLIT WITH STRATIFICATION ──
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

print(f"Training sequences: {len(X_train)}")
print(f"Testing sequences:  {len(X_test)}")
print(f"Test label distribution:\n{pd.Series(y_test).value_counts()}\n")

# ── DATASET ──
class SensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SensorDataset(X_train, y_train)
test_dataset  = SensorDataset(X_test,  y_test)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ── LSTM MODEL ──
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.3)
        self.fc1  = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fc2  = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = out[:, -1, :]
        out    = self.fc1(out)
        out    = self.relu(out)
        out    = self.drop(out)
        out    = self.fc2(out)
        return out

INPUT_SIZE  = len(features)
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
NUM_CLASSES = 3

device = torch.device('cpu')
model  = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE,
                        NUM_LAYERS, NUM_CLASSES).to(device)
print(f"Model architecture:\n{model}\n")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.5)

# ── TRAINING ──
print("=" * 50)
print("TRAINING LSTM")
print("=" * 50)

train_losses = []
train_accs   = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds       = outputs.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += y_batch.size(0)

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    train_losses.append(avg_loss)
    train_accs.append(accuracy)

    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} — "
              f"Loss: {avg_loss:.4f} — "
              f"Accuracy: {accuracy*100:.2f}%")

# ── EVALUATION ──
print("\n" + "=" * 50)
print("EVALUATION")
print("=" * 50)

model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

unique_labels = sorted(np.unique(all_labels))
label_names   = {0: 'Normal', 1: 'Spoofing', 2: 'Jamming'}
target_names  = [label_names[l] for l in unique_labels]
n_classes     = len(unique_labels)

lstm_accuracy = accuracy_score(all_labels, all_preds)
lstm_f1       = f1_score(all_labels, all_preds, average='weighted')

print(f"Accuracy : {lstm_accuracy:.4f} ({lstm_accuracy*100:.2f}%)")
print(f"F1 Score : {lstm_f1:.4f}")
print()
print("Classification Report:")
print(classification_report(all_labels, all_preds,
      labels=unique_labels, target_names=target_names))

cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)
print("Confusion Matrix:")
print(cm)

# ── SAVE MODEL ──
torch.save(model.state_dict(),
           os.path.join(RESULTS_DIR, 'lstm_model.pt'))
with open(os.path.join(RESULTS_DIR, 'lstm_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print("\nLSTM model saved.")

# ── PLOT TRAINING CURVE ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, color='steelblue')
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2.plot([a * 100 for a in train_accs], color='green')
ax2.set_title('Training Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
plt.suptitle('LSTM Training Curves')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,
            'lstm_training_curves.png'), dpi=150)
plt.close()

# ── PLOT CONFUSION MATRIX ──
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(cm, cmap='Oranges')
ax.set_xticks(range(n_classes))
ax.set_yticks(range(n_classes))
ax.set_xticklabels(target_names)
ax.set_yticklabels(target_names)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'LSTM Confusion Matrix\n'
             f'Accuracy: {lstm_accuracy*100:.2f}% | '
             f'F1: {lstm_f1:.4f}')
for i in range(n_classes):
    for j in range(n_classes):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black',
                fontsize=14, fontweight='bold')
plt.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,
            'lstm_confusion_matrix.png'), dpi=150)
plt.close()
print("Plots saved.")

print("\n" + "=" * 50)
print("RESULTS SUMMARY")
print("=" * 50)
print(f"LSTM Accuracy : {lstm_accuracy*100:.2f}%")
print(f"LSTM F1 Score : {lstm_f1:.4f}")
print(f"\nAll results saved to: {RESULTS_DIR}")
