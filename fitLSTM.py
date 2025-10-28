import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# =======================
# Dataset Loader
# =======================
class CryptoDataset(Dataset):
    def __init__(self, data, target_col, window=30):
        self.features = data.drop(columns=[target_col, "Date"]).values
        self.target = data[target_col].values
        self.window = window

    def __len__(self):
        return len(self.features) - self.window

    def __getitem__(self, idx):
        X = self.features[idx:idx+self.window]
        y = self.target[idx+self.window]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# =======================
# LSTM Model
# =======================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# =======================
# Loss functions
# =======================
def mae_loss(y_hat, y_true):
    return torch.mean(torch.abs(y_hat - y_true))

def qlike_loss(sigma_hat, sigma_true, eps=1e-12):
    ratio = (sigma_true.clamp_min(eps) ** 2) / (sigma_hat.clamp_min(eps) ** 2)
    return torch.mean(ratio - torch.log(ratio) - 1)

# =======================
# Evaluation Function
# =======================
def evaluate_on_loader(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in loader:
            y_hat = model(X).squeeze()
            total_loss += loss_fn(y_hat, y).item() * len(X)
    return total_loss / len(loader.dataset)

# =======================
# Main Pipeline
# =======================
cryptos = ["BTC", "ETH", "BNB", "XRP"]
target_types = ["Log Returns", "RV"]
window_size = 30
max_epochs = 100  # maximum epochs for validation
patience = 10

# Define chronological splits
train_end = pd.Timestamp("2022-12-31")
val_end = pd.Timestamp("2023-12-31")
test_start = pd.Timestamp("2024-01-01")

results = []

for crypto in cryptos:
    df_crypto = pd.read_csv(f"Data/Verified/Verif_{crypto}.csv", parse_dates=["Date"])
    df_usdt = pd.read_csv("Data/Verified/Verif_USDT.csv", parse_dates=["Date"])

    # Select crypto features
    crypto_cols = ["Log Returns", "RV"]
    crypto_cols = [c for c in crypto_cols if c in df_crypto.columns]
    df_crypto = df_crypto[["Date"] + crypto_cols]

    # Select USDT features
    usdt_cols = ["RV"]
    usdt_cols = [c for c in usdt_cols if c in df_usdt.columns]
    df_usdt = df_usdt[["Date"] + usdt_cols]
    df_usdt = df_usdt.rename(columns={c: f"{c}_USDT" for c in usdt_cols})

    # Merge crypto and USDT
    df_merged = pd.merge(df_crypto, df_usdt, on="Date", how="inner")
    df_merged = df_merged.fillna(method="ffill").dropna()

    # Split train/val/test chronologically
    train_df = df_merged[df_merged["Date"] <= train_end].reset_index(drop=True)
    val_df = df_merged[(df_merged["Date"] > train_end) & (df_merged["Date"] <= val_end)].reset_index(drop=True)
    test_df = df_merged[df_merged["Date"] >= test_start].reset_index(drop=True)

    for target in target_types:
        loss_dict = {"Crypto": crypto, "Target": target}

        for with_usdt in [True, False]:
            if with_usdt:
                features_train = train_df.copy()
                features_val = val_df.copy()
                features_test = test_df.copy()
                print(f'Running LSTM for {crypto} {target} using Stablecoin data')
            else:
                drop_cols = [c for c in df_merged.columns if "_USDT" in c]
                features_train = train_df.drop(columns=drop_cols)
                features_val = val_df.drop(columns=drop_cols)
                features_test = test_df.drop(columns=drop_cols)
                print(f'Running LSTM for {crypto} {target} without using Stablecoin data')

            # Scale features based on training data
            feature_cols = [c for c in features_train.columns if c != target and c != "Date"]
            scaler = StandardScaler()
            scaled_train_values = scaler.fit_transform(features_train[feature_cols])
            scaled_val_values = scaler.transform(features_val[feature_cols])
            scaled_test_values = scaler.transform(features_test[feature_cols])

            scaled_train = pd.DataFrame(scaled_train_values, columns=feature_cols)
            scaled_train[target] = features_train[target].values
            scaled_train["Date"] = features_train["Date"].values

            scaled_val = pd.DataFrame(scaled_val_values, columns=feature_cols)
            scaled_val[target] = features_val[target].values
            scaled_val["Date"] = features_val["Date"].values

            scaled_test = pd.DataFrame(scaled_test_values, columns=feature_cols)
            scaled_test[target] = features_test[target].values
            scaled_test["Date"] = features_test["Date"].values

            # Create datasets and loaders
            train_dataset = CryptoDataset(scaled_train, target_col=target, window=window_size)
            val_dataset = CryptoDataset(scaled_val, target_col=target, window=window_size)
            test_dataset = CryptoDataset(scaled_test, target_col=target, window=window_size)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # Initialize model
            model = LSTMModel(input_size=len(feature_cols))
            loss_fn = mae_loss if target == "Log Returns" else qlike_loss
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Train with validation-based early stopping
            # Train with validation-based early stopping
            best_val_loss = float("inf")
            best_model_state = None
            best_epoch = 0
            epochs_no_improve = 0  # Track stagnation

            for epoch in range(1, max_epochs + 1):
                model.train()
                for X, y in train_loader:
                    optimizer.zero_grad()
                    y_hat = model(X).squeeze()
                    loss = loss_fn(y_hat, y)
                    loss.backward()
                    optimizer.step()

                # Evaluate on validation set
                val_loss = evaluate_on_loader(model, val_loader, loss_fn)
                
                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                    best_epoch = epoch
                    epochs_no_improve = 0  # Reset counter
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f'Early stopping at epoch {epoch} (no improvement for {patience} epochs)')
                        break

            # Load best model based on validation loss
            model.load_state_dict(best_model_state)

            # Evaluate on test data
            test_loss = evaluate_on_loader(model, test_loader, loss_fn)

            if with_usdt:

                col_name = "Loss with USDT"
                loss_dict["Optimal Epoch with USDT"] = best_epoch
            else:

                col_name = "Loss without USDT"
                loss_dict["Optimal Epoch without USDT"] = best_epoch

           
            loss_dict[col_name] = test_loss

        results.append(loss_dict)

# Convert results to DataFrame

results_df = pd.DataFrame(results)
results_df['Stablecoin Improvement %'] = (1 - (results_df['Loss with USDT'] / results_df['Loss without USDT'])) * 100
print("\n===== Summary Table =====")
print(results_df)