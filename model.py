import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# ── Energy Prediction Model (Regression) ──────────────────────────────
class EnergyModel(nn.Module):
    def __init__(self):
        super(EnergyModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


# ── Activity Detection Model (Classification) ─────────────────────────
class ActivityModel(nn.Module):
    def __init__(self):
        super(ActivityModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)   # 3 classes: sleeping, active, away
        )

    def forward(self, x):
        return self.network(x)


# ── Load and prepare data for one client ──────────────────────────────
def load_client_data(home_id):
    df = pd.read_csv(f"data/client_{home_id}.csv")

    features = ["hour", "temperature", "motion", "light_usage"]
    X = df[features].values.astype(np.float32)
    y_energy   = df["energy"].values.astype(np.float32)
    y_activity = df["activity"].values.astype(np.int64)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # Convert to tensors
    X_tensor        = torch.tensor(X)
    y_energy_tensor = torch.tensor(y_energy).unsqueeze(1)
    y_activity_tensor = torch.tensor(y_activity)

    # Split 80% train, 20% test
    split = int(0.8 * len(X_tensor))
    return (
        X_tensor[:split], y_energy_tensor[:split], y_activity_tensor[:split],
        X_tensor[split:], y_energy_tensor[split:], y_activity_tensor[split:]
    )


# ── Train one round locally ────────────────────────────────────────────
def train(energy_model, activity_model, X_train, y_energy, y_activity, epochs=5):
    energy_optimizer   = torch.optim.Adam(energy_model.parameters(),   lr=0.001)
    activity_optimizer = torch.optim.Adam(activity_model.parameters(), lr=0.001)

    energy_loss_fn   = nn.MSELoss()
    activity_loss_fn = nn.CrossEntropyLoss()

    energy_model.train()
    activity_model.train()

    for epoch in range(epochs):
        # Train energy model
        energy_optimizer.zero_grad()
        energy_pred = energy_model(X_train)
        energy_loss = energy_loss_fn(energy_pred, y_energy)
        energy_loss.backward()
        energy_optimizer.step()

        # Train activity model
        activity_optimizer.zero_grad()
        activity_pred = activity_model(X_train)
        activity_loss = activity_loss_fn(activity_pred, y_activity)
        activity_loss.backward()
        activity_optimizer.step()

    return float(energy_loss), float(activity_loss)


# ── Evaluate models ────────────────────────────────────────────────────
def evaluate(energy_model, activity_model, X_test, y_energy, y_activity):
    energy_model.eval()
    activity_model.eval()

    with torch.no_grad():
        # Energy: lower MSE = better
        energy_pred = energy_model(X_test)
        energy_mse  = float(nn.MSELoss()(energy_pred, y_energy))

        # Activity: accuracy
        activity_pred    = activity_model(X_test)
        predicted_labels = torch.argmax(activity_pred, dim=1)
        accuracy = float((predicted_labels == y_activity).float().mean())

    return energy_mse, accuracy


# ── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing model with Home 1 data...")

    X_train, y_e_train, y_a_train, X_test, y_e_test, y_a_test = load_client_data(1)

    energy_model   = EnergyModel()
    activity_model = ActivityModel()

    e_loss, a_loss = train(energy_model, activity_model,
                           X_train, y_e_train, y_a_train, epochs=10)

    e_mse, accuracy = evaluate(energy_model, activity_model,
                                X_test, y_e_test, y_a_test)

    print(f"  Energy MSE  : {e_mse:.4f}")
    print(f"  Activity Acc: {accuracy:.4f}")
    print("Model test passed!")