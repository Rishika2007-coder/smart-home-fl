import flwr as fl
import torch
import numpy as np
from model import (EnergyModel, ActivityModel,
                   load_client_data, train, evaluate)


class SmartHomeClient(fl.client.NumPyClient):
    def __init__(self, home_id):
        self.home_id = home_id
        self.energy_model   = EnergyModel()
        self.activity_model = ActivityModel()

        # Load this home's private data
        (self.X_train, self.y_energy_train, self.y_activity_train,
         self.X_test,  self.y_energy_test,  self.y_activity_test
        ) = load_client_data(home_id)

        print(f"  Home {home_id} client ready | "
              f"Train: {len(self.X_train)} | Test: {len(self.X_test)}")

    # ── Send model weights to server ───────────────────────────────────
    def get_parameters(self, config):
        energy_params   = [p.detach().numpy()
                           for p in self.energy_model.parameters()]
        activity_params = [p.detach().numpy()
                           for p in self.activity_model.parameters()]
        return energy_params + activity_params

    # ── Receive global weights from server & train locally ─────────────
    def fit(self, parameters, config):
        # Split parameters between the two models
        energy_param_count = len(list(self.energy_model.parameters()))
        energy_params      = parameters[:energy_param_count]
        activity_params    = parameters[energy_param_count:]

        # Load global weights into local models
        for local_p, global_p in zip(self.energy_model.parameters(), energy_params):
            local_p.data = torch.tensor(global_p)
        for local_p, global_p in zip(self.activity_model.parameters(), activity_params):
            local_p.data = torch.tensor(global_p)

        # Train locally on private data
        e_loss, a_loss = train(
            self.energy_model, self.activity_model,
            self.X_train, self.y_energy_train, self.y_activity_train,
            epochs=5
        )
        print(f"  Home {self.home_id} trained | "
              f"Energy loss: {e_loss:.4f} | Activity loss: {a_loss:.4f}")

        return self.get_parameters(config={}), len(self.X_train), {}

    # ── Evaluate global model on local private data ────────────────────
    def evaluate(self, parameters, config):
        energy_param_count = len(list(self.energy_model.parameters()))
        energy_params      = parameters[:energy_param_count]
        activity_params    = parameters[energy_param_count:]

        for local_p, global_p in zip(self.energy_model.parameters(), energy_params):
            local_p.data = torch.tensor(global_p)
        for local_p, global_p in zip(self.activity_model.parameters(), activity_params):
            local_p.data = torch.tensor(global_p)

        e_mse, accuracy = evaluate(
            self.energy_model, self.activity_model,
            self.X_test, self.y_energy_test, self.y_activity_test
        )
        print(f"  Home {self.home_id} eval | "
              f"Energy MSE: {e_mse:.4f} | Activity Acc: {accuracy:.4f}")

        return float(e_mse), len(self.X_test), {"accuracy": float(accuracy)}


# ── Start a client (called by server simulation) ───────────────────────
def start_client(home_id, server_address="127.0.0.1:8080"):
    client = SmartHomeClient(home_id)
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )


if __name__ == "__main__":
    import sys
    home_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"Starting FL client for Home {home_id}...")
    start_client(home_id)