import flwr as fl
import numpy as np
import json
import os
from model import EnergyModel, ActivityModel

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ── Track metrics across rounds ────────────────────────────────────────
round_metrics = {
    "rounds":       [],
    "accuracy":     [],
    "energy_mse":   []
}


def fit_config(server_round):
    return {"round": server_round}


def evaluate_config(server_round):
    return {"round": server_round}


def get_evaluate_fn():
    """Return a function that the server uses to evaluate the global model."""
    def evaluate(server_round, parameters, config):
        print(f"\nRound {server_round} global evaluation complete")
        return 0.0, {}
    return evaluate


class SaveMetricsStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy that saves metrics after every round."""

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        if results:
            accuracies = [r.metrics["accuracy"] for _, r in results
                          if "accuracy" in r.metrics]
            losses     = [r.loss for _, r in results]

            avg_accuracy = float(np.mean(accuracies)) if accuracies else 0.0
            avg_loss     = float(np.mean(losses))     if losses     else 0.0

            round_metrics["rounds"].append(server_round)
            round_metrics["accuracy"].append(round(avg_accuracy, 4))
            round_metrics["energy_mse"].append(round(avg_loss, 4))

            print(f"  Round {server_round} | "
                  f"Avg Accuracy: {avg_accuracy:.4f} | "
                  f"Avg Energy MSE: {avg_loss:.4f}")

            # Save metrics to JSON after every round
            with open("logs/metrics.json", "w") as f:
                json.dump(round_metrics, f, indent=2)

        return aggregated


def run_server():
    # Build initial model parameters
    energy_model   = EnergyModel()
    activity_model = ActivityModel()

    initial_params = (
        [p.detach().numpy() for p in energy_model.parameters()] +
        [p.detach().numpy() for p in activity_model.parameters()]
    )

    strategy = SaveMetricsStrategy(
        fraction_fit=1.0,           # use all available clients
        fraction_evaluate=1.0,
        min_fit_clients=5,          # wait for all 5 homes
        min_evaluate_clients=5,
        min_available_clients=5,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_params)
    )

    print("Starting Federated Learning Server...")
    print("Waiting for 5 smart home clients...\n")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )

    print("\nFederated Learning complete!")
    print(f"Metrics saved to logs/metrics.json")


if __name__ == "__main__":
    run_server()