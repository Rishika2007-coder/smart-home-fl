import pandas as pd
import numpy as np
import os

# Make sure data folder exists
os.makedirs("data", exist_ok=True)

np.random.seed(42)

def generate_home_data(home_id, num_samples=500):
    """
    Generate realistic smart home sensor data for one house.
    Each house has slightly different patterns (different families).
    """

    hours = np.tile(np.arange(24), num_samples // 24 + 1)[:num_samples]

    # Temperature varies by time of day
    temperature = (
        20
        + 5 * np.sin(2 * np.pi * hours / 24)
        + np.random.normal(0, 1, num_samples)
        + home_id  # each home is slightly different
    )

    # Motion: more activity in morning (7-9am) and evening (6-9pm)
    motion_prob = np.where(
        (hours >= 7) & (hours <= 9), 0.85,
        np.where((hours >= 18) & (hours <= 21), 0.80,
        np.where((hours >= 22) | (hours <= 6), 0.05, 0.40))
    )
    motion = np.random.binomial(1, motion_prob, num_samples)

    # Light usage: on during evening, off at night
    light_usage = np.where(
        (hours >= 18) & (hours <= 23), np.random.uniform(0.6, 1.0, num_samples),
        np.where((hours >= 7) & (hours <= 9), np.random.uniform(0.3, 0.7, num_samples),
        np.random.uniform(0.0, 0.2, num_samples))
    )

    # Energy consumption (what we want to PREDICT)
    energy = (
        1.5
        + 0.3 * temperature
        + 0.8 * motion
        + 0.5 * light_usage
        + 0.2 * np.random.normal(0, 1, num_samples)
        + 0.1 * home_id  # each home consumes slightly differently
    )
    energy = np.clip(energy, 0.5, 10.0)  # realistic kWh range

    # Activity label for classification (0=sleeping, 1=home active, 2=away)
    activity = np.where(
        (hours >= 23) | (hours <= 6), 0,          # sleeping
        np.where(motion == 1, 1, 2)                # active or away
    )

    df = pd.DataFrame({
        "home_id":     home_id,
        "hour":        hours,
        "temperature": np.round(temperature, 2),
        "motion":      motion,
        "light_usage": np.round(light_usage, 2),
        "energy":      np.round(energy, 2),        # regression target
        "activity":    activity                    # classification target
    })

    return df


# ── Generate data for 5 homes (5 FL clients) ──────────────────────────
print("Generating smart home data for 5 clients...\n")

all_data = []

for home_id in range(1, 6):
    df = generate_home_data(home_id=home_id, num_samples=500)
    df.to_csv(f"data/client_{home_id}.csv", index=False)
    all_data.append(df)
    print(f"  Home {home_id}: {len(df)} samples saved → data/client_{home_id}.csv")

# Save a combined dataset too (useful for visualization later)
combined = pd.concat(all_data, ignore_index=True)
combined.to_csv("data/combined.csv", index=False)

print(f"\nTotal samples: {len(combined)}")
print("\nSample from Home 1:")
print(pd.read_csv("data/client_1.csv").head(8).to_string(index=False))
print("\nData generation complete!")