import requests
import pandas as pd
import json
import os

# -----------------------------
# CONFIG
# -----------------------------
BASE_URL = "http://localhost:3000"

MACHINE_IDS = [
    "CNC_01",
    "CNC_02",
    "PUMP_03",
    "CONVEYOR_04"
]

OUTPUT_FILE = "baselines.json"


# -----------------------------
# CLEANING FUNCTION
# -----------------------------
def clean_data(df):
    # Keep only normal data
    df = df[df["status"] == "running"]

    # Drop missing
    df = df.dropna()

    # Remove unrealistic values
    df = df[
        (df["temperature_C"] > 0) & (df["temperature_C"] < 120) &
        (df["vibration_mm_s"] > 0) & (df["vibration_mm_s"] < 10) &
        (df["rpm"] > 0) &
        (df["current_A"] > 0) & (df["current_A"] < 30)
    ]

    return df


# -----------------------------
# COMPUTE BASELINES
# -----------------------------
def compute_baseline(machine_id):
    print(f"Processing {machine_id}...")

    # Fetch history
    response = requests.get(f"{BASE_URL}/history/{machine_id}")
    data = response.json()

    df = pd.DataFrame(data["readings"])

    # Clean
    df = clean_data(df)

    # Compute stats
    mean = df[["temperature_C", "vibration_mm_s", "current_A"]].mean()
    var  = df[["temperature_C", "vibration_mm_s", "current_A"]].var()
    std  = df[["temperature_C", "vibration_mm_s", "current_A"]].std()

    return {
        "mean": mean.to_dict(),
        "variance": var.to_dict(),
        "std": std.to_dict()
    }


# -----------------------------
# MAIN
# -----------------------------
def main():
    baselines = {}

    for machine_id in MACHINE_IDS:
        baselines[machine_id] = compute_baseline(machine_id)

    # Save to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(baselines, f, indent=4)

    print(f"\n✅ Baselines saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()