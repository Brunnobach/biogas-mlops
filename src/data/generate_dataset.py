import pandas as pd
import numpy as np
from pathlib import Path


def generate_biogas_data(
    n_samples: int = 5000,
    start_date: str = "2022-01-01",
    freq: str = "D",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic dataset for biogas production forecasting.

    Features mimic real operational parameters of a biodigester:
    - temperature (°C)
    - ph
    - organic_load (kg VS / m³ / day)
    - hydraulic_retention_time (days)
    - substrate_type (categorical: cattle_swine_poultry, energy_crops, food_waste, agricultural_residue)
    - humidity (%)
    - ambient_temperature (°C)
    - previous_day_production (m³) - autoregressive feature

    Target:
    - biogas_production (m³ / day)
    """
    rng = np.random.RandomState(random_state)

    dates = pd.date_range(start=start_date, periods=n_samples, freq=freq)

    substrate_types = [
        "cattle_swine_poultry",
        "energy_crops",
        "food_waste",
        "agricultural_residue",
    ]

    data = {
        "date": dates,
        "temperature": rng.normal(37, 2.5, n_samples).clip(30, 55),
        "ph": rng.normal(7.2, 0.3, n_samples).clip(6.0, 8.5),
        "organic_load": rng.normal(3.5, 1.0, n_samples).clip(0.5, 8.0),
        "hydraulic_retention_time": rng.normal(25, 5, n_samples).clip(10, 60),
        "substrate_type": rng.choice(substrate_types, n_samples),
        "humidity": rng.normal(85, 5, n_samples).clip(60, 98),
        "ambient_temperature": rng.normal(22, 6, n_samples).clip(5, 40),
    }

    df = pd.DataFrame(data)

    # Base biogas production formula (simplified but physically inspired)
    substrate_factor = df["substrate_type"].map(
        {
            "cattle_swine_poultry": 1.0,
            "energy_crops": 1.25,
            "food_waste": 1.15,
            "agricultural_residue": 0.85,
        }
    )

    # Temperature efficiency: mesophilic optimum around 35-37°C, drop outside
    temp_efficiency = 1.0 - 0.02 * np.abs(df["temperature"] - 37)

    # pH efficiency: optimum around 7.0-7.5
    ph_efficiency = 1.0 - 0.4 * np.abs(df["ph"] - 7.25)

    # Organic load drives production volume
    base_production = (
        50
        * substrate_factor
        * df["organic_load"]
        * (df["hydraulic_retention_time"] / 25)
        * temp_efficiency
        * ph_efficiency
    )

    # Add autoregressive component (today depends on yesterday)
    production = np.zeros(n_samples)
    noise = rng.normal(0, 8, n_samples)
    for i in range(n_samples):
        prev = production[i - 1] if i > 0 else base_production.iloc[i]
        ar_component = 0.15 * prev
        production[i] = (
            0.85 * base_production.iloc[i] + ar_component + noise[i]
        ).clip(20, 500)

    df["biogas_production"] = production

    # Add lag feature
    df["previous_day_production"] = df["biogas_production"].shift(1).fillna(df["biogas_production"].median())

    return df


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    """Save dataset to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df = generate_biogas_data()
    save_dataset(df, Path("data/biogas_production.csv"))
    print(f"Dataset generated with {len(df)} rows")
    print(df.head())
    print(f"Target mean: {df['biogas_production'].mean():.2f} m³/day")
    print(f"Target std: {df['biogas_production'].std():.2f} m³/day")
