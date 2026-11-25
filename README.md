# F1 Monte Carlo Simulation (2026–2027)

A stochastic Monte Carlo simulation engine in Python for modeling Formula 1 race outcomes using empirical telemetry and race data via `fastf1`.

## Overview

The pipeline combines season performance data with historical circuit parameters to run **10,000 Monte Carlo race simulations**:

1. **Data Ingestion (`get_2026_fastf1_data`)**: Pulls completed race and sprint results from `fastf1`, computes driver and constructor standings, and estimates each driver's `Street_Affinity` across street circuits (`Monaco`, `Baku`, `Singapore`).
2. **Strength Modeling (`compute_strength_score`)**: Normalizes constructor strength, driver points, and street-circuit affinity into a composite `Strength_Score_2026_2027` in $[0, 1]$:
   $$\text{Strength\_Score}_{2026\text{–}2027} = w_{\text{car}} \cdot \text{Car\_Norm} + w_{\text{drv}} \cdot \text{Driver\_Norm} + w_{\text{str}} \cdot \text{Street\_Norm}$$
   with default weights $w_{\text{car}} = 0.50$, $w_{\text{drv}} = 0.35$, and $w_{\text{str}} = 0.15$.
3. **Empirical Calibration (`extract_circuit_empirical_params`)**: Analyzes past editions of a target Grand Prix (e.g., Singapore, 2022–2025) to extract:
   - Average neutralizations (`avg_neutralizations`, Safety Car / Virtual Safety Car statuses `4` and `6`)
   - Tyre life vs. lap time correlation (`corr_tyre_lap`)
   - Qualifying grid vs. race finish Spearman correlation (`corr_grid_final`)
   - Pole-sitter win rate (`poleman_win_rate`)
4. **Monte Carlo Engine (`simulate_race_with_empirical_params`)**: Simulates $N$ race realizations incorporating neutralization counts, gap compression, qualifying variance, grid inertia, pole advantage, tyre noise, driver mistakes, and pit-stop variance, then exports win counts and win probabilities.
5. **Visualization (`plot_monte_carlo_results`)**: Generates a horizontal bar chart comparing simulated win probabilities and strength scores and saves `montecarlo_f1_2026_2027.png`.

## Setup

1. Clone the repository and create a virtual environment:
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # macOS / Linux:
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables (if needed) instead of hardcoding credentials in your script, and ensure the cache directory path is portable (e.g., `.cache/fastf1`).

## Usage

Run the simulation script from the project root:

```bash
python f1_montecarlo.py
```

## Notes & Assumptions

- Total simulated race time uses a simplified baseline (`5400.0` seconds) adjusted by relative performance, grid position, neutralizations, tyre variance, mistakes, and pit stops rather than a lap-by-lap physics model.
- Street affinity is clipped to `[0.94, 1.10]` based on the difference between overall average finish position and street-circuit average finish position.