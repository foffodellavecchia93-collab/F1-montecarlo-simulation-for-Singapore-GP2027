import os
import time
import fastf1 as f1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastf1.core import Laps

if os.getenv("F1_USERNAME") and os.getenv("F1_PASSWORD"):
    os.environ["F1_USERNAME"] = os.environ["F1_USERNAME"]
    os.environ["F1_PASSWORD"] = os.environ["F1_PASSWORD"]

CACHE_DIR = os.getenv("FASTF1_CACHE_DIR", ".cache/fastf1")
os.makedirs(CACHE_DIR, exist_ok=True)
f1.Cache.enable_cache(CACHE_DIR)
STREET_CIRCUITS = ("Monaco", "Baku", "Singapore")

TEAM_COLORS = {
    "Mercedes": "#27F4D2",
    "Ferrari": "#E8002D",
    "Red Bull Racing": "#3671C6",
    "McLaren": "#FF8000",
    "Racing Bulls": "#6692FF",
    "Alpine": "#FF87BC",
    "Haas F1 Team": "#B6BABD",
    "Audi": "#F50537",
    "Williams": "#64C4FF",
    "Aston Martin": "#229971",
    "Cadillac": "#1E1E1E",
}


def get_2026_fastf1_data(year: int = 2026):
    schedule = f1.get_event_schedule(year, include_testing=False)
    now = pd.Timestamp.now(tz=None)

    completed = schedule[
        (schedule["RoundNumber"] > 0) & (pd.to_datetime(schedule["EventDate"]) <= now)
    ]

    all_results = []
    street_positions = {}
    overall_positions = {}

    for _, ev in completed.iterrows():
        round_num = int(ev["RoundNumber"])
        event_name = str(ev["EventName"])
        is_street = any(name.lower() in event_name.lower() for name in STREET_CIRCUITS)

        session_types = ["R"]
        if str(ev.get("EventFormat", "")).lower().find("sprint") != -1:
            session_types.append("S")

        for stype in session_types:
            try:
                sess = f1.get_session(year, round_num, stype)
                sess.load(laps=False, telemetry=False, weather=False, messages=False)
                res = sess.results
                if res is None or res.empty:
                    continue

                tmp = res[["Abbreviation", "FullName", "TeamName", "Points", "Position"]].copy()
                tmp["Points"] = pd.to_numeric(tmp["Points"], errors="coerce").fillna(0.0)
                tmp["Position"] = pd.to_numeric(tmp["Position"], errors="coerce")
                all_results.append(tmp)

                if stype == "R":
                    for _, r in tmp.dropna(subset=["Position"]).iterrows():
                        abbr = str(r["Abbreviation"])
                        pos = float(r["Position"])
                        overall_positions.setdefault(abbr, []).append(pos)
                        if is_street:
                            street_positions.setdefault(abbr, []).append(pos)

                time.sleep(1.2)
            except Exception:
                time.sleep(2.5)
                continue

    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    drivers_standings = (
        results_df.groupby(["Abbreviation", "FullName", "TeamName"], as_index=False)["Points"]
        .sum()
        .sort_values("Points", ascending=False)
        .reset_index(drop=True)
    )

    constructors_standings = (
        results_df.groupby("TeamName", as_index=False)["Points"]
        .sum()
        .sort_values("Points", ascending=False)
        .reset_index(drop=True)
    )

    street_coef = {}
    for abbr, s_pos in street_positions.items():
        avg_all = float(np.mean(overall_positions.get(abbr, s_pos)))
        avg_street = float(np.mean(s_pos))
        delta = avg_all - avg_street
        street_coef[abbr] = round(float(np.clip(1.0 + delta * 0.015, 0.94, 1.10)), 4)

    drivers_standings["Street_Affinity"] = (
        drivers_standings["Abbreviation"].map(street_coef).fillna(1.00)
    )

    return drivers_standings, constructors_standings


def compute_strength_score(
    drivers_df: pd.DataFrame,
    constructors_df: pd.DataFrame,
    w_car: float = 0.50,
    w_drv: float = 0.35,
    w_str: float = 0.15,
) -> pd.DataFrame:
    if not np.isclose(w_car + w_drv + w_str, 1.0):
        raise ValueError("Weights w_car, w_drv, and w_str must sum to 1.0.")

    c_df = constructors_df.copy()
    c_df["Points"] = pd.to_numeric(c_df["Points"], errors="coerce").fillna(0.0)
    c_df["Car_Norm"] = c_df["Points"] / max(float(c_df["Points"].max()), 1.0)

    d_df = drivers_df.copy()
    d_df["Points"] = pd.to_numeric(d_df["Points"], errors="coerce").fillna(0.0)
    d_df["Street_Affinity"] = pd.to_numeric(
        d_df.get("Street_Affinity", 1.0), errors="coerce"
    ).fillna(1.0)

    d_df["Driver_Norm"] = d_df["Points"] / max(float(d_df["Points"].max()), 1.0)
    d_df["Street_Norm"] = ((d_df["Street_Affinity"] - 0.94) / (1.10 - 0.94)).clip(0.0, 1.0)

    out = d_df.merge(
        c_df[["TeamName", "Car_Norm"]],
        on="TeamName",
        how="left",
    )
    out["Car_Norm"] = out["Car_Norm"].fillna(0.10)

    out["Strength_Score_2026_2027"] = (
        w_car * out["Car_Norm"]
        + w_drv * out["Driver_Norm"]
        + w_str * out["Street_Norm"]
    ).round(4)

    return (
        out.sort_values(["Points", "Strength_Score_2026_2027"], ascending=False)
        .drop_duplicates(subset=["Abbreviation"], keep="first")
        .sort_values("Strength_Score_2026_2027", ascending=False)
        .reset_index(drop=True)
    )


def extract_circuit_empirical_params(
    gp_name: str = "Singapore",
    years=(2022, 2023, 2024, 2025),
    plot_tyre_deg: bool = True,
) -> dict:
    session_results_list = []
    compound_deg_list = []
    sc_events_list = []

    for year in years:
        session_ = f1.get_session(year, gp_name, "Race")
        session_.load()

        res_sub = session_.results[["Abbreviation", "GridPosition", "Position"]].copy()
        res_sub["GridPosition"] = pd.to_numeric(res_sub["GridPosition"], errors="coerce")
        res_sub["Position"] = pd.to_numeric(res_sub["Position"], errors="coerce")
        session_results_list.append(res_sub)

        track_status = session_.track_status
        sc_events = track_status[track_status["Status"].isin(["4", "6"])]
        sc_events_list.append(sc_events)

        laps_: Laps = session_.laps.pick_accurate().dropna(subset=["LapTime"])
        compounds = laps_["Compound"].dropna().unique()

        for comp in compounds:
            stint_df = laps_[laps_["Compound"] == comp].copy()
            stint_df["LapTime_s"] = stint_df["LapTime"].dt.total_seconds()
            stint_df["TyreLife"] = pd.to_numeric(stint_df["TyreLife"], errors="coerce")
            tyre_deg = (
                stint_df.groupby(["Stint", "TyreLife"], as_index=False)["LapTime_s"]
                .mean()
                .rename(columns={"LapTime_s": "LapTime"})
            )
            compound_deg_list.append(tyre_deg)

            if plot_tyre_deg and not tyre_deg.empty:
                tyre_deg.plot.scatter(
                    x="LapTime",
                    y="TyreLife",
                    color="Blue",
                    s=15,
                )
                plt.xlabel("Average Lap Time (s)", fontsize=11)
                plt.ylabel("Tyre Life (Laps)", fontsize=11)
                plt.title(f"Tyre deg analysis ({comp}, {year})", fontsize=12, fontweight="bold")
                plt.grid(True, linestyle=":", alpha=0.6)
                plt.tight_layout()
                plt.show()

    n_years = max(len(years), 1)
    sc_events_tot = pd.concat(sc_events_list, ignore_index=True) if sc_events_list else pd.DataFrame()
    sc_avg = len(sc_events_tot) / n_years

    df_tyre_history = (
        pd.concat(compound_deg_list, ignore_index=True).dropna(subset=["TyreLife", "LapTime"])
        if compound_deg_list
        else pd.DataFrame()
    )
    corr_tyre = (
        float(df_tyre_history["TyreLife"].corr(df_tyre_history["LapTime"]))
        if len(df_tyre_history) > 1
        else -0.13
    )

    df_history = (
        pd.concat(session_results_list, ignore_index=True).dropna(subset=["GridPosition", "Position"])
        if session_results_list
        else pd.DataFrame()
    )
    corr_grid = (
        float(df_history["GridPosition"].corr(df_history["Position"], method="spearman"))
        if len(df_history) > 1
        else 0.65
    )
    poleman_wins = len(df_history[(df_history["GridPosition"] == 1) & (df_history["Position"] == 1)])
    poleman_win_rate = poleman_wins / n_years

    return {
        "avg_neutralizations": round(sc_avg, 4),
        "corr_tyre_lap": round(corr_tyre, 4),
        "corr_grid_final": round(corr_grid, 4),
        "poleman_win_rate": round(poleman_win_rate, 4),
    }


def simulate_race_with_empirical_params(
    drivers_df: pd.DataFrame,
    params: dict,
    n_sims: int = 10_000,
    is_street_circuit: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_drivers = len(drivers_df)

    strength = pd.to_numeric(drivers_df["Strength_Score_2026_2027"], errors="coerce").fillna(0.10).to_numpy()
    street_aff = pd.to_numeric(drivers_df.get("Street_Affinity", 1.0), errors="coerce").fillna(1.0).to_numpy()

    eff_strength = strength * (street_aff if is_street_circuit else 1.0)

    neut_count = rng.poisson(lam=params["avg_neutralizations"], size=n_sims)
    gap_compression = 1.0 / (1.0 + 0.22 * neut_count[:, None])

    qual_score = -eff_strength[None, :] + rng.normal(0.0, 0.14, size=(n_sims, n_drivers))
    grid_pos = np.argsort(np.argsort(qual_score, axis=1), axis=1) + 1

    grid_inertia = params["corr_grid_final"] / (1.0 + 0.25 * neut_count[:, None])
    grid_penalty = (grid_pos - 1) * (2.2 * grid_inertia)

    is_pole = (grid_pos == 1)
    pole_bonus = is_pole * (3.5 * params["poleman_win_rate"]) * (1.0 / (1.0 + 0.20 * neut_count[:, None]))

    tyre_scale = 1.0 + abs(params["corr_tyre_lap"]) * 0.6
    tyre_noise = rng.normal(0.0, (4.5 + 0.5 * neut_count[:, None]) * tyre_scale, size=(n_sims, n_drivers))

    mistake_prob = np.clip((0.11 if is_street_circuit else 0.09) - 0.03 * (strength - 0.5), 0.04, 0.18)[None, :]
    mistake_loss = (rng.random((n_sims, n_drivers)) < mistake_prob) * rng.uniform(3.0, 25.0, size=(n_sims, n_drivers))

    pit_prob = np.clip(0.08 + 0.02 * (neut_count[:, None] > 0), 0.0, 0.30)
    pit_loss = (rng.random((n_sims, n_drivers)) < pit_prob) * rng.uniform(2.5, 11.0, size=(n_sims, n_drivers))

    base_time = 5400.0 - (eff_strength[None, :] * 42.0 * gap_compression)
    total_time = base_time + grid_penalty - pole_bonus + tyre_noise + mistake_loss + pit_loss

    winners = np.argmin(total_time, axis=1)
    win_counts = np.bincount(winners, minlength=n_drivers)

    out = drivers_df[[
        "Abbreviation", "FullName", "TeamName", "Points", "Street_Affinity", "Strength_Score_2026_2027"
    ]].copy()
    out["Wins_Sim"] = win_counts
    out["Win_Pct"] = np.round((win_counts / n_sims) * 100.0, 2)
    return out.sort_values("Win_Pct", ascending=False).reset_index(drop=True)


def plot_monte_carlo_results(sim_df: pd.DataFrame, top_n: int = 12, n_sims: int = 10_000):
    plot_df = sim_df.head(top_n).iloc[::-1].reset_index(drop=True)
    colors = [TEAM_COLORS.get(t, "#888888") for t in plot_df["TeamName"]]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    fig.patch.set_facecolor("#111317")
    for ax in (ax1, ax2):
        ax.set_facecolor("#181B20")
        ax.tick_params(colors="#E5E7EB", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#2E3440")

    bars = ax1.barh(
        plot_df["Abbreviation"],
        plot_df["Win_Pct"],
        color=colors,
        height=0.68,
        edgecolor="#111317",
    )
    ax1.set_xlabel("Win Probability (%)", color="#E5E7EB", fontsize=11)
    ax1.set_title(
        f"Monte Carlo Simulation ({n_sims:,} Races) – Win Rate",
        color="#FFFFFF",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax1.set_xlim(0, max(float(plot_df["Win_Pct"].max()) * 1.18, 5.0))
    ax1.grid(axis="x", linestyle="--", alpha=0.25, color="#9CA3AF")

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        w = bar.get_width()
        ax1.text(
            w + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{row['Win_Pct']:.2f}% ({int(row['Wins_Sim'])})",
            va="center",
            ha="left",
            color="#F3F4F6",
            fontsize=9.5,
            fontweight="bold",
        )

    y_pos = np.arange(len(plot_df))
    ax2.barh(
        y_pos,
        plot_df["Strength_Score_2026_2027"],
        color="#60A5FA",
        height=0.55,
        alpha=0.88,
        edgecolor="#111317",
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(plot_df["Abbreviation"])
    ax2.set_xlim(0, 1.08)
    ax2.set_xlabel("Strength Score [0 – 1]", color="#E5E7EB", fontsize=11)
    ax2.set_title("Driver / Car Strength Score", color="#FFFFFF", fontsize=12, fontweight="bold", pad=12)
    ax2.grid(axis="x", linestyle="--", alpha=0.25, color="#9CA3AF")

    plt.tight_layout()
    plt.savefig("montecarlo_f1_2026_2027.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    final_sim = None

    drv_df, con_df = get_2026_fastf1_data(2026)
    print("--- 2026 CONSTRUCTORS (from FastF1) ---")
    print(con_df.to_string(index=False))
    print("\n--- 2026 DRIVERS (from FastF1) ---")
    print(drv_df.to_string(index=False))

    scored_df = compute_strength_score(drv_df, con_df)
    print("\n--- 2026-2027 STRENGTH SCORE ---")
    print(
        scored_df[["Abbreviation", "FullName", "TeamName", "Points", "Street_Affinity", "Strength_Score_2026_2027"]]
        .to_string(index=False)
    )

    empirical_params = extract_circuit_empirical_params(
        gp_name="Singapore",
        years=(2022, 2023, 2024, 2025),
        plot_tyre_deg=True,
    )
    print("\nSingapore empirical parameters:", empirical_params)

    final_sim = simulate_race_with_empirical_params(
        drivers_df=scored_df,
        params=empirical_params,
        n_sims=10_000,
        is_street_circuit=True,
        seed=42,
    )
    print("\n--- MONTE CARLO SIMULATION (10,000 RACES) ---")
    print(final_sim.to_string(index=False))

    if final_sim is not None and not final_sim.empty:
        plot_monte_carlo_results(final_sim, top_n=12, n_sims=10_000)