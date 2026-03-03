"""
Golf DFS Lineup Optimizer
=========================
Streamlit app backed by golf_dfs.db — single source of truth for all data.

Workflow:
  Tuesday: Import DK projections → calibration + captain model applied automatically
  Wednesday-Sunday: Build & optimize lineups from DB data
  Monday: Import FanTeam results → updates calibration, retrains captain model
"""

import streamlit as st
import pandas as pd
from pulp import *
from collections import Counter
from io import BytesIO
import os
import xlsxwriter
import numpy as np
from datetime import datetime, time
import sys
import io
import contextlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from db import GolfDFSDatabase
from prepare_weekly import calibrate_ownership
from captain_model import load_training_data, fit_model, predict_captain_own

# --- Configuration ---
DB_PATH = "golf_dfs.db"
MAPPINGS_FILE = "manual_matches.csv"
OPPONENT_DB_FILE = "opponent_database.csv"
RESULTS_FILE = "results_tracker.csv"
KNOWN_ACCOUNTS = [
    "45722304", "ImOnTilt", "Inittobinkit", "mathm05002", "HX30661",
    "AlexTheGrea", "LynxUnited", "ChesterBowles", "drtyrbyr",
    "Surrey_sports", "KTodorov17",
]


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def to_excel(df_lineups, df_summary):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_lineups.to_excel(writer, sheet_name="Lineups", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            df = df_lineups if sheet_name == "Lineups" else df_summary
            for idx, col in enumerate(df.columns):
                series = df[col]
                max_len = max(
                    series.astype(str).map(len).max(), len(str(series.name))
                ) + 1
                worksheet.set_column(idx, idx, max_len)
    return output.getvalue()


def load_manual_mappings():
    if os.path.exists(MAPPINGS_FILE):
        return (
            pd.read_csv(MAPPINGS_FILE)
            .set_index("projection_name")["salary_name"]
            .to_dict()
        )
    return {}


def save_manual_mappings(mappings_dict):
    pd.DataFrame(
        list(mappings_dict.items()), columns=["projection_name", "salary_name"]
    ).to_csv(MAPPINGS_FILE, index=False)


# =========================================================================
# DATABASE LOADING
# =========================================================================

def get_db():
    """Get a fresh database connection (suppresses console print)."""
    with contextlib.redirect_stdout(io.StringIO()):
        db = GolfDFSDatabase(DB_PATH)
    return db


def get_tournament_list():
    """Return tournament summary DataFrame from the database."""
    db = get_db()
    summary = db.get_tournament_summary()
    db.close()
    return summary


def load_players_from_db(tournament_name, year):
    """
    Load player data from DB for a given tournament, apply calibration
    and captain model, return DataFrame ready for the optimizer.

    Returns None if no projection data is available.
    """
    db = get_db()
    df = db.get_ownership_data(tournament_name, year)

    if df.empty:
        db.close()
        return None

    # For optimizer: need dk_projected_pts
    df_proj = df[df["dk_projected_pts"].notna()].copy()

    if df_proj.empty:
        db.close()
        return None

    # Fill missing ft_salary from dk_salary estimate
    mask = df_proj["ft_salary"].isna() & df_proj["dk_salary"].notna()
    df_proj.loc[mask, "ft_salary"] = df_proj.loc[mask, "dk_salary"] / 500

    # Drop players without salary
    df_proj = df_proj[df_proj["ft_salary"].notna()].copy()

    if df_proj.empty:
        db.close()
        return None

    # Apply ownership calibration (DK projected -> FanTeam predicted)
    df_proj["calibrated_ft2_own"] = df_proj.apply(
        lambda r: calibrate_ownership(r["dk_projected_own"], r["ft_salary"], "ft2"),
        axis=1,
    )
    df_proj["calibrated_ft10_own"] = df_proj.apply(
        lambda r: calibrate_ownership(r["dk_projected_own"], r["ft_salary"], "ft10"),
        axis=1,
    )

    # Run captain model predictions
    pred_captain = pd.Series([None] * len(df_proj), index=df_proj.index)
    captain_r2 = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            df_train = load_training_data(db)
        if len(df_train) >= 20:
            coef10, r2_10, feat10 = fit_model(df_train, "ft10_captain_own")
            captain_r2 = r2_10
            df_pred = df_proj.copy()
            df_pred["tournament"] = tournament_name
            pred_captain = pd.Series(
                predict_captain_own(df_pred, coef10, feat10), index=df_proj.index
            )
    except Exception:
        pass

    db.close()

    # Build optimizer-ready DataFrame
    result = pd.DataFrame(
        {
            "Id": range(1, len(df_proj) + 1),
            "display_name": df_proj["player"].values,
            "Salary": df_proj["ft_salary"].values,
            "FPPG": df_proj["dk_projected_pts"].values,
            "Position": "G",
            "projected_ownership": df_proj["calibrated_ft2_own"].values,
            "ft10_predicted_own": df_proj["calibrated_ft10_own"].values,
            "predicted_captain_own": pred_captain.values,
            "std_dev": df_proj["dk_projected_std"].values,
            "dk_salary": df_proj["dk_salary"].values,
            "dk_projected_own_raw": df_proj["dk_projected_own"].values,
            "ft_salary": df_proj["ft_salary"].values,
            "ft2_own_actual": df_proj["ft2_own"].values,
            "ft10_own_actual": df_proj["ft10_own"].values,
            "ft2_captain_own_actual": df_proj["ft2_captain_own"].values,
            "ft10_captain_own_actual": df_proj["ft10_captain_own"].values,
            "total_pts_actual": df_proj["total_pts"].values,
            "finish": df_proj["finish"].values,
        }
    )

    # Std dev estimates
    if result["std_dev"].isna().all():
        result["std_dev"] = result["FPPG"] * 0.30
    else:
        result["std_dev"] = result["std_dev"].fillna(result["FPPG"] * 0.30)

    result["FPPG_std_72"] = result["std_dev"]
    result["FPPG_std_18"] = result["std_dev"] / 2

    result.dropna(subset=["Salary", "FPPG"], inplace=True)
    result = result[result["FPPG"] > 0].copy()

    # Store captain model R2 for display
    if captain_r2 is not None:
        result.attrs["captain_r2"] = captain_r2

    return result


# =========================================================================
# CORE OPTIMIZER (preserved from original)
# =========================================================================

def run_optimizer(
    players_df,
    num_lineups,
    salary_cap,
    min_salary_filter,
    underdog_salary_filter,
    game_id,
    diversity,
    max_early,
    max_late,
    tee_time_cutoff,
    max_exposure,
    disregard_exposure_players,
    ownership_penalty_strength=0,
):
    # --- Constants ---
    NUM_PLAYERS_IN_LINEUP = 6
    UNDERDOG_BONUS_MULTIPLIER = 0.25
    CAPTAIN_POINT_BONUS = 0.25
    MIN_SALARY_THRESHOLD = 98.0
    LOWEST_SALARY_FILTER_PRIMARY = 15.0
    LOWEST_SALARY_FILTER_SECONDARY = 14.5
    DIVERSITY_SCALING_FACTOR = 0.15

    OWNERSHIP_TIERS = {
        "chalk": 0.15,
        "popular": 0.05,
        "moderate": 0.0,
        "low": -0.03,
    }

    CAPTAIN_OWNERSHIP_TIERS = {
        "chalk": 0.40,
        "popular": 0.15,
        "moderate": 0.0,
        "low": -0.10,
    }

    # --- Tee Time Filter ---
    filtered_players_df = players_df.copy()
    if tee_time_cutoff and "tee_time" in players_df.columns:
        try:

            def parse_tee_time(t):
                try:
                    t = str(t).strip().rstrip("*").upper()
                    return pd.to_datetime(t, format="%I:%M %p").time()
                except (ValueError, TypeError):
                    return None

            players_df["parsed_tee_time"] = players_df["tee_time"].apply(
                parse_tee_time
            )
            invalid_tee_times = players_df[players_df["parsed_tee_time"].isna()][
                "display_name"
            ].tolist()
            if invalid_tee_times:
                st.warning(
                    f"Invalid tee times for players: {', '.join(invalid_tee_times)}. These players will be excluded."
                )

            filtered_players_df = players_df[
                players_df["parsed_tee_time"].notna()
                & (players_df["parsed_tee_time"] <= tee_time_cutoff)
            ]
            filtered_players_df = filtered_players_df.drop(
                columns=["parsed_tee_time"], errors="ignore"
            )

            if len(filtered_players_df) < NUM_PLAYERS_IN_LINEUP:
                st.error(
                    f"After applying tee time filter (before {tee_time_cutoff.strftime('%H:%M')}), only {len(filtered_players_df)} players remain. Need at least {NUM_PLAYERS_IN_LINEUP} players."
                )
                return None, None, None
        except Exception as e:
            st.error(
                f"Error processing tee times: {e}. Proceeding without tee time filter."
            )
            filtered_players_df = players_df
    elif tee_time_cutoff:
        st.warning(
            "No 'tee_time' column found in player data. Ignoring tee time filter."
        )

    if filtered_players_df.empty:
        st.error(
            "No players available after filtering. Check your input data or filters."
        )
        return None, None, None

    # --- Data Setup ---
    player_names = dict(
        zip(filtered_players_df["Id"], filtered_players_df["display_name"])
    )
    player_data_map = filtered_players_df.set_index("Id").to_dict("index")

    player_exposures = {pid: 0 for pid in filtered_players_df["Id"]}
    max_exposure_count = max_exposure * num_lineups / 100.0

    disregard_exposure_ids = filtered_players_df[
        filtered_players_df["display_name"].isin(disregard_exposure_players)
    ]["Id"].tolist()

    salaries, points = {}, {}
    for pos in filtered_players_df.Position.unique():
        pos_players = filtered_players_df[filtered_players_df["Position"] == pos]
        salaries[pos] = dict(zip(pos_players.Id, pos_players.Salary * 10))
        points[pos] = dict(zip(pos_players.Id, pos_players.FPPG))

    # --- Ownership Penalty Setup ---
    ownership_penalties = {}
    ownership_col = None
    if "projected_ownership" in filtered_players_df.columns:
        ownership_col = "projected_ownership"
    elif "ownership" in filtered_players_df.columns:
        ownership_col = "ownership"

    has_ownership_data = ownership_col is not None

    if has_ownership_data and ownership_penalty_strength > 0:
        penalty_scale = ownership_penalty_strength / 100.0

        for _, row in filtered_players_df.iterrows():
            pid = row["Id"]
            own_raw = row.get(ownership_col, 0)
            if pd.isna(own_raw):
                own_raw = 0
            own = own_raw * 100 if own_raw <= 1 else own_raw
            fppg = row["FPPG"]

            if own > 20:
                tier_penalty = OWNERSHIP_TIERS["chalk"]
            elif own > 10:
                tier_penalty = OWNERSHIP_TIERS["popular"]
            elif own > 5:
                tier_penalty = OWNERSHIP_TIERS["moderate"]
            else:
                tier_penalty = OWNERSHIP_TIERS["low"]

            ownership_penalties[pid] = tier_penalty * fppg * penalty_scale
    else:
        for pid in filtered_players_df["Id"]:
            ownership_penalties[pid] = 0

    if has_ownership_data and ownership_penalty_strength > 0:
        st.info(
            f"Ownership penalty active (strength: {ownership_penalty_strength}%). Penalizing chalk, boosting low-owned."
        )

    # --- Captain Ownership Penalty Setup ---
    captain_ownership_penalties = {}
    has_captain_data = "predicted_captain_own" in filtered_players_df.columns

    if has_captain_data and ownership_penalty_strength > 0:
        penalty_scale = ownership_penalty_strength / 100.0
        for _, row in filtered_players_df.iterrows():
            pid = row["Id"]
            capt_raw = row.get("predicted_captain_own", 0) or 0
            if pd.isna(capt_raw):
                capt_raw = 0
            capt_own = capt_raw * 100 if capt_raw <= 1 else capt_raw
            fppg = row["FPPG"]

            if capt_own > 15:
                tier_penalty = CAPTAIN_OWNERSHIP_TIERS["chalk"]
            elif capt_own > 8:
                tier_penalty = CAPTAIN_OWNERSHIP_TIERS["popular"]
            elif capt_own > 3:
                tier_penalty = CAPTAIN_OWNERSHIP_TIERS["moderate"]
            else:
                tier_penalty = CAPTAIN_OWNERSHIP_TIERS["low"]

            captain_ownership_penalties[pid] = tier_penalty * fppg * penalty_scale
    else:
        for pid in filtered_players_df["Id"]:
            captain_ownership_penalties[pid] = 0

    # --- Pre-flight check ---
    all_salaries = [
        s for pos_salaries in salaries.values() for s in pos_salaries.values()
    ]
    if len(all_salaries) < NUM_PLAYERS_IN_LINEUP:
        st.error(
            f"Not enough players ({len(all_salaries)}) to form a lineup of {NUM_PLAYERS_IN_LINEUP}."
        )
        return None, None, None

    min_6_player_salary = sum(sorted(all_salaries)[:NUM_PLAYERS_IN_LINEUP])
    if min_6_player_salary > salary_cap:
        st.error("It is impossible to form a lineup under the salary cap.")
        st.error(
            f"The cheapest possible 6-player lineup costs {min_6_player_salary / 10:.2f}M."
        )
        st.error(
            f"Your current salary cap is {salary_cap / 10:.2f}M. Please increase the cap."
        )
        return None, None, None

    # --- Generate Multiple Lineups ---
    generated_lineups = []
    for i in range(num_lineups):
        prob = LpProblem(f"Fantasy_Lineup_{i+1}", LpMaximize)

        player_vars = {
            k: LpVariable.dict(k, v, cat="Binary") for k, v in points.items()
        }
        captain_vars = {
            k: LpVariable.dict(f"is_captain_{k}", v, cat="Binary")
            for k, v in points.items()
        }

        base_points_obj = lpSum(
            points[k][p_id] * player_vars[k][p_id]
            for k in player_vars
            for p_id in player_vars[k]
        )
        captain_bonus_obj = lpSum(
            CAPTAIN_POINT_BONUS * points[k][p_id] * captain_vars[k][p_id]
            for k in captain_vars
            for p_id in captain_vars[k]
        )

        ownership_penalty_obj = lpSum(
            ownership_penalties.get(p_id, 0) * player_vars[k][p_id]
            for k in player_vars
            for p_id in player_vars[k]
        )

        captain_own_penalty_obj = lpSum(
            captain_ownership_penalties.get(p_id, 0) * captain_vars[k][p_id]
            for k in captain_vars
            for p_id in captain_vars[k]
        )
        prob.setObjective(
            base_points_obj
            + captain_bonus_obj
            - ownership_penalty_obj
            - captain_own_penalty_obj
        )

        # Constraints
        total_cost = lpSum(
            salaries[k][p_id] * player_vars[k][p_id]
            for k in player_vars
            for p_id in player_vars[k]
        )
        prob += total_cost <= salary_cap
        prob += (
            lpSum(
                player_vars[k][p_id]
                for k in player_vars
                for p_id in player_vars[k]
            )
            == NUM_PLAYERS_IN_LINEUP
        )
        prob += (
            lpSum(
                captain_vars[k][p_id]
                for k in captain_vars
                for p_id in captain_vars[k]
            )
            == 1
        )

        for k in player_vars:
            for player_id in player_vars[k]:
                prob += captain_vars[k][player_id] <= player_vars[k][player_id]

        # Tee Time Constraints
        if "wave" in filtered_players_df.columns:
            early_ids = filtered_players_df[
                filtered_players_df["wave"] == "Early"
            ]["Id"].tolist()
            late_ids = filtered_players_df[filtered_players_df["wave"] == "Late"][
                "Id"
            ].tolist()
            prob += (
                lpSum(
                    player_vars[player_data_map[p_id]["Position"]][p_id]
                    for p_id in early_ids
                    if p_id in player_data_map
                )
                <= max_early
            )
            prob += (
                lpSum(
                    player_vars[player_data_map[p_id]["Position"]][p_id]
                    for p_id in late_ids
                    if p_id in player_data_map
                )
                <= max_late
            )

        # Minimum Salary Filter
        if min_salary_filter:
            prob += total_cost >= MIN_SALARY_THRESHOLD * 10

        # Underdog Salary Filter
        if underdog_salary_filter:
            valid_underdog_ids = filtered_players_df[
                filtered_players_df["Salary"] >= LOWEST_SALARY_FILTER_PRIMARY
            ]["Id"].tolist()
            if not valid_underdog_ids:
                valid_underdog_ids = filtered_players_df[
                    filtered_players_df["Salary"] >= LOWEST_SALARY_FILTER_SECONDARY
                ]["Id"].tolist()
            for pos in player_vars:
                for pid in player_vars[pos]:
                    if pid not in valid_underdog_ids:
                        prob += player_vars[pos][pid] + captain_vars[pos][pid] <= 1

        # Max Exposure Constraint
        for pid in player_exposures:
            if (
                pid not in disregard_exposure_ids
                and player_exposures[pid] >= max_exposure_count
            ):
                pos = player_data_map[pid]["Position"]
                prob += player_vars[pos][pid] == 0

        # Exclude Previous Lineups
        for prev_lineup in generated_lineups:
            prob += (
                lpSum(
                    player_vars[player_data_map[p_id]["Position"]][p_id]
                    for p_id in prev_lineup
                )
                <= NUM_PLAYERS_IN_LINEUP - 1
            )

        # Apply Diversity
        if diversity > 0:
            shocked_points = {
                pos: {p_id: p for p_id, p in players.items()}
                for pos, players in points.items()
            }
            for pos in shocked_points:
                for p_id in shocked_points[pos]:
                    fppg = points[pos][p_id]
                    if fppg > 0:
                        shock_std_dev = (
                            fppg * (diversity / 100) * DIVERSITY_SCALING_FACTOR
                        )
                        shock = np.random.normal(0, shock_std_dev)
                        shocked_points[pos][p_id] += shock

            shocked_ownership_penalty = lpSum(
                ownership_penalties.get(p_id, 0) * player_vars[k][p_id]
                for k in player_vars
                for p_id in player_vars[k]
            )
            shocked_captain_own_penalty = lpSum(
                captain_ownership_penalties.get(p_id, 0) * captain_vars[k][p_id]
                for k in captain_vars
                for p_id in captain_vars[k]
            )
            prob.setObjective(
                lpSum(
                    shocked_points[k][p_id] * player_vars[k][p_id]
                    for k in player_vars
                    for p_id in player_vars[k]
                )
                + lpSum(
                    CAPTAIN_POINT_BONUS
                    * shocked_points[k][p_id]
                    * captain_vars[k][p_id]
                    for k in captain_vars
                    for p_id in captain_vars[k]
                )
                - shocked_ownership_penalty
                - shocked_captain_own_penalty
            )

        prob.solve(PULP_CBC_CMD(msg=False))
        if LpStatus[prob.status] != "Optimal":
            break

        lineup = []
        captain_id = None
        for k, v in player_vars.items():
            for p_id in v:
                if v[p_id].varValue > 0:
                    lineup.append(p_id)
        for k, v in captain_vars.items():
            for p_id in v:
                if v[p_id].varValue > 0:
                    captain_id = p_id
        generated_lineups.append(lineup)
        player_exposures.update(
            {pid: player_exposures[pid] + 1 for pid in lineup}
        )

    if not generated_lineups:
        st.error(
            "No optimal lineups found. Try relaxing the salary cap, filters, or max exposure."
        )
        return None, None, None

    # --- Post-process ---
    lineup_data = []
    for lineup_ids in generated_lineups:
        lineup_players = filtered_players_df[
            filtered_players_df["Id"].isin(lineup_ids)
        ]

        min_salary_val = lineup_players["Salary"].min()
        underdog_players = lineup_players[
            lineup_players["Salary"] == min_salary_val
        ]
        underdog_player = underdog_players.loc[underdog_players["FPPG"].idxmax()]

        captain_player = lineup_players.loc[lineup_players["FPPG"].idxmax()]

        base_points = lineup_players["FPPG"].sum()
        captain_bonus = captain_player["FPPG"] * CAPTAIN_POINT_BONUS
        underdog_bonus = underdog_player["FPPG"] * UNDERDOG_BONUS_MULTIPLIER

        if captain_player.Id == underdog_player.Id:
            final_score = base_points + captain_bonus
        else:
            final_score = base_points + captain_bonus + underdog_bonus

        total_salary = lineup_players["Salary"].sum()

        lineup_data.append(
            {
                "score": final_score,
                "raw_score": base_points,
                "salary": total_salary,
                "players": lineup_ids,
                "underdog_salary": min_salary_val,
                "captain_id": captain_player.Id,
            }
        )

    # --- Filter and Finalize ---
    filtered_lineups = lineup_data
    if min_salary_filter:
        filtered_lineups = [
            l for l in filtered_lineups if l["salary"] >= MIN_SALARY_THRESHOLD
        ]
    if underdog_salary_filter:
        primary_filtered = [
            l
            for l in filtered_lineups
            if l["underdog_salary"] >= LOWEST_SALARY_FILTER_PRIMARY
        ]
        if len(primary_filtered) < num_lineups and len(primary_filtered) < len(
            filtered_lineups
        ):
            filtered_lineups = [
                l
                for l in filtered_lineups
                if l["underdog_salary"] >= LOWEST_SALARY_FILTER_SECONDARY
            ]
        else:
            filtered_lineups = primary_filtered

    final_lineups = sorted(
        filtered_lineups, key=lambda x: x["score"], reverse=True
    )[:num_lineups]
    if not final_lineups:
        st.error(
            "No lineups remain after applying filters. Try relaxing min salary or underdog salary filters."
        )
        return None, None, None

    # --- Prepare DataFrames for Display ---
    display_data = []
    for lineup in final_lineups:
        lineup_players_df = filtered_players_df[
            filtered_players_df["Id"].isin(lineup["players"])
        ]
        lineup_players_df_sorted = lineup_players_df.sort_values(
            by="Salary", ascending=False
        )
        player_names_list_sorted = lineup_players_df_sorted[
            "display_name"
        ].tolist()

        captain_name = player_names.get(lineup["captain_id"])
        display_data.append(
            {
                "Score": f"{lineup['score']:.2f}",
                "Points": f"{lineup['raw_score']:.2f}",
                "Salary": f"{lineup['salary']:.2f}M",
                "Captain": captain_name,
                "Players": ", ".join(player_names_list_sorted),
            }
        )
    final_lineups_df = pd.DataFrame(display_data)

    total_final_lineups = len(final_lineups)
    player_counts = Counter(
        p_id for lineup in final_lineups for p_id in lineup["players"]
    )
    captain_counts = Counter(lineup["captain_id"] for lineup in final_lineups)
    summary_data = []
    for pid, count in player_counts.items():
        summary_data.append(
            {
                "Player": player_names.get(pid),
                "% In Lineups": count / total_final_lineups,
                "% As Captain": captain_counts.get(pid, 0)
                / total_final_lineups,
                "Exposure (%)": (count / total_final_lineups) * 100,
            }
        )
    summary_df = pd.DataFrame(summary_data).sort_values(
        by="% In Lineups", ascending=False
    )

    csv_data = []
    for lineup in final_lineups:
        row = [int(game_id)] + [int(pid) for pid in lineup["players"]] + [
            int(lineup["captain_id"])
        ]
        csv_data.append(row)
    upload_columns = [
        "game_id",
        "player1",
        "player2",
        "player3",
        "player4",
        "player5",
        "player6",
        "captain_id",
    ]
    upload_df = pd.DataFrame(csv_data, columns=upload_columns)

    return final_lineups_df, summary_df, upload_df


# =========================================================================
# MONTE CARLO SIMULATION (preserved from original)
# =========================================================================

def run_monte_carlo_simulation(
    players_df, assessed_lineups, std_col, prizes, entry_fee, use_cut_model=False
):
    num_simulations = 10000
    st.info(f"Running {num_simulations:,} simulations...")
    progress_bar = st.progress(0)

    if use_cut_model:
        player_projections = (
            players_df.set_index("display_name")[["FPPG", std_col, "make_cut"]]
            .to_dict("index")
        )
    else:
        player_projections = (
            players_df.set_index("display_name")[["FPPG", std_col]].to_dict("index")
        )

    lineup_total_winnings = {lineup["name"]: 0 for lineup in assessed_lineups}
    lineup_wins = {lineup["name"]: 0 for lineup in assessed_lineups}

    for i in range(num_simulations):
        simulated_scores = {}
        for name, data in player_projections.items():
            if use_cut_model:
                made_cut = np.random.rand() < data.get("make_cut", 0)
                if made_cut:
                    simulated_scores[name] = np.random.normal(
                        loc=data["FPPG"], scale=data[std_col]
                    )
                else:
                    simulated_scores[name] = data["FPPG"] / 2
            else:
                simulated_scores[name] = np.random.normal(
                    loc=data["FPPG"], scale=data[std_col]
                )

        simulated_lineup_scores = {}
        for lineup_data in assessed_lineups:
            lineup_name = lineup_data["name"]

            base_score = sum(
                simulated_scores.get(p, 0) for p in lineup_data["players"]
            )
            captain_sim_score = simulated_scores.get(lineup_data["captain"], 0)
            underdog_sim_score = simulated_scores.get(lineup_data["underdog"], 0)

            captain_bonus = captain_sim_score * 0.25
            underdog_bonus = underdog_sim_score * 0.25

            if lineup_data["captain"] == lineup_data["underdog"]:
                total_score = base_score + captain_bonus
            else:
                total_score = base_score + captain_bonus + underdog_bonus

            simulated_lineup_scores[lineup_name] = total_score

        if simulated_lineup_scores:
            sorted_lineups = sorted(
                simulated_lineup_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            rank = 0
            while rank < len(sorted_lineups):
                current_score = sorted_lineups[rank][1]
                tied_lineups = [
                    name
                    for name, score in sorted_lineups
                    if score == current_score
                ]
                num_tied = len(tied_lineups)

                if rank < len(prizes):
                    prize_pool = sum(prizes[rank : rank + num_tied])
                    prize_per_lineup = prize_pool / num_tied
                    for lineup_name in tied_lineups:
                        lineup_total_winnings[lineup_name] += prize_per_lineup

                if rank == 0:
                    for lineup_name in tied_lineups:
                        lineup_wins[lineup_name] += 1 / num_tied

                rank += num_tied

        progress_bar.progress((i + 1) / num_simulations)

    expected_values = {
        name: (total_winnings / num_simulations) - entry_fee
        for name, total_winnings in lineup_total_winnings.items()
    }
    win_probabilities = {
        name: (wins / num_simulations) * 100
        for name, wins in lineup_wins.items()
    }

    return expected_values, win_probabilities


# =========================================================================
# LINEUP ASSESSOR UI (preserved from original)
# =========================================================================

def add_lineup_callback(contest_type):
    session_state_key = f"assessed_lineups_{contest_type}"
    lineup_name = st.session_state[f"name_{contest_type}"]
    selected_players = st.session_state[f"selector_{contest_type}"]
    captain_name = st.session_state.get(f"captain_{contest_type}")
    underdog_player_name = st.session_state.get(f"underdog_{contest_type}")

    if len(selected_players) == 6 and captain_name and underdog_player_name:
        if any(
            lineup["name"] == lineup_name
            for lineup in st.session_state[session_state_key]
        ):
            st.warning(f"A lineup named '{lineup_name}' already exists.")
        else:
            st.session_state[session_state_key].append(
                {
                    "name": lineup_name,
                    "players": selected_players,
                    "captain": captain_name,
                    "underdog": underdog_player_name,
                }
            )
            st.success(f"Lineup '{lineup_name}' added!")
            st.session_state[f"selector_{contest_type}"] = []
    else:
        st.warning(
            "Please select exactly 6 players and ensure roles are assigned."
        )


def build_and_assess_ui(
    players_df, contest_type, std_col, session_state_key, use_cut_model=False
):
    st.info(
        f"Use the controls below to build and add lineups for the {contest_type} contest."
    )

    st.subheader("Build a Lineup")

    lineup_name_options = KNOWN_ACCOUNTS + ["other"]

    st.selectbox(
        "Select Lineup Name:",
        options=lineup_name_options,
        key=f"name_{contest_type}",
    )

    player_list = players_df["display_name"].tolist()

    st.multiselect(
        "Select 6 players for a lineup:",
        options=player_list,
        key=f"selector_{contest_type}",
    )

    if len(st.session_state[f"selector_{contest_type}"]) == 6:
        selected_players = st.session_state[f"selector_{contest_type}"]
        lineup_df = players_df[
            players_df["display_name"].isin(selected_players)
        ]

        most_expensive_player = lineup_df.loc[lineup_df["Salary"].idxmax()]
        default_captain_index = selected_players.index(
            most_expensive_player["display_name"]
        )
        st.selectbox(
            "Select Captain:",
            options=selected_players,
            index=default_captain_index,
            key=f"captain_{contest_type}",
        )

        min_salary = lineup_df["Salary"].min()
        players_at_min_salary = lineup_df[lineup_df["Salary"] == min_salary]

        if len(players_at_min_salary) > 1:
            underdog_options = players_at_min_salary["display_name"].tolist()
            st.selectbox(
                "TIE-BREAKER: Select Underdog Player:",
                options=underdog_options,
                key=f"underdog_{contest_type}",
            )
        elif len(players_at_min_salary) == 1:
            st.session_state[f"underdog_{contest_type}"] = (
                players_at_min_salary["display_name"].iloc[0]
            )

    st.button(
        "Add Lineup",
        key=f"add_{contest_type}",
        on_click=add_lineup_callback,
        args=(contest_type,),
    )

    if st.session_state[session_state_key]:
        st.subheader("Current Lineups for Assessment")

        display_data = []
        for lineup_data in st.session_state[session_state_key]:
            lineup_df = players_df[
                players_df["display_name"].isin(lineup_data["players"])
            ]
            lineup_df_sorted = lineup_df.sort_values(
                by="Salary", ascending=False
            )

            player_list_str = []
            for _, player_row in lineup_df_sorted.iterrows():
                player_name = player_row["display_name"]
                role = ""
                if player_name == lineup_data["captain"]:
                    role = " (c)"
                if player_name == lineup_data["underdog"]:
                    role = f"{role[:-1]}, ud)" if role else " (ud)"
                player_list_str.append(f"{player_name}{role}")

            display_data.append(
                {
                    "Lineup Name": lineup_data["name"],
                    "Players": ", ".join(player_list_str),
                }
            )

        st.dataframe(
            pd.DataFrame(display_data),
            hide_index=True,
            use_container_width=True,
        )

        if st.button("Clear All Lineups", key=f"clear_{contest_type}"):
            st.session_state[session_state_key] = []
            st.rerun()

    st.divider()

    st.subheader("Run Simulation")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "Entry Fee", min_value=0.0, step=1.0, key=f"entry_fee_{contest_type}"
        )
    with col2:
        st.text_input(
            "Prizes (comma-separated)", key=f"prizes_{contest_type}"
        )

    if st.button(
        "Assess All Lineups",
        type="primary",
        key=f"assess_{contest_type}",
    ):
        if not st.session_state[session_state_key]:
            st.warning("Please add at least one lineup to assess.")
            return

        try:
            prizes_str = st.session_state[f"prizes_{contest_type}"]
            prizes = (
                [float(p.strip()) for p in prizes_str.split(",")]
                if prizes_str
                else []
            )
        except ValueError:
            st.error(
                "Invalid prize format. Please enter a comma-separated list of numbers."
            )
            return

        expected_values, win_probabilities = run_monte_carlo_simulation(
            players_df,
            st.session_state[session_state_key],
            std_col,
            prizes,
            st.session_state[f"entry_fee_{contest_type}"],
            use_cut_model,
        )

        lineup_details = []
        for lineup_data in st.session_state[session_state_key]:
            lineup_df = players_df[
                players_df["display_name"].isin(lineup_data["players"])
            ]
            lineup_df_sorted = lineup_df.sort_values(
                by="Salary", ascending=False
            )

            player_list_str = []
            for _, player_row in lineup_df_sorted.iterrows():
                player_name = player_row["display_name"]
                role = ""
                if player_name == lineup_data["captain"]:
                    role = " (c)"
                if player_name == lineup_data["underdog"]:
                    role = f"{role[:-1]}, ud)" if role else " (ud)"
                player_list_str.append(f"{player_name}{role}")

            base_points = lineup_df["FPPG"].sum()
            captain_player = lineup_df[
                lineup_df["display_name"] == lineup_data["captain"]
            ]
            underdog_player = lineup_df[
                lineup_df["display_name"] == lineup_data["underdog"]
            ]
            captain_bonus = captain_player["FPPG"].iloc[0] * 0.25
            underdog_bonus = underdog_player["FPPG"].iloc[0] * 0.25

            if captain_player.index[0] == underdog_player.index[0]:
                projected_score = base_points + captain_bonus
            else:
                projected_score = base_points + captain_bonus + underdog_bonus

            lineup_details.append(
                {
                    "Lineup": lineup_data["name"],
                    "Players": ", ".join(player_list_str),
                    "Projected Score": projected_score,
                    "Expected Value": expected_values.get(
                        lineup_data["name"],
                        -st.session_state[f"entry_fee_{contest_type}"],
                    ),
                    "Win Probability": win_probabilities.get(
                        lineup_data["name"], 0
                    ),
                }
            )

        results_df = pd.DataFrame(lineup_details).sort_values(
            by="Expected Value", ascending=False
        )
        st.session_state.simulation_results = results_df

        def color_ev(val):
            color = "green" if val > 0 else "red" if val < 0 else "white"
            return f"color: {color}"

        st.subheader("Simulation Results")
        st.dataframe(
            results_df.style.applymap(color_ev, subset=["Expected Value"]).format(
                {"Expected Value": "{:.2f}", "Projected Score": "{:.2f}"}
            ),
            use_container_width=True,
            column_config={
                "Win Probability": st.column_config.ProgressColumn(
                    "Win Probability (%)",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100,
                )
            },
            hide_index=True,
        )


# =========================================================================
# OPPONENT DATABASE (preserved from original)
# =========================================================================

def load_opponent_db():
    if os.path.exists(OPPONENT_DB_FILE):
        return pd.read_csv(OPPONENT_DB_FILE)
    return pd.DataFrame(columns=["Date", "Opponent Name", "Expected Value"])


def save_to_opponent_db(results_df):
    if results_df.empty:
        st.warning("No simulation results to save.")
        return

    db_data = results_df[["Lineup", "Expected Value"]].copy()
    db_data.rename(columns={"Lineup": "Opponent Name"}, inplace=True)
    db_data["Date"] = datetime.now().strftime("%Y-%m-%d")

    db_df = load_opponent_db()
    updated_db = pd.concat([db_df, db_data], ignore_index=True)
    updated_db.to_csv(OPPONENT_DB_FILE, index=False)
    st.success("Results saved to opponent database!")


def create_opponent_database_tab():
    st.header("Opponent Database")
    st.info(
        "This table shows the long-term performance of the opponents you have assessed."
    )

    db_df = load_opponent_db()

    if db_df.empty:
        st.warning(
            "No data in the opponent database. Run an assessment and save the results to begin."
        )
    else:
        summary_df = (
            db_df.groupby("Opponent Name")
            .agg(
                Average_EV=("Expected Value", "mean"),
                Entries_Tracked=("Opponent Name", "size"),
            )
            .sort_values(by="Average_EV", ascending=False)
            .reset_index()
        )

        def color_ev(val):
            color = "green" if val > 0 else "red" if val < 0 else "white"
            return f"color: {color}"

        st.dataframe(
            summary_df.style.applymap(color_ev, subset=["Average_EV"]).format(
                {"Average_EV": "{:.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Clear Opponent Database", type="secondary"):
            if os.path.exists(OPPONENT_DB_FILE):
                os.remove(OPPONENT_DB_FILE)
                st.success("Opponent database cleared.")
                st.rerun()


# =========================================================================
# IMPORT TABS
# =========================================================================

def create_monday_import_tab():
    """Monday workflow: import FanTeam results, update calibration, retrain captain model."""
    st.header("Monday Import")
    st.info(
        "Upload FanTeam contest CSVs from the completed tournament. "
        "This imports ownership data, updates calibration, and retrains the captain model."
    )

    col1, col2 = st.columns(2)
    with col1:
        tournament_name = st.text_input(
            "Tournament Name", key="mon_tournament", placeholder="e.g. Genesis Invitational"
        )
    with col2:
        year = st.number_input(
            "Year", value=datetime.now().year, key="mon_year", min_value=2020, max_value=2030
        )

    ft2_file = st.file_uploader("FanTeam 2 CSV", type="csv", key="mon_ft2")
    ft10_file = st.file_uploader("FanTeam 10 CSV", type="csv", key="mon_ft10")

    if st.button("Import FanTeam Results", type="primary", key="mon_import"):
        if not tournament_name:
            st.error("Please enter a tournament name.")
            return
        if not ft2_file or not ft10_file:
            st.error("Please upload both FanTeam 2 and FanTeam 10 CSVs.")
            return

        with st.spinner("Importing FanTeam data..."):
            df_ft2 = pd.read_csv(ft2_file)
            df_ft10 = pd.read_csv(ft10_file)

            # Validate columns
            required_ft_cols = ["Player"]
            for label, df_check in [("FT 2", df_ft2), ("FT 10", df_ft10)]:
                if not all(c in df_check.columns for c in required_ft_cols):
                    st.error(f"{label} CSV missing required columns. Need at least: {required_ft_cols}")
                    return

            db = get_db()

            # Capture import output
            output_buf = io.StringIO()
            with contextlib.redirect_stdout(output_buf):
                t_id = db.import_fanteam(df_ft2, df_ft10, tournament_name, int(year))

            import_log = output_buf.getvalue()
            st.code(import_log, language=None)

            # Show updated calibration analysis
            st.subheader("Updated Calibration Analysis")
            try:
                cal_df = db.calibration_analysis(use_ft_salary=True)
                if not cal_df.empty:
                    st.dataframe(cal_df.round(4), use_container_width=True, hide_index=True)
                else:
                    st.info("Not enough data for calibration analysis yet.")
            except Exception as e:
                st.warning(f"Calibration analysis error: {e}")

            # Retrain captain model and show diagnostics
            st.subheader("Captain Model (retrained)")
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    df_train = load_training_data(db)
                if len(df_train) >= 20:
                    _, r2_ft2, _ = fit_model(df_train, "ft2_captain_own")
                    _, r2_ft10, _ = fit_model(df_train, "ft10_captain_own")
                    st.metric("Training rows", len(df_train))
                    c1, c2 = st.columns(2)
                    c1.metric("FT 2 Captain R-squared", f"{r2_ft2:.3f}")
                    c2.metric("FT 10 Captain R-squared", f"{r2_ft10:.3f}")
                else:
                    st.info(f"Captain model needs more data ({len(df_train)}/20 rows minimum).")
            except Exception as e:
                st.warning(f"Captain model error: {e}")

            db.close()

        st.success(f"FanTeam results imported for {tournament_name} {year}!")

    # --- Auto-populate results tracker ---
    st.divider()
    st.subheader("Auto-Populate Results")
    st.info(
        "If you have saved lineups for this tournament, calculate actual scores "
        "from the database and save to the results tracker."
    )

    calc_tournament = st.text_input(
        "Tournament for score calculation",
        value=st.session_state.get("mon_tournament", ""),
        key="calc_tournament",
    )
    calc_year = st.number_input(
        "Year",
        value=datetime.now().year,
        key="calc_year",
        min_value=2020,
        max_value=2030,
    )

    if calc_tournament:
        db = get_db()
        df_all = db.get_ownership_data(calc_tournament, int(calc_year))
        db.close()

        players_with_pts = df_all[df_all["total_pts"].notna()]
        if not players_with_pts.empty:
            player_options = sorted(players_with_pts["player"].tolist())
            selected_6 = st.multiselect(
                "Select your 6 lineup players:",
                options=player_options,
                key="calc_players",
            )

            if len(selected_6) == 6:
                lineup_df = players_with_pts[
                    players_with_pts["player"].isin(selected_6)
                ].copy()

                # Auto-detect captain (highest projected pts or salary) and underdog (cheapest)
                if lineup_df["ft_salary"].notna().any():
                    captain_name = lineup_df.loc[lineup_df["ft_salary"].idxmax(), "player"]
                    underdog_name = lineup_df.loc[lineup_df["ft_salary"].idxmin(), "player"]
                else:
                    captain_name = selected_6[0]
                    underdog_name = selected_6[-1]

                captain_name = st.selectbox(
                    "Captain (1.25x bonus):", selected_6,
                    index=selected_6.index(captain_name),
                    key="calc_captain",
                )
                underdog_name = st.selectbox(
                    "Underdog (cheapest, 1.25x bonus):", selected_6,
                    index=selected_6.index(underdog_name),
                    key="calc_underdog",
                )

                # Calculate actual score
                base_pts = lineup_df["total_pts"].sum()
                captain_pts = lineup_df.loc[
                    lineup_df["player"] == captain_name, "total_pts"
                ].iloc[0]
                underdog_pts = lineup_df.loc[
                    lineup_df["player"] == underdog_name, "total_pts"
                ].iloc[0]

                captain_bonus = captain_pts * 0.25
                underdog_bonus = underdog_pts * 0.25

                if captain_name == underdog_name:
                    total_score = base_pts + captain_bonus
                else:
                    total_score = base_pts + captain_bonus + underdog_bonus

                st.metric("Calculated Actual Score", f"{total_score:.1f}")

                col_a, col_b = st.columns(2)
                contest = col_a.selectbox("Contest", ["FT 2", "FT 10"], key="calc_contest")
                entry_fee = col_b.number_input("Entry Fee", value=2.0, key="calc_entry")
                prize = st.number_input("Prize Won", min_value=0.0, step=0.5, key="calc_prize")
                finish_pos = st.number_input("Finishing Position", min_value=1, step=1, key="calc_finish")

                if st.button("Save to Results Tracker", key="calc_save"):
                    save_result({
                        "date": str(datetime.now().date()),
                        "tournament": calc_tournament,
                        "contest": contest,
                        "entry_fee": entry_fee,
                        "score": total_score,
                        "finish": finish_pos,
                        "prize": prize,
                        "profit": prize - entry_fee,
                        "captain": captain_name,
                        "notes": f"Auto-calculated from DB. Players: {', '.join(selected_6)}",
                    })
                    st.success("Result saved to tracker!")
        else:
            st.info("No actual results (total_pts) available for this tournament yet.")


def create_tuesday_import_tab():
    """Tuesday workflow: import DK projections, apply calibration, make tournament available."""
    st.header("Tuesday Import")
    st.info(
        "Upload a DataGolf DK projection CSV to make a tournament available for lineup building. "
        "Calibration and captain model predictions are applied automatically when you load the tournament."
    )

    col1, col2 = st.columns(2)
    with col1:
        tournament_name = st.text_input(
            "Tournament Name", key="tue_tournament", placeholder="e.g. Arnold Palmer Invitational"
        )
    with col2:
        year = st.number_input(
            "Year", value=datetime.now().year, key="tue_year", min_value=2020, max_value=2030
        )

    proj_file = st.file_uploader(
        "DataGolf DK Projection CSV", type="csv", key="tue_proj"
    )

    st.caption("Expected columns: dk_name, dk_salary, projected_ownership, total_points, std_dev")

    if st.button("Import Projections", type="primary", key="tue_import"):
        if not tournament_name:
            st.error("Please enter a tournament name.")
            return
        if not proj_file:
            st.error("Please upload a DK projection CSV.")
            return

        with st.spinner("Importing projections..."):
            df_proj = pd.read_csv(proj_file)
            df_proj.columns = df_proj.columns.str.strip()

            required = ["dk_name", "dk_salary", "total_points"]
            missing = [c for c in required if c not in df_proj.columns]
            if missing:
                st.error(f"CSV missing required columns: {missing}")
                return

            db = get_db()

            output_buf = io.StringIO()
            with contextlib.redirect_stdout(output_buf):
                t_id = db.import_dk_projected(df_proj, tournament_name, int(year))

            import_log = output_buf.getvalue()
            st.code(import_log, language=None)
            db.close()

        st.success(
            f"Projections imported for {tournament_name} {year}! "
            "Select it from the sidebar to build lineups."
        )

        # Preview calibrated data
        players_df = load_players_from_db(tournament_name, int(year))
        if players_df is not None:
            st.subheader("Preview: Top 20 Players (Calibrated)")
            preview = players_df.sort_values("FPPG", ascending=False).head(20)
            display_cols = ["display_name", "Salary", "FPPG", "projected_ownership", "predicted_captain_own"]
            available = [c for c in display_cols if c in preview.columns]
            fmt = {}
            if "projected_ownership" in available:
                fmt["projected_ownership"] = "{:.1%}"
            if "predicted_captain_own" in available:
                fmt["predicted_captain_own"] = "{:.1%}"
            if "FPPG" in available:
                fmt["FPPG"] = "{:.1f}"
            if "Salary" in available:
                fmt["Salary"] = "{:.1f}M"
            st.dataframe(
                preview[available].style.format(fmt),
                use_container_width=True,
                hide_index=True,
            )


# =========================================================================
# RESULTS TRACKER
# =========================================================================

def load_results():
    if os.path.exists(RESULTS_FILE):
        return pd.read_csv(RESULTS_FILE)
    return pd.DataFrame(
        columns=[
            "date", "tournament", "contest", "entry_fee",
            "score", "finish", "prize", "profit", "captain", "notes",
        ]
    )


def save_result(entry):
    df = load_results()
    new_row = pd.DataFrame([entry])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(RESULTS_FILE, index=False)


def create_results_tracker_tab():
    st.header("Results Tracker")

    tab_log, tab_history, tab_summary = st.tabs(
        ["Log Result", "History", "P&L Summary"]
    )

    with tab_log:
        st.subheader("Log a Contest Result")
        st.info("Record your result immediately after each contest closes.")

        col1, col2 = st.columns(2)
        with col1:
            tournament = st.text_input(
                "Tournament", placeholder="e.g. Cognizant Classic", key="rt_tournament"
            )
            contest = st.selectbox("Contest", ["FT 2", "FT 10"], key="rt_contest")
            entry_fee = st.number_input(
                "Entry Fee", min_value=0.0, value=2.0, step=1.0, key="rt_entry"
            )
            score = st.number_input(
                "Your Score", min_value=0.0, step=0.5, key="rt_score"
            )
        with col2:
            finish = st.number_input(
                "Finishing Position", min_value=1, step=1, key="rt_finish"
            )
            prize = st.number_input(
                "Prize Won", min_value=0.0, step=0.5, key="rt_prize"
            )
            captain = st.text_input(
                "Captain Used",
                placeholder="e.g. Scottie Scheffler",
                key="rt_captain",
            )
            notes = st.text_input(
                "Notes (optional)",
                placeholder="e.g. high ownership week",
                key="rt_notes",
            )

        date = st.date_input("Date", key="rt_date")

        if st.button("Save Result", type="primary", key="rt_save"):
            if not tournament:
                st.warning("Please enter a tournament name.")
            elif score == 0:
                st.warning("Please enter your score.")
            else:
                save_result(
                    {
                        "date": str(date),
                        "tournament": tournament,
                        "contest": contest,
                        "entry_fee": entry_fee,
                        "score": score,
                        "finish": finish,
                        "prize": prize,
                        "profit": prize - entry_fee,
                        "captain": captain,
                        "notes": notes,
                    }
                )
                st.success(
                    f"Result saved: {tournament} {contest} -- Score {score}, "
                    f"Finish #{finish}, Prize {prize:.2f}"
                )
                st.rerun()

    with tab_history:
        st.subheader("All Results")
        df = load_results()
        if df.empty:
            st.info(
                "No results logged yet. Use the Log Result tab after each contest."
            )
        else:
            df_display = df.copy()
            if "profit" not in df_display.columns:
                df_display["profit"] = df_display["prize"] - df_display["entry_fee"]

            def color_profit(val):
                try:
                    color = (
                        "green" if float(val) > 0 else "red" if float(val) < 0 else ""
                    )
                    return f"color: {color}"
                except Exception:
                    return ""

            st.dataframe(
                df_display.style.applymap(color_profit, subset=["profit"]).format(
                    {
                        "entry_fee": "{:.2f}",
                        "prize": "{:.2f}",
                        "profit": "{:.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            if st.button("Delete Last Entry", key="rt_delete"):
                df = df.iloc[:-1]
                df.to_csv(RESULTS_FILE, index=False)
                st.success("Last entry deleted.")
                st.rerun()

    with tab_summary:
        st.subheader("P&L Summary")
        df = load_results()
        if df.empty:
            st.info("No results logged yet.")
            return

        if "profit" not in df.columns:
            df["profit"] = df["prize"] - df["entry_fee"]

        total_entries = len(df)
        total_staked = df["entry_fee"].sum()
        total_prizes = df["prize"].sum()
        total_profit = df["profit"].sum()
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
        win_rate = (df["prize"] > 0).mean() * 100
        avg_finish = df["finish"].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Entries", total_entries)
        col2.metric("Total Staked", f"{total_staked:.2f}")
        col3.metric("Total Prizes", f"{total_prizes:.2f}")
        col4.metric("Net P&L", f"{total_profit:.2f}", delta=f"{roi:.1f}% ROI")

        st.divider()

        col5, col6 = st.columns(2)
        col5.metric("Win Rate (any prize)", f"{win_rate:.1f}%")
        col6.metric("Avg Finishing Position", f"#{avg_finish:.0f}")

        if df["contest"].nunique() > 1:
            st.subheader("By Contest Type")
            by_contest = (
                df.groupby("contest")
                .agg(
                    entries=("entry_fee", "count"),
                    staked=("entry_fee", "sum"),
                    prizes=("prize", "sum"),
                    profit=("profit", "sum"),
                    avg_finish=("finish", "mean"),
                )
                .reset_index()
            )
            by_contest["roi"] = (
                by_contest["profit"] / by_contest["staked"] * 100
            ).round(1)

            def color_profit_2(val):
                try:
                    color = (
                        "green"
                        if float(val) > 0
                        else "red" if float(val) < 0 else ""
                    )
                    return f"color: {color}"
                except Exception:
                    return ""

            st.dataframe(
                by_contest.style.applymap(
                    color_profit_2, subset=["profit", "roi"]
                ).format(
                    {
                        "staked": "{:.2f}",
                        "prizes": "{:.2f}",
                        "profit": "{:.2f}",
                        "roi": "{:.1f}%",
                        "avg_finish": "#{:.0f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("By Tournament")
        by_tourn = (
            df.groupby("tournament")
            .agg(
                entries=("entry_fee", "count"),
                staked=("entry_fee", "sum"),
                prizes=("prize", "sum"),
                profit=("profit", "sum"),
                best_finish=("finish", "min"),
            )
            .reset_index()
            .sort_values("profit", ascending=False)
        )
        by_tourn["roi"] = (by_tourn["profit"] / by_tourn["staked"] * 100).round(1)

        def color_profit_3(val):
            try:
                color = (
                    "green" if float(val) > 0 else "red" if float(val) < 0 else ""
                )
                return f"color: {color}"
            except Exception:
                return ""

        st.dataframe(
            by_tourn.style.applymap(
                color_profit_3, subset=["profit", "roi"]
            ).format(
                {
                    "staked": "{:.2f}",
                    "prizes": "{:.2f}",
                    "profit": "{:.2f}",
                    "roi": "{:.1f}%",
                    "best_finish": "#{:.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        if df["captain"].notna().any() and (df["captain"] != "").any():
            st.subheader("Captain Performance")
            df_capt = df[df["captain"].notna() & (df["captain"] != "")]
            by_captain = (
                df_capt.groupby("captain")
                .agg(
                    times_used=("captain", "count"),
                    avg_score=("score", "mean"),
                    avg_finish=("finish", "mean"),
                    total_profit=("profit", "sum"),
                )
                .reset_index()
                .sort_values("avg_score", ascending=False)
            )

            st.dataframe(
                by_captain.style.applymap(
                    color_profit_3, subset=["total_profit"]
                ).format(
                    {
                        "avg_score": "{:.1f}",
                        "avg_finish": "#{:.0f}",
                        "total_profit": "{:.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Running P&L")
        df_sorted = df.sort_values("date").copy()
        df_sorted["cumulative_profit"] = df_sorted["profit"].cumsum()
        st.line_chart(df_sorted.set_index("date")["cumulative_profit"])


# =========================================================================
# MAIN APP LAYOUT
# =========================================================================

st.set_page_config(page_title="Golf DFS Optimizer", layout="wide")
st.title("Golf DFS Lineup Optimizer")

# --- Initialize Session State ---
for key, default in {
    "final_lineups_72": None,
    "summary_72": None,
    "upload_72": None,
    "assessed_lineups_72_hole": [],
    "assessed_lineups_showdown": [],
    "entry_fee_72_hole": 50.0,
    "prizes_72_hole": "",
    "entry_fee_showdown": 50.0,
    "prizes_showdown": "",
    "simulation_results": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Sidebar: Tournament Selector ---
st.sidebar.header("Tournament")

tournaments = get_tournament_list()
players_df_72 = None
selected_t_name = None
selected_t_year = None

if tournaments.empty:
    st.sidebar.warning("No tournaments in database. Use the import tabs to begin.")
else:
    tournament_options = []
    for _, row in tournaments.iterrows():
        label = f"{row['name']} ({int(row['year'])})"
        has_proj = row["has_dk_pts"] > 0
        if has_proj:
            label += " [projections]"
        tournament_options.append(label)

    selected_idx = st.sidebar.selectbox(
        "Select Tournament",
        range(len(tournament_options)),
        format_func=lambda i: tournament_options[i],
    )

    selected_row = tournaments.iloc[selected_idx]
    selected_t_name = selected_row["name"]
    selected_t_year = int(selected_row["year"])

    # --- Weekly Checklist ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Weekly Checklist")

    has_proj = selected_row["has_dk_pts"] > 0
    has_ft2 = selected_row["has_ft2"] > 0

    results_df = load_results()
    has_logged = False
    if not results_df.empty and "tournament" in results_df.columns:
        has_logged = len(results_df[results_df["tournament"] == selected_t_name]) > 0

    st.sidebar.markdown(
        f"{'✅' if has_proj else '❌'} Projections imported"
    )
    st.sidebar.markdown(
        f"{'✅' if has_logged else '❌'} Lineups built"
    )
    st.sidebar.markdown(
        f"{'✅' if has_ft2 else '❌'} Results imported"
    )

    # Load player data if projections exist
    if has_proj:
        players_df_72 = load_players_from_db(selected_t_name, selected_t_year)

# --- Sidebar: Optimizer Controls ---
if players_df_72 is not None:
    st.sidebar.markdown("---")
    st.sidebar.header("Optimizer Controls")
    game_id = st.sidebar.number_input("Enter Game ID for Upload", value=977562)
    num_lineups = st.sidebar.slider("Number of Lineups", 1, 100, 25)
    salary_cap = st.sidebar.slider("Salary Cap (M)", 90.0, 110.0, 100.0, 0.1)
    diversity = st.sidebar.slider("Diversity", 0, 100, 0)
    max_exposure = st.sidebar.slider(
        "Max Player Exposure (%)",
        0,
        100,
        100,
        help="Limit the percentage of lineups any single player can appear in.",
    )
    disregard_exposure_players = st.sidebar.multiselect(
        "Disregard Max Exposure for Players",
        options=players_df_72["display_name"].tolist(),
        default=[],
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Tee Time Constraints")
    max_early = st.sidebar.slider("Max Early Wave Players", 0, 6, 6)
    max_late = st.sidebar.slider("Max Late Wave Players", 0, 6, 6)
    tee_time_cutoff = st.sidebar.time_input(
        "All Players Tee Off Before (Optional)",
        value=None,
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Filters")
    min_salary_filter = st.sidebar.checkbox("Apply minimum total salary filter (>= 98.0M)")
    underdog_salary_filter = st.sidebar.checkbox("Apply underdog salary filter (>= 15.0M)")

    st.sidebar.markdown("---")
    st.sidebar.header("Ownership Leverage")
    ownership_penalty_strength = st.sidebar.slider(
        "Ownership Penalty Strength",
        0,
        100,
        0,
        help="Penalize high-owned players and boost low-owned. 0 = off, 100 = full effect.",
    )
    st.sidebar.caption(
        "Tiers: >20% = Chalk (penalized), 10-20% = Popular (slight penalty), "
        "5-10% = Neutral, <5% = Low-owned (boosted)"
    )

# --- Main Tabs ---
tab_gen, tab_assess, tab_monday, tab_tuesday, tab_results, tab_opponents = st.tabs(
    [
        "Lineup Generator",
        "Lineup Assessor",
        "Monday Import",
        "Tuesday Import",
        "Results Tracker",
        "Opponent Database",
    ]
)

# --- Tab 1: Lineup Generator ---
with tab_gen:
    if players_df_72 is not None:
        st.header(f"Lineup Generator: {selected_t_name}")

        # Show player overview
        n_players = len(players_df_72)
        has_own = players_df_72["projected_ownership"].notna().sum()
        has_capt = players_df_72["predicted_captain_own"].notna().sum()
        captain_r2 = players_df_72.attrs.get("captain_r2")

        info_parts = [f"{n_players} players loaded"]
        if has_own > 0:
            info_parts.append(f"calibrated ownership for {has_own}")
        if has_capt > 0:
            r2_str = f" (R²={captain_r2:.3f})" if captain_r2 else ""
            info_parts.append(f"captain predictions for {has_capt}{r2_str}")
        st.info(" | ".join(info_parts))

        # Player data table
        with st.expander("Player Data", expanded=False):
            show_cols = [
                "display_name", "Salary", "FPPG", "std_dev",
                "projected_ownership", "predicted_captain_own",
                "dk_projected_own_raw", "ft2_own_actual", "ft10_own_actual",
            ]
            available = [c for c in show_cols if c in players_df_72.columns]
            fmt = {}
            for c in available:
                if "own" in c.lower():
                    fmt[c] = "{:.1%}"
                elif c in ("FPPG", "std_dev"):
                    fmt[c] = "{:.1f}"
                elif c == "Salary":
                    fmt[c] = "{:.1f}M"
            st.dataframe(
                players_df_72[available]
                .sort_values("FPPG", ascending=False)
                .style.format(fmt),
                use_container_width=True,
                hide_index=True,
                height=400,
            )

        if st.button("Generate Lineups", type="primary"):
            with st.spinner("Optimizing..."):
                scaled_salary_cap = salary_cap * 10
                (
                    st.session_state["final_lineups_72"],
                    st.session_state["summary_72"],
                    st.session_state["upload_72"],
                ) = run_optimizer(
                    players_df_72,
                    num_lineups,
                    scaled_salary_cap,
                    min_salary_filter,
                    underdog_salary_filter,
                    game_id,
                    diversity,
                    max_early,
                    max_late,
                    tee_time_cutoff,
                    max_exposure,
                    disregard_exposure_players,
                    ownership_penalty_strength,
                )

        if st.session_state.get("final_lineups_72") is not None:
            st.success("Optimization Complete!")
            st.header("Generated Lineups")
            st.dataframe(
                st.session_state["final_lineups_72"],
                use_container_width=True,
            )

            st.header("Player Summary")
            st.dataframe(
                st.session_state["summary_72"]
                .style.format(
                    {
                        "% In Lineups": "{:.1%}",
                        "% As Captain": "{:.1%}",
                        "Exposure (%)": "{:.2f}%",
                    }
                ),
                use_container_width=True,
            )

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    label="Download Lineups for Upload (CSV)",
                    data=st.session_state["upload_72"]
                    .to_csv(index=False, header=False)
                    .encode("utf-8"),
                    file_name="upload_lineups.csv",
                    mime="text/csv",
                )
            with dl_col2:
                excel_data = to_excel(
                    st.session_state["final_lineups_72"],
                    st.session_state["summary_72"],
                )
                st.download_button(
                    label="Download Full Summary (Excel)",
                    data=excel_data,
                    file_name="lineup_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
    else:
        st.header("Lineup Generator")
        if selected_t_name:
            st.warning(
                f"No projection data available for {selected_t_name}. "
                "Use the **Tuesday Import** tab to add DK projections."
            )
        else:
            st.info(
                "Welcome! Select a tournament from the sidebar, or use the import tabs "
                "to add tournament data."
            )

# --- Tab 2: Lineup Assessor ---
with tab_assess:
    if players_df_72 is not None:
        st.header(f"Lineup Assessor: {selected_t_name}")
        players_df_18 = players_df_72.copy()

        contest_type = st.radio(
            "Select Contest Type:",
            ("72-Hole Contest", "Single Round Showdown"),
            horizontal=True,
            key="assessor_contest",
        )

        if contest_type == "72-Hole Contest":
            build_and_assess_ui(
                players_df_72,
                "72_hole",
                "FPPG_std_72",
                "assessed_lineups_72_hole",
                False,
            )
        else:
            build_and_assess_ui(
                players_df_18,
                "showdown",
                "FPPG_std_18",
                "assessed_lineups_showdown",
                False,
            )

        if st.session_state.get("simulation_results") is not None:
            if st.button("Save Results to Opponent Database"):
                save_to_opponent_db(st.session_state.simulation_results)
    else:
        st.header("Lineup Assessor")
        st.info("Select a tournament with projections to use the lineup assessor.")

# --- Tab 3: Monday Import ---
with tab_monday:
    create_monday_import_tab()

# --- Tab 4: Tuesday Import ---
with tab_tuesday:
    create_tuesday_import_tab()

# --- Tab 5: Results Tracker ---
with tab_results:
    create_results_tracker_tab()

# --- Tab 6: Opponent Database ---
with tab_opponents:
    create_opponent_database_tab()
