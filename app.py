import streamlit as st
import pandas as pd
from pulp import *
from collections import Counter
from io import BytesIO
import os
import xlsxwriter
import numpy as np
from datetime import datetime, time

# --- Configuration ---
MAPPINGS_FILE = 'manual_matches.csv'
OPPONENT_DB_FILE = 'opponent_database.csv'
DISCARDED_PLAYERS_FILE = 'discarded_players.csv'

# --- Helper Functions ---
def to_excel(df_lineups, df_summary):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_lineups.to_excel(writer, sheet_name='Lineups', index=False)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            df = df_lineups if sheet_name == 'Lineups' else df_summary
            for idx, col in enumerate(df.columns):
                series = df[col]
                max_len = max((series.astype(str).map(len).max(), len(str(series.name)))) + 1
                worksheet.set_column(idx, idx, max_len)
    processed_data = output.getvalue()
    return processed_data

def load_manual_mappings():
    if os.path.exists(MAPPINGS_FILE):
        return pd.read_csv(MAPPINGS_FILE).set_index('projection_name')['salary_name'].to_dict()
    return {}

def save_manual_mappings(mappings_dict):
    pd.DataFrame(list(mappings_dict.items()), columns=['projection_name', 'salary_name']).to_csv(MAPPINGS_FILE, index=False)

def load_discarded_players():
    if os.path.exists(DISCARDED_PLAYERS_FILE):
        return pd.read_csv(DISCARDED_PLAYERS_FILE)['player_name'].tolist()
    return []

def save_discarded_players(player_list):
    pd.DataFrame(player_list, columns=['player_name']).to_csv(DISCARDED_PLAYERS_FILE, index=False)

def process_player_df(merged_df, projection_col_name):
    required_base_cols = ['PlayerID', 'Price', 'dk_name', 'Position']
    required_cols = required_base_cols + [projection_col_name]
    
    if not all(col in merged_df.columns for col in required_cols):
        st.error(f"A projection file is missing one or more required columns. It needs: {', '.join(required_cols)}")
        return None
    
    players_df = merged_df.copy()
    
    try:
        if 'Price' in players_df.columns:
            players_df['Price'] = players_df['Price'].astype(str).str.replace(r'[£,]', '', regex=True)
            players_df['Price'] = pd.to_numeric(players_df['Price'], errors='coerce')
        if 'PlayerID' in players_df.columns:
            players_df['PlayerID'] = pd.to_numeric(players_df['PlayerID'], errors='coerce')
        if projection_col_name in players_df.columns:
            players_df[projection_col_name] = pd.to_numeric(players_df[projection_col_name], errors='coerce')
    except Exception as e:
        st.error(f"Failed to convert data to numeric types. Please check your files for non-numeric characters. Error: {e}")
        return None
    
    rename_map = {
        'PlayerID': 'Id',
        projection_col_name: 'FPPG',
        'Price': 'Salary',
        'dk_name': 'display_name'
    }
    players_df = players_df.rename(columns=rename_map)

    final_required_cols = ['Id', 'FPPG', 'Salary', 'Position', 'display_name']
    
    initial_rows = len(players_df)
    players_df.dropna(subset=final_required_cols, inplace=True)
    final_rows = len(players_df)

    if final_rows == 0 and initial_rows > 0:
        st.error(f"Data processing failed. After cleaning, 0 players remained. This is often caused by non-numeric data in the '{projection_col_name}', 'Price', or 'PlayerID' columns of your uploaded files.")
        return None

    players_df['Id'] = players_df['Id'].astype(int)
    return players_df

# --- Core Optimizer Logic ---
def run_optimizer(players_df, num_lineups, salary_cap, min_salary_filter, underdog_salary_filter, game_id, diversity, max_early, max_late, tee_time_cutoff, max_exposure, disregard_exposure_players):
    """
    Runs a robust optimizer to generate the top N lineups, ensuring all players tee off before the specified time
    and respecting max exposure constraints, except for disregarded players.
    """
    # --- Constants ---
    NUM_PLAYERS_IN_LINEUP = 6
    UNDERDOG_BONUS_MULTIPLIER = 0.25
    CAPTAIN_POINT_BONUS = 0.25
    MIN_SALARY_THRESHOLD = 98.0
    LOWEST_SALARY_FILTER_PRIMARY = 15.0
    LOWEST_SALARY_FILTER_SECONDARY = 14.5
    DIVERSITY_SCALING_FACTOR = 0.15

    # --- Tee Time Filter ---
    filtered_players_df = players_df.copy()
    if tee_time_cutoff and 'tee_time' in players_df.columns:
        try:
            def parse_tee_time(t):
                try:
                    t = str(t).strip().rstrip('*').upper()
                    return pd.to_datetime(t, format='%I:%M %p').time()
                except (ValueError, TypeError):
                    return None

            players_df['parsed_tee_time'] = players_df['tee_time'].apply(parse_tee_time)
            invalid_tee_times = players_df[players_df['parsed_tee_time'].isna()]['display_name'].tolist()
            if invalid_tee_times:
                st.warning(f"Invalid tee times for players: {', '.join(invalid_tee_times)}. These players will be excluded.")

            filtered_players_df = players_df[players_df['parsed_tee_time'].notna() & (players_df['parsed_tee_time'] <= tee_time_cutoff)]
            filtered_players_df = filtered_players_df.drop(columns=['parsed_tee_time'], errors='ignore')

            if len(filtered_players_df) < NUM_PLAYERS_IN_LINEUP:
                st.error(f"After applying tee time filter (before {tee_time_cutoff.strftime('%H:%M')}), only {len(filtered_players_df)} players remain. Need at least {NUM_PLAYERS_IN_LINEUP} players.")
                return None, None, None
        except Exception as e:
            st.error(f"Error processing tee times: {e}. Proceeding without tee time filter.")
            filtered_players_df = players_df
    elif tee_time_cutoff:
        st.warning("No 'tee_time' column found in player data. Ignoring tee time filter.")

    if filtered_players_df.empty:
        st.error("No players available after filtering. Check your input data or filters.")
        return None, None, None

    # --- Data Setup ---
    player_names = dict(zip(filtered_players_df['Id'], filtered_players_df['display_name']))
    player_data_map = filtered_players_df.set_index('Id').to_dict('index')

    # Initialize player exposure counter
    player_exposures = {pid: 0 for pid in filtered_players_df['Id']}
    max_exposure_count = max_exposure * num_lineups / 100.0  # Convert percentage to count

    # Convert disregarded player names to IDs
    disregard_exposure_ids = filtered_players_df[filtered_players_df['display_name'].isin(disregard_exposure_players)]['Id'].tolist()

    salaries, points = {}, {}
    for pos in filtered_players_df.Position.unique():
        pos_players = filtered_players_df[filtered_players_df["Position"] == pos]
        salaries[pos] = dict(zip(pos_players.Id, pos_players.Salary * 10))
        points[pos] = dict(zip(pos_players.Id, pos_players.FPPG))

    # --- Pre-flight check for feasibility ---
    all_salaries = [s for pos_salaries in salaries.values() for s in pos_salaries.values()]
    if len(all_salaries) < NUM_PLAYERS_IN_LINEUP:
        st.error(f"Not enough players ({len(all_salaries)}) to form a lineup of {NUM_PLAYERS_IN_LINEUP}.")
        return None, None, None

    min_6_player_salary = sum(sorted(all_salaries)[:NUM_PLAYERS_IN_LINEUP])
    if min_6_player_salary > salary_cap:
        st.error(f"It is impossible to form a lineup under the salary cap.")
        st.error(f"The cheapest possible 6-player lineup costs £{min_6_player_salary / 10:.2f}.")
        st.error(f"Your current salary cap is £{salary_cap / 10:.2f}. Please increase the cap.")
        return None, None, None

    # --- Generate Multiple Lineups ---
    generated_lineups = []
    for i in range(num_lineups):
        prob = LpProblem(f"Fantasy_Lineup_{i+1}", LpMaximize)

        player_vars = {k: LpVariable.dict(k, v, cat='Binary') for k, v in points.items()}
        captain_vars = {k: LpVariable.dict(f"is_captain_{k}", v, cat='Binary') for k, v in points.items()}

        # Objective
        base_points_obj = lpSum(points[k][p_id] * player_vars[k][p_id] for k in player_vars for p_id in player_vars[k])
        captain_bonus_obj = lpSum(CAPTAIN_POINT_BONUS * points[k][p_id] * captain_vars[k][p_id] for k in captain_vars for p_id in captain_vars[k])
        prob.setObjective(base_points_obj + captain_bonus_obj)

        # Constraints
        total_cost = lpSum(salaries[k][p_id] * player_vars[k][p_id] for k in player_vars for p_id in player_vars[k])
        prob += total_cost <= salary_cap
        prob += lpSum(player_vars[k][p_id] for k in player_vars for p_id in player_vars[k]) == NUM_PLAYERS_IN_LINEUP
        prob += lpSum(captain_vars[k][p_id] for k in captain_vars for p_id in captain_vars[k]) == 1

        for k in player_vars:
            for player_id in player_vars[k]:
                prob += captain_vars[k][player_id] <= player_vars[k][player_id]

        # Tee Time Constraints
        if 'wave' in filtered_players_df.columns:
            early_ids = filtered_players_df[filtered_players_df['wave'] == 'Early']['Id'].tolist()
            late_ids = filtered_players_df[filtered_players_df['wave'] == 'Late']['Id'].tolist()
            prob += lpSum(player_vars[player_data_map[p_id]['Position']][p_id] for p_id in early_ids if p_id in player_data_map) <= max_early
            prob += lpSum(player_vars[player_data_map[p_id]['Position']][p_id] for p_id in late_ids if p_id in player_data_map) <= max_late

        # Minimum Salary Filter
        if min_salary_filter:
            prob += total_cost >= MIN_SALARY_THRESHOLD * 10

        # Underdog Salary Filter
        if underdog_salary_filter:
            valid_underdog_ids = filtered_players_df[
                filtered_players_df['Salary'] >= LOWEST_SALARY_FILTER_PRIMARY
            ]['Id'].tolist()
            if not valid_underdog_ids:
                valid_underdog_ids = filtered_players_df[
                    filtered_players_df['Salary'] >= LOWEST_SALARY_FILTER_SECONDARY
                ]['Id'].tolist()
            for pos in player_vars:
                for pid in player_vars[pos]:
                    if pid not in valid_underdog_ids:
                        prob += player_vars[pos][pid] + captain_vars[pos][pid] <= 1

        # Max Exposure Constraint (skip for disregarded players)
        for pid in player_exposures:
            if pid not in disregard_exposure_ids and player_exposures[pid] >= max_exposure_count:
                pos = player_data_map[pid]['Position']
                prob += player_vars[pos][pid] == 0

        # Exclude Previous Lineups
        for prev_lineup in generated_lineups:
            prob += lpSum(player_vars[player_data_map[p_id]['Position']][p_id] for p_id in prev_lineup) <= NUM_PLAYERS_IN_LINEUP - 1

        # Apply Diversity
        if diversity > 0:
            shocked_points = {pos: {p_id: p for p_id, p in players.items()} for pos, players in points.items()}
            for pos in shocked_points:
                for p_id in shocked_points[pos]:
                    fppg = points[pos][p_id]
                    if fppg > 0:
                        shock_std_dev = fppg * (diversity / 100) * DIVERSITY_SCALING_FACTOR
                        shock = np.random.normal(0, shock_std_dev)
                        shocked_points[pos][p_id] += shock
            prob.setObjective(
                lpSum(shocked_points[k][p_id] * player_vars[k][p_id] for k in player_vars for p_id in player_vars[k]) +
                lpSum(CAPTAIN_POINT_BONUS * shocked_points[k][p_id] * captain_vars[k][p_id] for k in captain_vars for p_id in captain_vars[k])
            )

        prob.solve(PULP_CBC_CMD(msg=False))
        if LpStatus[prob.status] != 'Optimal':
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
        player_exposures.update({pid: player_exposures[pid] + 1 for pid in lineup})

    if not generated_lineups:
        st.error("No optimal lineups found. Try relaxing the salary cap, filters, or max exposure.")
        return None, None, None

    # --- Post-process to calculate final scores and apply filters ---
    lineup_data = []
    for lineup_ids in generated_lineups:
        lineup_players = filtered_players_df[filtered_players_df['Id'].isin(lineup_ids)]
        
        min_salary_val = lineup_players['Salary'].min()
        underdog_players = lineup_players[lineup_players['Salary'] == min_salary_val]
        underdog_player = underdog_players.loc[underdog_players['FPPG'].idxmax()]
        
        captain_player = lineup_players.loc[lineup_players['FPPG'].idxmax()]
        
        base_points = lineup_players['FPPG'].sum()
        captain_bonus = captain_player['FPPG'] * CAPTAIN_POINT_BONUS
        underdog_bonus = underdog_player['FPPG'] * UNDERDOG_BONUS_MULTIPLIER

        if captain_player.Id == underdog_player.Id:
            final_score = base_points + captain_bonus
        else:
            final_score = base_points + captain_bonus + underdog_bonus
        
        total_salary = lineup_players['Salary'].sum()
        
        lineup_data.append({
            'score': final_score, 'raw_score': base_points, 'salary': total_salary,
            'players': lineup_ids, 'underdog_salary': min_salary_val,
            'captain_id': captain_player.Id
        })

    # --- Filter and Finalize ---
    filtered_lineups = lineup_data
    if min_salary_filter:
        filtered_lineups = [l for l in filtered_lineups if l['salary'] >= MIN_SALARY_THRESHOLD]
    if underdog_salary_filter:
        primary_filtered = [l for l in filtered_lineups if l['underdog_salary'] >= LOWEST_SALARY_FILTER_PRIMARY]
        if len(primary_filtered) < num_lineups and len(primary_filtered) < len(filtered_lineups):
            filtered_lineups = [l for l in filtered_lineups if l['underdog_salary'] >= LOWEST_SALARY_FILTER_SECONDARY]
        else:
            filtered_lineups = primary_filtered
    
    final_lineups = sorted(filtered_lineups, key=lambda x: x['score'], reverse=True)[:num_lineups]
    if not final_lineups:
        st.error("No lineups remain after applying filters. Try relaxing min salary or underdog salary filters.")
        return None, None, None

    # --- Prepare DataFrames for Display ---
    display_data = []
    for lineup in final_lineups:
        lineup_players_df = filtered_players_df[filtered_players_df['Id'].isin(lineup['players'])]
        lineup_players_df_sorted = lineup_players_df.sort_values(by='Salary', ascending=False)
        player_names_list_sorted = lineup_players_df_sorted['display_name'].tolist()
        
        captain_name = player_names.get(lineup['captain_id'])
        display_data.append({
            "Score": f"{lineup['score']:.2f}",
            "Points": f"{lineup['raw_score']:.2f}",
            "Salary": f"£{lineup['salary']:.2f}",
            "Captain": captain_name,
            "Players": ", ".join(player_names_list_sorted)
        })
    final_lineups_df = pd.DataFrame(display_data)

    total_final_lineups = len(final_lineups)
    player_counts = Counter(p_id for lineup in final_lineups for p_id in lineup['players'])
    captain_counts = Counter(lineup['captain_id'] for lineup in final_lineups)
    summary_data = []
    for pid, count in player_counts.items():
        summary_data.append({
            "Player": player_names.get(pid),
            "% In Lineups": count / total_final_lineups,
            "% As Captain": captain_counts.get(pid, 0) / total_final_lineups,
            "Exposure (%)": (count / total_final_lineups) * 100
        })
    summary_df = pd.DataFrame(summary_data).sort_values(by="% In Lineups", ascending=False)
    
    csv_data = []
    for lineup in final_lineups:
        row = [int(game_id)] + [int(pid) for pid in lineup['players']] + [int(lineup['captain_id'])]
        csv_data.append(row)
    upload_columns = ['game_id', 'player1', 'player2', 'player3', 'player4', 'player5', 'player6', 'captain_id']
    upload_df = pd.DataFrame(csv_data, columns=upload_columns)
    
    return final_lineups_df, summary_df, upload_df

# --- Monte Carlo Simulation Function ---
def run_monte_carlo_simulation(players_df, assessed_lineups, std_col, prizes, entry_fee, use_cut_model=False):
    num_simulations = 10000
    st.info(f"Running {num_simulations:,} simulations...")
    progress_bar = st.progress(0)
    
    player_projections = players_df.set_index('display_name')[['FPPG', std_col, 'make_cut']].to_dict('index') if use_cut_model else players_df.set_index('display_name')[['FPPG', std_col]].to_dict('index')
    
    lineup_total_winnings = {lineup['name']: 0 for lineup in assessed_lineups}
    lineup_wins = {lineup['name']: 0 for lineup in assessed_lineups}

    for i in range(num_simulations):
        simulated_scores = {}
        for name, data in player_projections.items():
            if use_cut_model:
                made_cut = np.random.rand() < data.get('make_cut', 0)
                if made_cut:
                    simulated_scores[name] = np.random.normal(loc=data['FPPG'], scale=data[std_col])
                else:
                    simulated_scores[name] = data['FPPG'] / 2 
            else:
                simulated_scores[name] = np.random.normal(loc=data['FPPG'], scale=data[std_col])

        simulated_lineup_scores = {}
        for lineup_data in assessed_lineups:
            lineup_name = lineup_data['name']
            
            base_score = sum(simulated_scores.get(p, 0) for p in lineup_data['players'])
            captain_sim_score = simulated_scores.get(lineup_data['captain'], 0)
            underdog_sim_score = simulated_scores.get(lineup_data['underdog'], 0)
            
            captain_bonus = captain_sim_score * 0.25
            underdog_bonus = underdog_sim_score * 0.25

            if lineup_data['captain'] == lineup_data['underdog']:
                total_score = base_score + captain_bonus
            else:
                total_score = base_score + captain_bonus + underdog_bonus
            
            simulated_lineup_scores[lineup_name] = total_score

        if simulated_lineup_scores:
            sorted_lineups = sorted(simulated_lineup_scores.items(), key=lambda item: item[1], reverse=True)
            
            rank = 0
            while rank < len(sorted_lineups):
                current_score = sorted_lineups[rank][1]
                tied_lineups = [name for name, score in sorted_lineups if score == current_score]
                num_tied = len(tied_lineups)
                
                if rank < len(prizes):
                    prize_pool = sum(prizes[rank : rank + num_tied])
                    prize_per_lineup = prize_pool / num_tied
                    for lineup_name in tied_lineups:
                        lineup_total_winnings[lineup_name] += prize_per_lineup
                
                if rank == 0:
                    for lineup_name in tied_lineups:
                        lineup_wins[lineup_name] += (1 / num_tied)

                rank += num_tied

        progress_bar.progress((i + 1) / num_simulations)

    expected_values = {name: (total_winnings / num_simulations) - entry_fee for name, total_winnings in lineup_total_winnings.items()}
    win_probabilities = {name: (wins / num_simulations) * 100 for name, wins in lineup_wins.items()}
    
    return expected_values, win_probabilities

# --- Callback function to add a lineup ---
def add_lineup_callback(contest_type):
    session_state_key = f"assessed_lineups_{contest_type}"
    lineup_name = st.session_state[f"name_{contest_type}"]
    selected_players = st.session_state[f"selector_{contest_type}"]
    captain_name = st.session_state.get(f"captain_{contest_type}")
    underdog_player_name = st.session_state.get(f"underdog_{contest_type}")

    if len(selected_players) == 6 and captain_name and underdog_player_name:
        if any(lineup['name'] == lineup_name for lineup in st.session_state[session_state_key]):
            st.warning(f"A lineup named '{lineup_name}' already exists.")
        else:
            st.session_state[session_state_key].append({
                'name': lineup_name,
                'players': selected_players,
                'captain': captain_name,
                'underdog': underdog_player_name
            })
            st.success(f"Lineup '{lineup_name}' added!")
            st.session_state[f"selector_{contest_type}"] = []
    else:
        st.warning("Please select exactly 6 players and ensure roles are assigned.")

# --- Generic Assessor UI Function ---
def build_and_assess_ui(players_df, contest_type, std_col, session_state_key, use_cut_model=False):
    st.info(f"Use the controls below to build and add lineups for the {contest_type} contest.")

    st.subheader("Build a Lineup")
    
    lineup_name_options = [
        "45722304", "ImOnTilt", "Inittobinkit", "mathm05002", "HX30661", 
        "AlexTheGrea", "LynxUnited", "ChesterBowles", "drtyrbyr", 
        "Surrey_sports", "KTodorov17", "other"
    ]
    
    st.selectbox("Select Lineup Name:", options=lineup_name_options, key=f"name_{contest_type}")
    
    player_list = players_df['display_name'].tolist()
    
    st.multiselect(
        "Select 6 players for a lineup:",
        options=player_list,
        key=f"selector_{contest_type}"
    )

    if len(st.session_state[f"selector_{contest_type}"]) == 6:
        selected_players = st.session_state[f"selector_{contest_type}"]
        lineup_df = players_df[players_df['display_name'].isin(selected_players)]
        
        most_expensive_player = lineup_df.loc[lineup_df['Salary'].idxmax()]
        default_captain_index = selected_players.index(most_expensive_player['display_name'])
        st.selectbox("Select Captain:", options=selected_players, index=default_captain_index, key=f"captain_{contest_type}")

        min_salary = lineup_df['Salary'].min()
        players_at_min_salary = lineup_df[lineup_df['Salary'] == min_salary]
        
        if len(players_at_min_salary) > 1:
            underdog_options = players_at_min_salary['display_name'].tolist()
            st.selectbox("TIE-BREAKER: Select Underdog Player:", options=underdog_options, key=f"underdog_{contest_type}")
        elif len(players_at_min_salary) == 1:
            st.session_state[f"underdog_{contest_type}"] = players_at_min_salary['display_name'].iloc[0]

    st.button("Add Lineup", key=f"add_{contest_type}", on_click=add_lineup_callback, args=(contest_type,))

    if st.session_state[session_state_key]:
        st.subheader("Current Lineups for Assessment")
        
        display_data = []
        for lineup_data in st.session_state[session_state_key]:
            lineup_df = players_df[players_df['display_name'].isin(lineup_data['players'])]
            lineup_df_sorted = lineup_df.sort_values(by='Salary', ascending=False)
            
            player_list_str = []
            for _, player_row in lineup_df_sorted.iterrows():
                player_name = player_row['display_name']
                role = ""
                if player_name == lineup_data['captain']:
                    role = " (c)"
                if player_name == lineup_data['underdog']:
                    role = f"{role[:-1]}, ud)" if role else " (ud)"
                player_list_str.append(f"{player_name}{role}")
            
            display_data.append({
                "Lineup Name": lineup_data['name'],
                "Players": ", ".join(player_list_str)
            })
        
        st.dataframe(pd.DataFrame(display_data), hide_index=True, use_container_width=True)

        if st.button("Clear All Lineups", key=f"clear_{contest_type}"):
            st.session_state[session_state_key] = []
            st.rerun()

    st.divider()

    st.subheader("Run Simulation")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Entry Fee (£)", min_value=0.0, step=1.0, key=f"entry_fee_{contest_type}")
    with col2:
        st.text_input("Prizes (comma-separated)", key=f"prizes_{contest_type}")
    
    if st.button("Assess All Lineups", type="primary", key=f"assess_{contest_type}"):
        if not st.session_state[session_state_key]:
            st.warning("Please add at least one lineup to assess.")
            return

        try:
            prizes_str = st.session_state[f"prizes_{contest_type}"]
            prizes = [float(p.strip()) for p in prizes_str.split(',')] if prizes_str else []
        except ValueError:
            st.error("Invalid prize format. Please enter a comma-separated list of numbers.")
            return

        expected_values, win_probabilities = run_monte_carlo_simulation(
            players_df, st.session_state[session_state_key], std_col, prizes, st.session_state[f"entry_fee_{contest_type}"], use_cut_model
        )
        
        lineup_details = []
        for lineup_data in st.session_state[session_state_key]:
            lineup_df = players_df[players_df['display_name'].isin(lineup_data['players'])]
            lineup_df_sorted = lineup_df.sort_values(by='Salary', ascending=False)
            
            player_list_str = []
            for _, player_row in lineup_df_sorted.iterrows():
                player_name = player_row['display_name']
                role = ""
                if player_name == lineup_data['captain']:
                    role = " (c)"
                if player_name == lineup_data['underdog']:
                    role = f"{role[:-1]}, ud)" if role else " (ud)"
                player_list_str.append(f"{player_name}{role}")

            base_points = lineup_df['FPPG'].sum()
            captain_player = lineup_df[lineup_df['display_name'] == lineup_data['captain']]
            underdog_player = lineup_df[lineup_df['display_name'] == lineup_data['underdog']]
            captain_bonus = captain_player['FPPG'].iloc[0] * 0.25
            underdog_bonus = underdog_player['FPPG'].iloc[0] * 0.25
            
            if captain_player.index[0] == underdog_player.index[0]:
                projected_score = base_points + captain_bonus
            else:
                projected_score = base_points + captain_bonus + underdog_bonus

            lineup_details.append({
                "Lineup": lineup_data['name'],
                "Players": ", ".join(player_list_str),
                "Projected Score": projected_score,
                "Expected Value (£)": expected_values.get(lineup_data['name'], -st.session_state[f"entry_fee_{contest_type}"]),
                "Win Probability": win_probabilities.get(lineup_data['name'], 0)
            })

        results_df = pd.DataFrame(lineup_details).sort_values(by="Expected Value (£)", ascending=False)
        st.session_state.simulation_results = results_df
        
        def color_ev(val):
            color = 'green' if val > 0 else 'red' if val < 0 else 'white'
            return f'color: {color}'

        st.subheader("Simulation Results")
        st.dataframe(
            results_df.style.applymap(color_ev, subset=['Expected Value (£)']).format({'Expected Value (£)': "£{:.2f}", 'Projected Score': "{:.2f}"}),
            use_container_width=True,
            column_config={
                "Win Probability": st.column_config.ProgressColumn(
                    "Win Probability (%)",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100,
                )
            },
            hide_index=True
        )

# --- Database Functions ---
def load_opponent_db():
    if os.path.exists(OPPONENT_DB_FILE):
        return pd.read_csv(OPPONENT_DB_FILE)
    return pd.DataFrame(columns=['Date', 'Opponent Name', 'Expected Value (£)'])

def save_to_opponent_db(results_df):
    if results_df.empty:
        st.warning("No simulation results to save.")
        return
        
    db_data = results_df[['Lineup', 'Expected Value (£)']].copy()
    db_data.rename(columns={'Lineup': 'Opponent Name'}, inplace=True)
    db_data['Date'] = datetime.now().strftime("%Y-%m-%d")
    
    db_df = load_opponent_db()
    updated_db = pd.concat([db_df, db_data], ignore_index=True)
    updated_db.to_csv(OPPONENT_DB_FILE, index=False)
    st.success("Results saved to opponent database!")

# --- Opponent Database Tab Function ---
def create_opponent_database_tab():
    st.header("Opponent Database")
    st.info("This table shows the long-term performance of the opponents you have assessed.")
    
    db_df = load_opponent_db()
    
    if db_df.empty:
        st.warning("No data in the opponent database. Run an assessment and save the results to begin.")
    else:
        summary_df = db_df.groupby('Opponent Name').agg(
            Average_EV=('Expected Value (£)', 'mean'),
            Entries_Tracked=('Opponent Name', 'size')
        ).sort_values(by='Average_EV', ascending=False).reset_index()

        def color_ev(val):
            color = 'green' if val > 0 else 'red' if val < 0 else 'white'
            return f'color: {color}'
            
        st.dataframe(
            summary_df.style.applymap(color_ev, subset=['Average_EV']).format({'Average_EV': "£{:.2f}"}),
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear Opponent Database", type="secondary"):
            if os.path.exists(OPPONENT_DB_FILE):
                os.remove(OPPONENT_DB_FILE)
                st.success("Opponent database cleared.")
                st.rerun()

# --- Lineup Assessor Tab Function ---
def create_lineup_assessor_tab(players_df_72, players_df_18):
    st.header("Lineup Assessor")
    contest_type = st.radio("Select Contest Type:", ("72-Hole Contest", "Single Round Showdown"), horizontal=True)
    
    if contest_type == "72-Hole Contest":
        players_df = players_df_72
        std_col = 'FPPG_std_72'
        session_state_key = 'assessed_lineups_72_hole'
        use_cut_model = 'make_cut' in players_df_72.columns
    else:
        if players_df_18 is None:
            st.error("Please upload a 'Single Round Projections' file to assess Showdown lineups.")
            return
        players_df = players_df_18
        std_col = 'FPPG_std_18'
        session_state_key = 'assessed_lineups_showdown'
        use_cut_model = False

    build_and_assess_ui(players_df, contest_type, std_col, session_state_key, use_cut_model)

# --- Main Streamlit User Interface ---
st.set_page_config(page_title="DFS Optimizer", layout="wide")
st.title("🚀 DFS Lineup Optimizer")

# --- Initialize Session State ---
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Lineup Generator"
if 'final_lineups_72' not in st.session_state:
    st.session_state.final_lineups_72 = None
if 'final_lineups_18' not in st.session_state:
    st.session_state.final_lineups_18 = None
if 'summary_72' not in st.session_state:
    st.session_state.summary_72 = None
if 'summary_18' not in st.session_state:
    st.session_state.summary_18 = None
if 'upload_72' not in st.session_state:
    st.session_state.upload_72 = None
if 'upload_18' not in st.session_state:
    st.session_state.upload_18 = None
if 'assessed_lineups_72_hole' not in st.session_state:
    st.session_state.assessed_lineups_72_hole = []
if 'assessed_lineups_showdown' not in st.session_state:
    st.session_state.assessed_lineups_showdown = []
if 'entry_fee_72_hole' not in st.session_state:
    st.session_state['entry_fee_72_hole'] = 50.0
if 'prizes_72_hole' not in st.session_state:
    st.session_state['prizes_72_hole'] = ""
if 'entry_fee_showdown' not in st.session_state:
    st.session_state['entry_fee_showdown'] = 50.0
if 'prizes_showdown' not in st.session_state:
    st.session_state['prizes_showdown'] = ""
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

st.sidebar.header("⚙️ Settings")
st.sidebar.info("Upload your projections and salary files to begin.")
show_debug_panel = st.sidebar.checkbox("Show debug panel", value=False)

with st.sidebar.expander("File Uploads", expanded=True):
    projections_file_72 = st.file_uploader("72-Hole Projections", type="csv")
    projections_file_18 = st.file_uploader("Single Round Projections (Optional)", type="csv")
    cut_prob_file = st.file_uploader("Make Cut Probabilities", type="csv")
    salaries_file = st.file_uploader("Salaries", type="csv")

# Initialize variable for players_df_72 to use in multiselect
players_df_72 = None

if projections_file_72 and salaries_file:
    df_projections_72 = pd.read_csv(projections_file_72)
    df_salaries = pd.read_csv(salaries_file)
    df_projections_72.columns = df_projections_72.columns.str.strip()
    df_salaries.columns = df_salaries.columns.str.strip()

    df_salaries['merge_name'] = df_salaries['FName'] + ' ' + df_salaries['Name']
    manual_mappings = load_manual_mappings()
    df_projections_72['mapped_dk_name'] = df_projections_72['dk_name'].replace(manual_mappings)
    merged_df_72 = pd.merge(df_projections_72, df_salaries, left_on="mapped_dk_name", right_on="merge_name", how="inner")

    if cut_prob_file:
        df_cut_prob = pd.read_csv(cut_prob_file)
        df_cut_prob.columns = df_cut_prob.columns.str.strip()
        
        if 'player_name' not in df_cut_prob.columns:
            st.error("Your 'Make Cut Probabilities' file is missing the required 'player_name' column.")
            st.stop()
        if 'make_cut' not in df_cut_prob.columns:
            st.error("Your 'Make Cut Probabilities' file is missing the required 'make_cut' column.")
            st.stop()
        
        df_cut_prob.rename(columns={'player_name': 'dk_name'}, inplace=True)
        df_cut_prob['make_cut'] = 1 / pd.to_numeric(df_cut_prob['make_cut'], errors='coerce')
            
        df_cut_prob['mapped_dk_name'] = df_cut_prob['dk_name'].replace(manual_mappings)
        merged_df_72 = pd.merge(merged_df_72, df_cut_prob[['mapped_dk_name', 'make_cut']], on="mapped_dk_name", how="left")
        merged_df_72['make_cut'].fillna(0, inplace=True)

    if projections_file_18:
        df_projections_18 = pd.read_csv(projections_file_18)
        df_projections_18.columns = df_projections_18.columns.str.strip()
        if len(df_projections_18.columns) >= 6:
            df_projections_18.rename(columns={df_projections_18.columns[5]: 'tee_time'}, inplace=True)
        else:
            st.warning("Single Round Projections file has fewer than 6 columns. Cannot assign tee_time (6th column). Ignoring tee time filter for Showdown.")
            df_projections_18['tee_time'] = pd.NA
        df_projections_18['mapped_dk_name'] = df_projections_18['dk_name'].replace(manual_mappings)
        merged_df_18 = pd.merge(df_projections_18, df_salaries, left_on="mapped_dk_name", right_on="merge_name", how="inner")
    else:
        merged_df_18 = None

    unmatched_proj_names = sorted(list(set(df_projections_72['dk_name']) - set(merged_df_72['dk_name'])))
    unmatched_salary_names = sorted(list(set(df_salaries['merge_name']) - set(merged_df_72['merge_name'])))

    if unmatched_proj_names and unmatched_salary_names:
        st.error(f"Found {len(unmatched_proj_names)} unmatched projection players and {len(unmatched_salary_names)} unmatched salary players.")
        with st.expander("🔗 Manually Match Unmatched Players", expanded=True):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                proj_choice = st.selectbox("Select Unmatched Projection Player:", options=[""] + unmatched_proj_names)
            with col2:
                salary_choice = st.selectbox("Select Unmatched Salary Player to Match:", options=[""] + unmatched_salary_names)
            with col3:
                st.write("") 
                if st.button("💾 Save Match", use_container_width=True):
                    if proj_choice and salary_choice:
                        manual_mappings[proj_choice] = salary_choice
                        save_manual_mappings(manual_mappings)
                        st.success(f"Saved match: {proj_choice} -> {salary_choice}. The app will now reload.")
                        st.rerun()
                    else:
                        st.warning("Please select a player from both lists.")
    else:
        st.success(f"✅ All players matched! Found {len(merged_df_72)} total players.")
        
        players_df_72 = process_player_df(merged_df_72, 'total_points')
        
        if players_df_72 is not None:
            if 'std_dev' in players_df_72.columns:
                st.info("Using 'std_dev' column from 72-hole file for simulations.")
                players_df_72['FPPG_std_72'] = players_df_72['std_dev']
                players_df_72['FPPG_std_18'] = players_df_72['std_dev'] / 2
            else:
                st.warning("No 'std_dev' column found. Using estimates for standard deviation.")
                players_df_72['FPPG_std_72'] = players_df_72['FPPG'] * 0.30
                conditions = [(players_df_72['Salary'] >= 9.5), (players_df_72['Salary'] >= 7.5) & (players_df_72['Salary'] < 9.5), (players_df_72['Salary'] < 7.5)]
                choices = [0.50, 0.65, 0.80]
                players_df_72['std_multiplier'] = np.select(conditions, choices, default=0.65)
                players_df_72['FPPG_std_18'] = players_df_72['FPPG'] * players_df_72['std_multiplier']

            if show_debug_panel:
                st.subheader("Simulation check (single golfer)")
                golfer_options = players_df_72['display_name'].dropna().sort_values().tolist()
                selected_golfer = st.selectbox("Select golfer", options=golfer_options)
                selected_row = players_df_72.loc[players_df_72['display_name'] == selected_golfer].iloc[0]
                fppg = float(selected_row['FPPG'])
                make_cut_prob = float(selected_row['make_cut']) if 'make_cut' in players_df_72.columns else 0.0
                std_col = 'FPPG_std_72' if 'FPPG_std_72' in players_df_72.columns else None
                std_value = None
                if std_col is not None:
                    std_value = selected_row.get(std_col)
                    if pd.isna(std_value):
                        std_value = None

                info_col1, info_col2 = st.columns(2)
                info_col1.metric("FPPG", f"{fppg:.2f}")
                info_col2.metric("Make-cut probability", f"{make_cut_prob:.2%}")

                if std_value is None:
                    std_value = st.number_input("Standard deviation", min_value=0.1, value=8.0, step=0.1)
                else:
                    st.write(f"Using standard deviation from {std_col}: {std_value:.2f}")

                iterations = st.slider("Iterations", 1_000, 200_000, 20_000, step=1_000)
                mu_miss = 0.5 * fppg
                mu_made = (fppg - (1 - make_cut_prob) * mu_miss) / max(make_cut_prob, 1e-6)
                rng = np.random.default_rng()
                made_cut = rng.random(iterations) < make_cut_prob
                scores = np.empty(iterations)
                scores[made_cut] = rng.normal(mu_made, std_value, made_cut.sum())
                scores[~made_cut] = rng.normal(mu_miss, 0.5 * std_value, (~made_cut).sum())
                sim_mean = scores.mean()
                p10, p50, p90 = np.percentile(scores, [10, 50, 90])

                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                stat_col1.metric("Sim mean", f"{sim_mean:.2f}")
                stat_col2.metric("P10", f"{p10:.2f}")
                stat_col3.metric("P50", f"{p50:.2f}")
                stat_col4.metric("P90", f"{p90:.2f}")
                st.line_chart(pd.Series(np.sort(scores), name="Simulated score"))

            if merged_df_18 is not None:
                players_df_18 = process_player_df(merged_df_18, 'scoring_points')
            else:
                players_df_18 = players_df_72.copy()

            if players_df_18 is not None:
                players_df_18 = pd.merge(
                    players_df_18.drop(columns=['FPPG_std_72', 'FPPG_std_18'], errors='ignore'),
                    players_df_72[['display_name', 'FPPG_std_72', 'FPPG_std_18']],
                    on='display_name',
                    how='left'
                )
            
                col1, col2, col3, _ = st.columns([1, 1, 1, 4])
                if col1.button("Lineup Generator"):
                    st.session_state.active_tab = "Lineup Generator"
                if col2.button("Lineup Assessor"):
                    st.session_state.active_tab = "Lineup Assessor"
                if col3.button("Opponent Database"):
                    st.session_state.active_tab = "Opponent Database"
                
                st.divider()

                if st.session_state.active_tab == "Lineup Generator":
                    st.header("Lineup Generator")
                    optimizer_type = st.radio("Select Contest Type:", ("72-Hole Contest", "Single Round Showdown"), horizontal=True)
                    st.divider()
                    
                    st.sidebar.markdown("---")
                    st.sidebar.header("🛠️ Optimizer Controls")
                    game_id = st.sidebar.number_input("Enter Game ID for Upload", value=977562)
                    num_lineups = st.sidebar.slider("Number of Lineups", 1, 100, 25)
                    salary_cap = st.sidebar.slider("Salary Cap", 90.0, 110.0, 100.0, 0.1)
                    diversity = st.sidebar.slider("Diversity", 0, 100, 0)
                    max_exposure = st.sidebar.slider("Max Player Exposure (%)", 0, 100, 100, help="Limit the percentage of lineups any single player can appear in. Set to 100% for no restriction.")
                    disregard_exposure_players = st.sidebar.multiselect(
                        "Disregard Max Exposure for Players",
                        options=players_df_72['display_name'].tolist() if players_df_72 is not None else [],
                        default=[],
                        help="Select players to exclude from the max exposure limit."
                    )
                    st.sidebar.markdown("---")
                    st.sidebar.header("🗓️ Tee Time Constraints")
                    max_early = st.sidebar.slider("Max Early Wave Players", 0, 6, 6)
                    max_late = st.sidebar.slider("Max Late Wave Players", 0, 6, 6)
                    tee_time_cutoff = st.sidebar.time_input(
                        "All Players Tee Off Before (Optional)",
                        value=None,
                        help="Select a time to filter players who tee off before it. Leave blank to include all players regardless of tee time."
                    )
                    st.sidebar.markdown("---")
                    st.sidebar.header("🔍 Optional Filters")
                    min_salary_filter = st.sidebar.checkbox("Apply minimum total salary filter (>= £98.0)")
                    underdog_salary_filter = st.sidebar.checkbox("Apply underdog salary filter (>= £15.0)")
                    
                    if st.button("🔥 Generate Lineups"):
                        if optimizer_type == "72-Hole Contest":
                            players_to_optimize = players_df_72
                            session_suffix = "_72"
                        else:
                            if players_df_18 is None:
                                st.error("Please upload a 'Single Round Projections' file for Showdown contests.")
                                st.stop()
                            players_to_optimize = players_df_18
                            session_suffix = "_18"
                        
                        with st.spinner("Optimizing..."):
                            scaled_salary_cap = salary_cap * 10
                            st.session_state[f'final_lineups{session_suffix}'], st.session_state[f'summary{session_suffix}'], st.session_state[f'upload{session_suffix}'] = run_optimizer(
                                players_to_optimize, num_lineups, scaled_salary_cap, min_salary_filter, underdog_salary_filter, game_id, diversity, max_early, max_late, tee_time_cutoff, max_exposure, disregard_exposure_players
                            )
                            
                    session_suffix_display = "_72" if optimizer_type == "72-Hole Contest" else "_18"
                    if st.session_state.get(f'final_lineups{session_suffix_display}') is not None:
                        st.success("🎉 Optimization Complete!")
                        st.header("Generated Lineups")
                        st.dataframe(st.session_state[f'final_lineups{session_suffix_display}'], use_container_width=True)
                        
                        st.header("Player Summary")
                        st.dataframe(st.session_state[f'summary{session_suffix_display}'].style.format({"% In Lineups": "{:.1%}", "% As Captain": "{:.1%}", "Exposure (%)": "{:.2f}%"}), use_container_width=True)

                        dl_col1, dl_col2 = st.columns(2)
                        with dl_col1:
                            st.download_button(label="📥 Download Lineups for Upload (CSV)", data=st.session_state[f'upload{session_suffix_display}'].to_csv(index=False, header=False).encode('utf-8'), file_name='upload_lineups.csv', mime='text/csv')
                        with dl_col2:
                            excel_data = to_excel(st.session_state[f'final_lineups{session_suffix_display}'], st.session_state[f'summary{session_suffix_display}'])
                            st.download_button(label="📊 Download Full Summary (Excel)", data=excel_data, file_name='lineup_summary.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

                elif st.session_state.active_tab == "Lineup Assessor":
                    create_lineup_assessor_tab(players_df_72, players_df_18)
                    
                    if st.session_state.get('simulation_results') is not None:
                        if st.button("Save Results to Database"):
                            save_to_opponent_db(st.session_state.simulation_results)

                elif st.session_state.active_tab == "Opponent Database":
                    create_opponent_database_tab()
else:
    st.info("👋 Welcome! Please upload your projection and salary files to begin.")
