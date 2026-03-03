"""
analyze.py
----------
Run analysis reports on the golf DFS database.

Usage:
  python analyze.py summary
  python analyze.py calibration
  python analyze.py calibration --tournament "Genesis Invitational"
  python analyze.py calibration-ft
  python analyze.py correlations
  python analyze.py captains
  python analyze.py salary
  python analyze.py player "Scottie Scheffler"
  python analyze.py predict --tournament "Honda Classic" --file dk_proj.csv
"""

import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from db import GolfDFSDatabase


def fmt_pct(val):
    try: return f"{float(val):.1%}"
    except: return "  -  "

def fmt_diff(val):
    try: return f"{float(val):+.1%}"
    except: return "  -  "


def cmd_summary(db, args):
    df = db.get_tournament_summary()
    print("\n" + "="*75)
    print("TOURNAMENT DATABASE SUMMARY")
    print("="*75)
    print(f"\n{'Tournament':<30} {'Year':>4} {'Players':>7} {'DK Proj':>8} {'DK Act':>7} {'FT £2':>6} {'FT £10':>7} {'FT Sal':>7}")
    print("-"*75)
    for _, r in df.iterrows():
        def tick(n): return "✓" if n > 0 else "✗"
        print(f"{r['name']:<30} {r['year']:>4} {int(r['n_players']):>7}  "
              f"   {tick(r['has_dk_proj'])}      {tick(r['has_dk_actual'])}    {tick(r['has_ft2'])}      {tick(r['has_ft10'])}     {tick(r['has_ft_salary'])}")
    print(f"\nTotal records: {int(df['n_players'].sum())}  |  Tournaments: {len(df)}")


def cmd_calibration(db, args):
    t = getattr(args, 'tournament', None)
    df = db.calibration_analysis(t, use_ft_salary=False)
    title = f"CALIBRATION (DK SALARY TIERS): {t}" if t else "CALIBRATION (DK SALARY TIERS): ALL TOURNAMENTS"
    print("\n" + "="*80)
    print(title)
    print("="*80)
    has_proj = "dk_proj_avg" in df.columns
    if has_proj:
        print(f"\n{'Tier':<12} {'n':>3}  {'DK Proj':>8} {'DK Act':>8} {'ProjBias':>9} │ {'FT £2':>7} {'DK→£2':>8} │ {'FT £10':>8} {'DK→£10':>9}")
        print("-"*80)
        for _, r in df.iterrows():
            print(f"{r['tier']:<12} {int(r['n']):>3}  {fmt_pct(r.get('dk_proj_avg')):>8} "
                  f"{fmt_pct(r.get('dk_actual_avg')):>8} {fmt_diff(r.get('proj_bias')):>9} │ "
                  f"{fmt_pct(r.get('ft2_avg')):>7} {fmt_diff(r.get('ft2_diff')):>8} │ "
                  f"{fmt_pct(r.get('ft10_avg')):>8} {fmt_diff(r.get('ft10_diff')):>9}")
    else:
        print(f"\n{'Tier':<12} {'n':>3}  {'DK Act':>8} │ {'FT £2':>7} {'DK→£2':>8} │ {'FT £10':>8} {'DK→£10':>9}")
        print("-"*60)
        for _, r in df.iterrows():
            print(f"{r['tier']:<12} {int(r['n']):>3}  {fmt_pct(r.get('dk_actual_avg')):>8} │ "
                  f"{fmt_pct(r.get('ft2_avg')):>7} {fmt_diff(r.get('ft2_diff')):>8} │ "
                  f"{fmt_pct(r.get('ft10_avg')):>8} {fmt_diff(r.get('ft10_diff')):>9}")
    print("\n(Positive diff = FanTeam HIGHER than DK, Negative = LOWER)")


def cmd_calibration_ft(db, args):
    t = getattr(args, 'tournament', None)
    df = db.calibration_analysis(t, use_ft_salary=True)
    title = f"CALIBRATION (FANTEAM SALARY TIERS): {t}" if t else "CALIBRATION (FANTEAM SALARY TIERS): ALL TOURNAMENTS"
    print("\n" + "="*70)
    print(title)
    print("="*70)
    print(f"\n{'Tier':<12} {'n':>3}  {'DK Act':>8} │ {'FT £2':>7} {'DK→£2':>8} │ {'FT £10':>8} {'DK→£10':>9}")
    print("-"*60)
    for _, r in df.iterrows():
        print(f"{r['tier']:<12} {int(r['n']):>3}  {fmt_pct(r.get('dk_actual_avg')):>8} │ "
              f"{fmt_pct(r.get('ft2_avg')):>7} {fmt_diff(r.get('ft2_diff')):>8} │ "
              f"{fmt_pct(r.get('ft10_avg')):>8} {fmt_diff(r.get('ft10_diff')):>9}")
    print("\n(Positive diff = FanTeam HIGHER than DK, Negative = LOWER)")


def cmd_salary(db, args):
    df = db.salary_comparison()
    if df.empty:
        print("No FanTeam salary data yet. Re-import FanTeam files to populate.")
        return

    print("\n" + "="*70)
    print("DK vs FANTEAM SALARY COMPARISON")
    print("="*70)

    print("\nAverage FT/DK salary ratio by tournament (2.0 = perfectly proportional):")
    for t, g in df.groupby("tournament"):
        print(f"  {t:<30} avg ratio: {g['ft_to_dk_ratio'].mean():.3f}  (n={len(g)})")

    print("\nPlayers priced HIGHER on FanTeam relative to DK (ratio > 2.1):")
    high = df[df["ft_to_dk_ratio"] > 2.1].sort_values("ft_to_dk_ratio", ascending=False)
    if len(high):
        print(f"  {'Player':<25} {'Tournament':<25} {'DK':>7} {'FT':>6} {'Ratio':>7}")
        print("  " + "-"*68)
        for _, r in high.drop_duplicates(subset=["player"]).head(15).iterrows():
            print(f"  {r['player']:<25} {r['tournament']:<25} ${r['dk_salary']:>6,} {r['ft_salary']:>5.1f}M {r['ft_to_dk_ratio']:>7.3f}")
    else:
        print("  None found.")

    print("\nPlayers priced CHEAPER on FanTeam relative to DK (ratio < 1.9):")
    low = df[df["ft_to_dk_ratio"] < 1.9].sort_values("ft_to_dk_ratio")
    if len(low):
        print(f"  {'Player':<25} {'Tournament':<25} {'DK':>7} {'FT':>6} {'Ratio':>7}")
        print("  " + "-"*68)
        for _, r in low.drop_duplicates(subset=["player"]).head(15).iterrows():
            print(f"  {r['player']:<25} {r['tournament']:<25} ${r['dk_salary']:>6,} {r['ft_salary']:>5.1f}M {r['ft_to_dk_ratio']:>7.3f}")
    else:
        print("  None found.")


def cmd_correlations(db, args):
    df = db.correlations()
    print("\n" + "="*60)
    print("DK ACTUAL → FANTEAM CORRELATION BY TOURNAMENT")
    print("="*60)
    print(f"\n{'Tournament':<30} {'r (£2)':>8} {'n':>4} │ {'r (£10)':>8} {'n':>4}")
    print("-"*55)
    for _, r in df.iterrows():
        r2  = f"{r['r_dk_ft2']:.3f}"  if pd.notna(r.get('r_dk_ft2'))  else "   -  "
        r10 = f"{r['r_dk_ft10']:.3f}" if pd.notna(r.get('r_dk_ft10')) else "   -  "
        n2  = str(int(r['n_ft2']))    if pd.notna(r.get('n_ft2'))     else " - "
        n10 = str(int(r['n_ft10']))   if pd.notna(r.get('n_ft10'))    else " - "
        print(f"{r['tournament']:<30} {r2:>8} {n2:>4} │ {r10:>8} {n10:>4}")
    all_r2  = df['r_dk_ft2'].dropna()
    all_r10 = df['r_dk_ft10'].dropna()
    if len(all_r2):
        print(f"\n{'Average':<30} {all_r2.mean():>8.3f}       │ {all_r10.mean():>8.3f}")


def cmd_captains(db, args):
    df = db.captain_concentration()
    print("\n" + "="*75)
    print("CAPTAIN CONCENTRATION (Top-3 %) BY TOURNAMENT")
    print("="*75)
    print(f"\n{'Tournament':<30} {'£2 Top-3':>9} {'£2 Top Capt':>20} │ {'£10 Top-3':>10} {'£10 Top Capt':>20}")
    print("-"*90)
    for _, r in df.iterrows():
        t3_2  = fmt_pct(r.get("ft2_top3_cap"))
        c2    = (r.get("ft2_top_captain") or "-")[:18]
        t3_10 = fmt_pct(r.get("ft10_top3_cap"))
        c10   = (r.get("ft10_top_captain") or "-")[:18]
        print(f"{r['tournament']:<30} {t3_2:>9} {c2:>20} │ {t3_10:>10} {c10:>20}")


def cmd_player(db, args):
    df = db.player_history(args.name)
    if df.empty:
        print(f"No data found for player: {args.name}")
        return
    print(f"\n{'='*80}")
    print(f"PLAYER HISTORY: {args.name.upper()}")
    print(f"{'='*80}")
    print(f"\n{'Tournament':<28} {'DK Sal':>7} {'FT Sal':>7} {'DK Proj':>8} {'DK Act':>7} │ {'FT£2':>6} {'Cap£2':>7} │ {'FT£10':>7} {'Cap£10':>8} {'Pts':>7}")
    print("-"*95)
    for _, r in df.iterrows():
        ft_sal = f"{r['ft_salary']:.1f}M" if pd.notna(r.get('ft_salary')) else "   -  "
        print(
            f"{r['tournament']:<28} "
            f"${r['dk_salary']:>6,} " if pd.notna(r.get('dk_salary')) else "       - "
            f"{ft_sal:>7} "
            f"{fmt_pct(r.get('dk_projected_own')):>8} "
            f"{fmt_pct(r.get('dk_actual_own')):>7} │ "
            f"{fmt_pct(r.get('ft2_own')):>6} "
            f"{fmt_pct(r.get('ft2_captain_own')):>7} │ "
            f"{fmt_pct(r.get('ft10_own')):>7} "
            f"{fmt_pct(r.get('ft10_captain_own')):>8} "
            f"{r['total_pts']:>7.1f}" if pd.notna(r.get('total_pts')) else "      -"
        )


def cmd_predict(db, args):
    df = pd.read_csv(args.file)
    print(f"\nLoaded {len(df)} players from {args.file}")

    MULT = {
        "£10": {(10000, 99999): 0.70, (9000, 9999): 0.90,
                (8000, 8999): 1.00, (7000, 7999): 1.08, (0, 6999): 1.10},
        "£2":  {(10000, 99999): 0.72, (9000, 9999): 0.92,
                (8000, 8999): 1.00, (7000, 7999): 1.05, (0, 6999): 1.08},
    }

    def get_mult(salary, entry):
        for (lo, hi), m in MULT[entry].items():
            if lo <= salary <= hi: return m
        return 1.0

    salary_col = next((c for c in df.columns if "salary" in c.lower()), None)
    own_col    = next((c for c in df.columns if "projected_ownership" in c.lower() or c.lower() == "ownership"), None)
    name_col   = next((c for c in df.columns if "name" in c.lower()), df.columns[0])

    if not salary_col or not own_col:
        print(f"Could not find salary/ownership columns. Columns: {df.columns.tolist()}")
        return

    results = []
    for _, row in df.iterrows():
        sal = int(row[salary_col])
        own = float(row[own_col])
        if own > 1.5: own = own / 100
        results.append({
            "Player":       row[name_col],
            "DK Salary":    sal,
            "DK Projected": own,
            "FT £2 Pred":   round(own * get_mult(sal, "£2"), 4),
            "FT £10 Pred":  round(own * get_mult(sal, "£10"), 4),
        })

    out = pd.DataFrame(results).sort_values("DK Projected", ascending=False)
    print(f"\n{'='*70}")
    print(f"FANTEAM OWNERSHIP PREDICTIONS: {getattr(args, 'tournament', 'Tournament')}")
    print(f"{'='*70}")
    print(f"\n{'Player':<26} {'Salary':>8} {'DK Proj':>9} {'FT £2 Pred':>11} {'FT £10 Pred':>12}")
    print("-"*68)
    for _, r in out.iterrows():
        print(f"{r['Player']:<26} ${r['DK Salary']:>7,} {fmt_pct(r['DK Projected']):>9} "
              f"{fmt_pct(r['FT £2 Pred']):>11} {fmt_pct(r['FT £10 Pred']):>12}")


def cmd_leverage(db, args):
    """
    Leverage analysis: projected pts vs ownership vs actual pts.
    Identifies players who were good/bad GPP targets.
    """
    df = db.get_ownership_data(getattr(args, 'tournament', None))

    # Need projected pts and actual ownership at minimum
    df = df.dropna(subset=["dk_projected_pts", "dk_actual_own", "total_pts"])
    if df.empty:
        print("Not enough data for leverage analysis. Need tournaments with DK projected pts imported.")
        return

    df = df.copy()
    df["proj_leverage"]   = df["dk_projected_pts"] / (df["dk_actual_own"] * 100)
    df["actual_leverage"] = df["total_pts"]         / (df["dk_actual_own"] * 100)
    df["pts_vs_proj"]     = df["total_pts"] - df["dk_projected_pts"]
    df["own_vs_proj"]     = df["dk_actual_own"] - df["dk_projected_own"].fillna(0)

    print("\n" + "="*80)
    print("LEVERAGE ANALYSIS: PROJECTED vs ACTUAL POINTS PER OWNERSHIP %")
    print("="*80)

    df["tier"] = df["dk_salary"].apply(db._dk_tier)
    print(f"\n{'Tier':<12} {'n':>3}  {'Proj Pts':>9} {'Act Pts':>8} {'Pts Diff':>9} │ {'Proj Lev':>9} {'Act Lev':>8}")
    print("-"*65)
    for tier_key, g in df.groupby("tier"):
        label = tier_key[2:]
        print(f"{label:<12} {len(g):>3}  "
              f"{g['dk_projected_pts'].mean():>9.1f} "
              f"{g['total_pts'].mean():>8.1f} "
              f"{g['pts_vs_proj'].mean():>+9.1f} │ "
              f"{g['proj_leverage'].mean():>9.2f} "
              f"{g['actual_leverage'].mean():>8.2f}")

    # Best actual leverage plays - minimum 2% ownership to filter out noise
    MIN_OWN = 0.02
    df_filtered = df[df["dk_actual_own"] >= MIN_OWN]

    print(f"\n{'='*80}")
    print(f"BEST GPP PLAYS (≥{MIN_OWN:.0%} owned - scored big relative to ownership):")
    print(f"{'='*80}")
    top = df_filtered.nlargest(15, "actual_leverage")
    print(f"\n{'Player':<26} {'Tournament':<25} {'Salary':>7} {'Own%':>5} {'Proj':>6} {'Actual':>7} {'vs Proj':>8} {'Act Lev':>8}")
    print("-"*90)
    for _, r in top.iterrows():
        print(f"{r['player']:<26} {r['tournament']:<25} "
              f"${r['dk_salary']:>6,} {r['dk_actual_own']:>5.1%} "
              f"{r['dk_projected_pts']:>6.1f} {r['total_pts']:>7.1f} "
              f"{r['pts_vs_proj']:>+8.1f} {r['actual_leverage']:>8.2f}")

    print(f"\n{'='*80}")
    print("LEVERAGE TRAPS (popular plays that underdelivered):")
    print(f"{'='*80}")
    own_thresh = df["dk_actual_own"].quantile(0.75)
    traps = df[(df["dk_actual_own"] >= own_thresh) & (df["pts_vs_proj"] < 0)].nsmallest(10, "pts_vs_proj")
    print(f"\n{'Player':<26} {'Tournament':<25} {'Salary':>7} {'Own%':>5} {'Proj':>6} {'Actual':>7} {'vs Proj':>8}")
    print("-"*85)
    for _, r in traps.iterrows():
        print(f"{r['player']:<26} {r['tournament']:<25} "
              f"${r['dk_salary']:>6,} {r['dk_actual_own']:>5.1%} "
              f"{r['dk_projected_pts']:>6.1f} {r['total_pts']:>7.1f} "
              f"{r['pts_vs_proj']:>+8.1f}")

    df_std = df.dropna(subset=["dk_projected_std"])
    df_std = df_std[df_std["dk_actual_own"] >= MIN_OWN]
    if not df_std.empty:
        print(f"\n{'='*80}")
        print(f"CEILING PLAYS: High std_dev + low ownership, ≥{MIN_OWN:.0%} owned (GPP sweet spot):")
        print(f"{'='*80}")
        df_std = df_std.copy()
        df_std["ceiling_score"] = df_std["dk_projected_std"] / (df_std["dk_actual_own"] * 100)
        sweet = df_std.nlargest(15, "ceiling_score")
        print(f"\n{'Player':<26} {'Tournament':<25} {'Salary':>7} {'Own%':>5} {'Proj':>6} {'StdDev':>7} {'Ceil Score':>10} {'Actual':>7}")
        print("-"*95)
        for _, r in sweet.iterrows():
            print(f"{r['player']:<26} {r['tournament']:<25} "
                  f"${r['dk_salary']:>6,} {r['dk_actual_own']:>5.1%} "
                  f"{r['dk_projected_pts']:>6.1f} {r['dk_projected_std']:>7.1f} "
                  f"{r['ceiling_score']:>10.2f} {r['total_pts']:>7.1f}")


def cmd_captainvalue(db, args):
    """
    Captain value analysis: actual pts x 1.25 per FanTeam captain ownership %.
    Shows which salary tiers the field is over/under-captaining relative to scoring.
    """
    df = db.get_ownership_data(getattr(args, 'tournament', None))

    # Need actual points and at least one captain ownership column
    df = df.dropna(subset=["total_pts"])
    df = df[df["ft2_captain_own"].notna() | df["ft10_captain_own"].notna()]
    if df.empty:
        print("No captain value data available.")
        return

    df = df.copy()
    df["capt_pts"] = df["total_pts"] * 1.25  # points if captained

    # Captain value = captain points per % of field that captained you
    # Higher = better value relative to how popular the captain choice was
    MIN_CAPT = 0.005  # 0.5% minimum captain ownership to filter noise

    for col, label in [("ft2_captain_own", "£2"), ("ft10_captain_own", "£10")]:
        sub = df[df[col] >= MIN_CAPT].copy()
        if sub.empty:
            continue
        sub["capt_value"] = sub["capt_pts"] / (sub[col] * 100)

        print("\n" + "="*80)
        print(f"CAPTAIN VALUE ANALYSIS — {label} CONTEST")
        print("="*80)

        # By FanTeam salary tier
        sub["tier"] = sub["ft_salary"].apply(db._ft_tier)
        print(f"\nBy FanTeam salary tier (capt value = 1.25x pts per captain %):")
        print(f"\n{'Tier':<12} {'n':>3}  {'Capt Own%':>10} {'Pts':>7} {'Capt Pts':>9} {'Capt Value':>11}")
        print("-"*55)
        for tier_key, g in sub.groupby("tier"):
            label_t = tier_key[2:]
            print(f"{label_t:<12} {len(g):>3}  "
                  f"{g[col].mean():>10.1%} "
                  f"{g['total_pts'].mean():>7.1f} "
                  f"{g['capt_pts'].mean():>9.1f} "
                  f"{g['capt_value'].mean():>11.2f}")

        # Best actual captain plays
        print(f"\nBest captain plays (≥0.5% captain owned, highest value):")
        print(f"\n{'Player':<26} {'Tournament':<25} {'FT Sal':>7} {'Capt%':>6} {'Pts':>6} {'Capt Pts':>9} {'Value':>7}")
        print("-"*85)
        top = sub.nlargest(12, "capt_value")
        for _, r in top.iterrows():
            print(f"{r['player']:<26} {r['tournament']:<25} "
                  f"{r['ft_salary']:>6.1f}M "
                  f"{r[col]:>6.1%} "
                  f"{r['total_pts']:>6.1f} "
                  f"{r['capt_pts']:>9.1f} "
                  f"{r['capt_value']:>7.2f}")

        # Worst captain traps - high captain ownership, poor scoring
        print(f"\nCaptain traps (high captain ownership, low points delivered):")
        print(f"\n{'Player':<26} {'Tournament':<25} {'FT Sal':>7} {'Capt%':>6} {'Pts':>6} {'Capt Pts':>9} {'Value':>7}")
        print("-"*85)
        # Top 25% most captained players
        capt_thresh = sub[col].quantile(0.75)
        traps = sub[sub[col] >= capt_thresh].nsmallest(10, "capt_value")
        for _, r in traps.iterrows():
            print(f"{r['player']:<26} {r['tournament']:<25} "
                  f"{r['ft_salary']:>6.1f}M "
                  f"{r[col]:>6.1%} "
                  f"{r['total_pts']:>6.1f} "
                  f"{r['capt_pts']:>9.1f} "
                  f"{r['capt_value']:>7.2f}")


def main():
    parser = argparse.ArgumentParser(description="Golf DFS Database Analysis")
    parser.add_argument("command",
                        choices=["summary", "calibration", "calibration-ft", "leverage",
                                 "correlations", "captains", "salary", "player", "predict",
                                 "captainvalue"])
    parser.add_argument("name",         nargs="?", default=None,  help="Player name (for 'player')")
    parser.add_argument("--db",         default="golf_dfs.db",    help="Database path")
    parser.add_argument("--tournament", default=None,             help="Filter by tournament")
    parser.add_argument("--file",       default=None,             help="DK projection CSV (for 'predict')")
    args = parser.parse_args()

    db = GolfDFSDatabase(args.db)

    dispatch = {
        "summary":        cmd_summary,
        "calibration":    cmd_calibration,
        "calibration-ft": cmd_calibration_ft,
        "correlations":   cmd_correlations,
        "captains":       cmd_captains,
        "salary":         cmd_salary,
        "player":         cmd_player,
        "predict":        cmd_predict,
        "leverage":       cmd_leverage,
        "captainvalue":   cmd_captainvalue,
    }

    dispatch[args.command](db, args)
    db.close()


if __name__ == "__main__":
    main()
