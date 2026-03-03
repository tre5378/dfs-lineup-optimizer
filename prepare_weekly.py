"""
prepare_weekly.py
-----------------
Pre-tournament workflow script. Takes a DataGolf DK projection CSV,
applies FanTeam ownership calibration and captain model predictions,
and outputs an augmented CSV ready to load directly into the Streamlit app.

The augmented CSV is a drop-in replacement for the raw DataGolf file:
- projected_ownership  → replaced with calibrated FanTeam £2 prediction
- predicted_ft10_own   → new column: calibrated FanTeam £10 prediction  
- predicted_captain_own → new column: predicted £10 captain ownership
- All original columns preserved

Usage:
  python prepare_weekly.py --proj data/cognizant/dk_projected.csv
  python prepare_weekly.py --proj data/cognizant/dk_projected.csv --ft-salary data/cognizant/fanteam_2.csv
  python prepare_weekly.py --proj data/cognizant/dk_projected.csv --tournament "Cognizant Classic"
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from db import GolfDFSDatabase
from captain_model import (
    load_training_data, fit_model, predict_captain_own,
    build_features
)


# ---------------------------------------------------------------------------
# CALIBRATION: DK projected → FanTeam actual (from historical data)
# ---------------------------------------------------------------------------

# Multipliers derived from database analysis across 5 tournaments
# Applied to DK projected ownership to get expected FanTeam ownership
CALIBRATION = {
    # (ft_salary_tier): (ft2_multiplier, ft10_multiplier)
    '£20M+':   (0.72, 0.70),   # premium: field underweights on FT, especially £10
    '£18-20M': (1.00, 1.04),   # slight over-ownership on FT £10
    '£16-18M': (0.98, 0.99),   # essentially flat
    '£14-16M': (0.98, 1.07),   # slight over-ownership on FT £10
    '<£14M':   (1.05, 1.08),   # minor over-ownership both contests
}

# DK projection bias correction (DK overprojects premium player ownership)
# Applied BEFORE platform calibration
DK_PROJ_BIAS = {
    '£20M+':   -0.064,   # DK overprojects by 6.4pp
    '£18-20M': -0.027,
    '£16-18M': -0.008,
    '£14-16M':  0.000,
    '<£14M':    0.000,
}


def ft_salary_tier(ft_sal):
    if pd.isna(ft_sal):    return '<£14M'
    if ft_sal >= 20:       return '£20M+'
    elif ft_sal >= 18:     return '£18-20M'
    elif ft_sal >= 16:     return '£16-18M'
    elif ft_sal >= 14:     return '£14-16M'
    else:                  return '<£14M'


def calibrate_ownership(dk_proj_own: float, ft_salary: float, contest: str = 'ft2') -> float:
    """
    Convert DK projected ownership to calibrated FanTeam prediction.
    1. Correct DK projection bias for this salary tier
    2. Apply FanTeam platform multiplier
    """
    if pd.isna(dk_proj_own) or dk_proj_own is None:
        return None

    tier = ft_salary_tier(ft_salary)
    bias = DK_PROJ_BIAS.get(tier, 0)
    mult_idx = 0 if contest == 'ft2' else 1
    mult = CALIBRATION.get(tier, (1.0, 1.0))[mult_idx]

    # Correct bias then apply platform multiplier
    corrected = dk_proj_own + bias
    calibrated = corrected * mult

    return float(np.clip(calibrated, 0.001, 0.80))


def load_ft_salary(ft_salary_path: str) -> dict:
    """Load FanTeam salary CSV, return dict of last_name -> ft_salary."""
    df = pd.read_csv(ft_salary_path)
    df['ft_salary'] = df['Price'].astype(str).str.replace('M','').str.replace('£','').str.strip()
    df['ft_salary'] = pd.to_numeric(df['ft_salary'], errors='coerce')
    df['ft_last'] = df['Player'].str.split().str[-1].str.lower()
    return df.set_index('ft_last')['ft_salary'].to_dict()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare weekly FanTeam-calibrated projection file")
    parser.add_argument("--proj",       required=True, help="DataGolf DK projection CSV")
    parser.add_argument("--ft-salary",  default=None,  help="FanTeam salary CSV (e.g. fanteam_2.csv)")
    parser.add_argument("--tournament", default=None,  help="Tournament name (for context only)")
    parser.add_argument("--db",         default="golf_dfs.db")
    parser.add_argument("--out",        default=None,  help="Output filename (default: <proj>_calibrated.csv)")
    args = parser.parse_args()

    proj_path = Path(args.proj)
    out_path  = Path(args.out) if args.out else proj_path.parent / f"{proj_path.stem}_calibrated.csv"

    print(f"\n{'='*65}")
    print(f"WEEKLY PREPARATION: FanTeam Ownership Calibration")
    print(f"{'='*65}")
    print(f"  Input:      {proj_path}")
    print(f"  Output:     {out_path}")
    if args.tournament:
        print(f"  Tournament: {args.tournament}")

    # Load projection file
    df = pd.read_csv(args.proj)
    df.columns = df.columns.str.strip()
    orig_rows = len(df)

    print(f"\n  Loaded {orig_rows} players from projection file")
    print(f"  Columns: {df.columns.tolist()}")

    # Normalise projected_ownership to 0-1
    if 'projected_ownership' in df.columns:
        df['projected_ownership'] = pd.to_numeric(df['projected_ownership'], errors='coerce')
        # Convert if stored as percentage (e.g. 27.4 → 0.274)
        if df['projected_ownership'].median() > 1.5:
            df['projected_ownership'] = df['projected_ownership'] / 100
        # Remove placeholder values
        df.loc[df['projected_ownership'] > 0.60, 'projected_ownership'] = None
    else:
        print("  ⚠ No 'projected_ownership' column found — calibration will be limited")
        df['projected_ownership'] = None

    # Get FanTeam salaries
    if args.ft_salary:
        print(f"\n  Loading FanTeam salaries from: {args.ft_salary}")
        ft_sal_map = load_ft_salary(args.ft_salary)
        df['ft_last'] = df['dk_name'].str.split().str[-1].str.lower()
        df['ft_salary'] = df['ft_last'].map(ft_sal_map)
        matched = df['ft_salary'].notna().sum()
        print(f"  Matched {matched}/{len(df)} players to FanTeam salaries")
        df.drop(columns=['ft_last'], inplace=True)
    else:
        print("\n  No FanTeam salary file provided — estimating ft_salary as dk_salary / 500")
        df['ft_salary'] = df['dk_salary'] / 500

    # Apply calibration
    print("\n  Applying calibration...")
    df['ft2_predicted_own']  = df.apply(
        lambda r: calibrate_ownership(r['projected_ownership'], r['ft_salary'], 'ft2'), axis=1
    )
    df['ft10_predicted_own'] = df.apply(
        lambda r: calibrate_ownership(r['projected_ownership'], r['ft_salary'], 'ft10'), axis=1
    )

    # Replace projected_ownership with calibrated FT2 prediction (what the app uses)
    df['dk_projected_own_raw'] = df['projected_ownership']  # preserve original
    df['projected_ownership']  = df['ft2_predicted_own']    # app reads this column

    # Captain model predictions
    print("  Running captain model...")
    try:
        db = GolfDFSDatabase(args.db)
        df_train = load_training_data(db)

        if len(df_train) < 20:
            print("  ⚠ Insufficient training data for captain model — skipping captain predictions")
            df['predicted_captain_own'] = None
        else:
            _, _, feat10 = fit_model(df_train, 'ft10_captain_own')
            coef10, r2_10, feat10 = fit_model(df_train, 'ft10_captain_own')

            # Prepare prediction dataframe
            df_pred = df.rename(columns={
                'projected_ownership':   'dk_projected_own',
                'total_points':          'dk_projected_pts',
                'std_dev':               'dk_projected_std',
            }).copy()
            df_pred['tournament'] = args.tournament or 'Prediction'
            # Restore actual projected_own for model features
            df_pred['dk_projected_own'] = df['dk_projected_own_raw']

            df['predicted_captain_own'] = predict_captain_own(df_pred, coef10, feat10)
            r2_str = f"R²={r2_10:.3f}"
            print(f"  Captain model applied ({r2_str}, trained on {len(df_train)} rows)")
        db.close()
    except Exception as e:
        print(f"  ⚠ Captain model error: {e} — skipping captain predictions")
        df['predicted_captain_own'] = None

    # Summary table
    print(f"\n{'='*75}")
    print(f"{'Player':<26} {'FT Sal':>7} {'DK Proj%':>9} {'FT2 Cal%':>9} {'FT10 Cal%':>10} {'Capt%':>7}")
    print("-"*70)

    display = df.dropna(subset=['ft2_predicted_own']).sort_values('ft2_predicted_own', ascending=False).head(20)
    for _, r in display.iterrows():
        dk_raw  = f"{r['dk_projected_own_raw']:.1%}" if pd.notna(r.get('dk_projected_own_raw')) else "  -  "
        ft2     = f"{r['ft2_predicted_own']:.1%}"    if pd.notna(r.get('ft2_predicted_own'))    else "  -  "
        ft10    = f"{r['ft10_predicted_own']:.1%}"   if pd.notna(r.get('ft10_predicted_own'))   else "  -  "
        capt    = f"{r['predicted_captain_own']:.1%}" if pd.notna(r.get('predicted_captain_own')) else "  -  "
        ft_sal  = f"{r['ft_salary']:.1f}M"           if pd.notna(r.get('ft_salary'))            else "  -  "
        print(f"{str(r['dk_name']):<26} {ft_sal:>7} {dk_raw:>9} {ft2:>9} {ft10:>10} {capt:>7}")

    # Save output
    df.to_csv(out_path, index=False)
    print(f"\n{'='*65}")
    print(f"✓ Calibrated file saved: {out_path}")
    print(f"\nLoad this file into the Streamlit app as your '72-Hole Projections'.")
    print(f"The 'projected_ownership' column now contains FanTeam-calibrated predictions.")
    print(f"Turn on 'Ownership Penalty Strength' in the sidebar to activate the leverage system.")
    if df['predicted_captain_own'].notna().any():
        print(f"\nTop predicted captains (£10 contest):")
        top_capts = df.dropna(subset=['predicted_captain_own']).nlargest(8, 'predicted_captain_own')
        for _, r in top_capts.iterrows():
            print(f"  {str(r['dk_name']):<26} {r['predicted_captain_own']:.1%} captain ownership predicted")


if __name__ == "__main__":
    main()
