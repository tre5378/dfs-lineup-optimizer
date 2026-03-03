"""
captain_model.py
----------------
Builds a captain ownership prediction model for FanTeam contests.

Uses tournaments where we have both DK projection inputs AND FanTeam
captain ownership data to learn the relationship, then applies it to
predict captain ownership for upcoming tournaments.

Usage:
  python captain_model.py                          # show model stats + all-tournament analysis
  python captain_model.py --predict dk_proj.csv    # predict for a new tournament
  python captain_model.py --tournament "Phoenix Open"  # analyse one tournament
"""

import argparse
import sqlite3
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from db import GolfDFSDatabase


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create model features from raw projection data.
    All features are normalised within each tournament so the model
    captures relative positioning rather than absolute values.
    """
    df = df.copy()

    # Within each tournament, rank players by projected pts and ownership
    for col in ['dk_projected_pts', 'dk_projected_own', 'dk_projected_std', 'ft_salary']:
        if col in df.columns:
            df[f'{col}_rank'] = df.groupby('tournament')[col].rank(pct=True, ascending=True)

    # Log-transform ownership (it's right-skewed)
    df['log_proj_own'] = np.log1p(df['dk_projected_own'] * 100)

    # Salary tier flags
    df['is_premium']  = (df['ft_salary'] >= 20).astype(int)   # £20M+
    df['is_high']     = ((df['ft_salary'] >= 18) & (df['ft_salary'] < 20)).astype(int)
    df['is_mid']      = ((df['ft_salary'] >= 16) & (df['ft_salary'] < 18)).astype(int)
    df['is_value']    = ((df['ft_salary'] >= 14) & (df['ft_salary'] < 16)).astype(int)
    df['is_cheap']    = (df['ft_salary'] < 14).astype(int)

    # Interaction: high projected pts + high salary = premium captain candidate
    if 'dk_projected_pts_rank' in df.columns:
        df['pts_x_salary'] = df['dk_projected_pts_rank'] * df['ft_salary_rank']

    return df


def load_training_data(db: GolfDFSDatabase) -> pd.DataFrame:
    """Load all rows with complete projection + captain ownership data."""
    df = db.get_ownership_data()
    
    required = ['dk_projected_pts', 'dk_projected_own', 'ft_salary', 'ft2_captain_own']
    df = df.dropna(subset=required)
    
    if df.empty:
        return df
    
    df = build_features(df)
    return df


# ---------------------------------------------------------------------------
# MODEL: Weighted linear regression on log-transformed captain ownership
# ---------------------------------------------------------------------------

def fit_model(df: pd.DataFrame, target: str = 'ft2_captain_own'):
    """
    Fit a simple linear model predicting captain ownership.
    Returns (coefficients dict, r_squared, feature_names).
    Uses numpy least squares — no sklearn dependency needed.
    """
    df = df.dropna(subset=[target]).copy()
    
    # Log-transform target (captain ownership is right-skewed)
    df['log_target'] = np.log1p(df[target] * 100)
    
    features = [
        'log_proj_own',
        'dk_projected_pts_rank',
        'ft_salary_rank',
        'dk_projected_std_rank',
        'is_premium',
        'is_high',
        'is_mid',
        'is_value',
    ]
    
    # Only use features that exist and have data
    features = [f for f in features if f in df.columns and df[f].notna().sum() > len(df) * 0.5]
    
    X = df[features].fillna(0).values
    X = np.column_stack([np.ones(len(X)), X])  # add intercept
    y = df['log_target'].values
    
    # Least squares fit
    coefs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    
    y_pred = X @ coefs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    feature_names = ['intercept'] + features
    coef_dict = dict(zip(feature_names, coefs))
    
    return coef_dict, r2, features


def predict_captain_own(df_pred: pd.DataFrame, coef_dict: dict, features: list) -> pd.Series:
    """Apply fitted model to new data. Returns predicted captain ownership (0-1)."""
    df_pred = build_features(df_pred)
    
    X = np.column_stack([
        np.ones(len(df_pred)),
        df_pred[[f for f in features if f in df_pred.columns]].fillna(0).values
    ])
    
    log_pred = X @ list(coef_dict.values())
    # Inverse of log1p(x*100): exp(log_pred) - 1 / 100
    pred = (np.expm1(log_pred)) / 100
    return np.clip(pred, 0, 1)


# ---------------------------------------------------------------------------
# CROSS-VALIDATION: Leave-one-tournament-out
# ---------------------------------------------------------------------------

def cross_validate(df: pd.DataFrame, target: str = 'ft2_captain_own'):
    """
    Leave-one-tournament-out cross-validation.
    Trains on N-1 tournaments, tests on the held-out one.
    Returns per-tournament performance metrics.
    """
    tournaments = df['tournament'].unique()
    if len(tournaments) < 2:
        print("Need at least 2 tournaments for cross-validation.")
        return None
    
    results = []
    for test_t in tournaments:
        train = df[df['tournament'] != test_t]
        test  = df[df['tournament'] == test_t].copy()
        
        if len(train) < 20 or len(test) < 5:
            continue
        
        coefs, r2_train, features = fit_model(train, target)
        
        # Predict on test tournament
        test_pred = df[df['tournament'] == test_t].copy()
        test_pred['pred_capt_own'] = predict_captain_own(test_pred, coefs, features)
        test_pred = test_pred.dropna(subset=[target, 'pred_capt_own'])
        
        if len(test_pred) < 5:
            continue
            
        # Correlation between predicted and actual
        r_test = test_pred[target].corr(test_pred['pred_capt_own'])
        
        # Top-5 accuracy: did the top-5 predicted captains include the top-5 actual?
        top5_actual = set(test_pred.nlargest(5, target)['player'])
        top5_pred   = set(test_pred.nlargest(5, 'pred_capt_own')['player'])
        top5_overlap = len(top5_actual & top5_pred)
        
        results.append({
            'held_out':     test_t,
            'n_test':       len(test_pred),
            'r_train':      round(r2_train, 3),
            'r_test':       round(r_test, 3),
            'top5_overlap': top5_overlap,
        })
    
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# DISPLAY HELPERS
# ---------------------------------------------------------------------------

def show_model_diagnostics(df, target='ft2_captain_own', label='£2'):
    coefs, r2, features = fit_model(df, target)
    
    print(f"\n{'='*65}")
    print(f"MODEL: {label} CAPTAIN OWNERSHIP  (R² = {r2:.3f}  n={len(df.dropna(subset=[target]))})")
    print(f"{'='*65}")
    print(f"\nFeature coefficients (log scale — positive = more captain ownership):")
    for name, val in coefs.items():
        if name != 'intercept':
            bar = '█' * int(abs(val) * 20)
            sign = '+' if val > 0 else '-'
            print(f"  {name:<30} {sign}{abs(val):.4f}  {bar}")
    
    return coefs, features


def show_predictions(df_pred, coef2, coef10, feat2, feat10, label='Predictions'):
    df_pred = df_pred.copy()
    df_pred['pred_ft2_capt']  = predict_captain_own(df_pred, coef2,  feat2)
    df_pred['pred_ft10_capt'] = predict_captain_own(df_pred, coef10, feat10)
    
    # Sort by £10 predicted captain own (highest first)
    out = df_pred.sort_values('pred_ft10_capt', ascending=False)
    
    print(f"\n{'='*80}")
    print(f"CAPTAIN OWNERSHIP PREDICTIONS: {label}")
    print(f"{'='*80}")
    print(f"\n{'Player':<26} {'FT Sal':>7} {'DK Proj%':>9} {'Proj Pts':>9} │ {'£2 Capt':>8} {'£10 Capt':>9}")
    print("-"*72)
    
    for _, r in out.head(25).iterrows():
        proj_own = f"{r['dk_projected_own']:.1%}" if pd.notna(r.get('dk_projected_own')) else "  -  "
        proj_pts = f"{r['dk_projected_pts']:.1f}"  if pd.notna(r.get('dk_projected_pts')) else "  -  "
        ft_sal   = f"{r['ft_salary']:.1f}M"         if pd.notna(r.get('ft_salary'))        else "  -  "
        print(f"{r['player']:<26} {ft_sal:>7} {proj_own:>9} {proj_pts:>9} │ "
              f"{r['pred_ft2_capt']:>8.1%} {r['pred_ft10_capt']:>9.1%}")
    
    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FanTeam Captain Ownership Prediction Model")
    parser.add_argument("--db",         default="golf_dfs.db")
    parser.add_argument("--predict",    default=None,  help="DK projection CSV for new tournament")
    parser.add_argument("--tournament", default=None,  help="Filter analysis to one tournament")
    parser.add_argument("--ft-salary",  default=None,  help="FanTeam salary CSV (for prediction enrichment)")
    args = parser.parse_args()

    db = GolfDFSDatabase(args.db)
    df = load_training_data(db)
    
    if df.empty:
        print("No training data available. Need tournaments with DK projections + FanTeam captain ownership.")
        db.close()
        return

    tournaments = df['tournament'].unique()
    print(f"\nTraining data: {len(df)} player-tournament rows across {len(tournaments)} tournaments")
    print(f"Tournaments: {', '.join(tournaments)}")

    # Fit models
    coef2,  r2_2,  feat2  = fit_model(df, 'ft2_captain_own')
    coef10, r2_10, feat10 = fit_model(df, 'ft10_captain_own')

    # Show diagnostics
    show_model_diagnostics(df, 'ft2_captain_own',  '£2')
    show_model_diagnostics(df, 'ft10_captain_own', '£10')

    # Cross-validation
    print(f"\n{'='*65}")
    print("CROSS-VALIDATION (leave-one-tournament-out):")
    print(f"{'='*65}")
    
    cv2  = cross_validate(df, 'ft2_captain_own')
    cv10 = cross_validate(df, 'ft10_captain_own')
    
    if cv2 is not None:
        print(f"\n{'Tournament held out':<30} {'n':>4} {'£2 r':>7} {'£10 r':>7} {'Top5 overlap':>13}")
        print("-"*60)
        for i, r in cv2.iterrows():
            r10 = cv10.iloc[i]['r_test'] if cv10 is not None and i < len(cv10) else '-'
            t5  = cv10.iloc[i]['top5_overlap'] if cv10 is not None and i < len(cv10) else '-'
            print(f"{r['held_out']:<30} {r['n_test']:>4} {r['r_test']:>7.3f} "
                  f"{r10:>7.3f} {t5:>13}")

    # In-sample: show top captain predictions vs actual for each training tournament
    df['pred_ft2_capt']  = predict_captain_own(df, coef2,  feat2)
    df['pred_ft10_capt'] = predict_captain_own(df, coef10, feat10)

    print(f"\n{'='*65}")
    print("IN-SAMPLE: TOP-5 PREDICTED vs ACTUAL CAPTAINS BY TOURNAMENT")
    print(f"{'='*65}")

    for t in sorted(df['tournament'].unique()):
        g = df[df['tournament'] == t].copy()
        top5_actual = g.nlargest(5, 'ft10_captain_own')[['player', 'ft10_captain_own']].values
        top5_pred   = g.nlargest(5, 'pred_ft10_capt')[['player', 'pred_ft10_capt']].values
        overlap     = set(g.nlargest(5, 'ft10_captain_own')['player']) & \
                      set(g.nlargest(5, 'pred_ft10_capt')['player'])
        
        print(f"\n{t} (overlap: {len(overlap)}/5)")
        print(f"  {'Actual top captains':<35} {'Predicted top captains'}")
        print(f"  {'-'*70}")
        for j in range(5):
            a_name = f"{top5_actual[j][0]} ({top5_actual[j][1]:.1%})" if j < len(top5_actual) else ""
            p_name = f"{top5_pred[j][0]} ({top5_pred[j][1]:.1%})"     if j < len(top5_pred)   else ""
            match  = "✓" if top5_actual[j][0] in overlap and j < len(top5_actual) else " "
            print(f"  {match} {a_name:<35} {p_name}")

    # Predict for new tournament if file provided
    if args.predict:
        print(f"\n\nLoading projection file: {args.predict}")
        df_new = pd.read_csv(args.predict)
        
        # Standardise columns
        if 'dk_name' in df_new.columns:
            df_new = df_new.rename(columns={'dk_name': 'player'})
        if 'player_name' in df_new.columns:
            df_new = df_new.rename(columns={'player_name': 'player'})
        
        df_new['dk_projected_pts'] = df_new.get('total_points', df_new.get('total_pts'))
        df_new['dk_projected_own'] = df_new['projected_ownership'].apply(
            lambda x: float(str(x).rstrip('%')) / 100 if float(str(x).rstrip('%')) > 1 
            else float(str(x).rstrip('%'))
        )
        df_new['dk_projected_std'] = df_new.get('std_dev')
        df_new['tournament'] = args.predict  # dummy tournament name for rank calculation
        
        # Try to merge FanTeam salary if provided
        if args.ft_salary:
            df_ft = pd.read_csv(args.ft_salary)
            df_ft['ft_salary'] = df_ft['Price'].str.replace('M','').astype(float)
            df_ft['ft_last'] = df_ft['Player'].str.split().str[-1].str.lower()
            df_new['ft_last'] = df_new['player'].str.split().str[-1].str.lower()
            df_new = df_new.merge(df_ft[['ft_last','ft_salary']], on='ft_last', how='left')
        else:
            # Estimate FanTeam salary as 2x DK (rough proxy)
            df_new['ft_salary'] = df_new['dk_salary'] / 500
            print("  (No FanTeam salary file provided — estimating ft_salary as 2x DK)")

        # Filter placeholder projections
        df_new = df_new[df_new['dk_projected_own'] < 0.60]

        show_predictions(df_new, coef2, coef10, feat2, feat10,
                        label=args.predict.split('/')[-1].replace('.csv',''))

    db.close()


if __name__ == "__main__":
    main()
