# Golf DFS Ownership Database

A SQLite-backed system for tracking DraftKings and FanTeam ownership patterns across PGA Tour DFS tournaments. Built for FanTeam £2 and £10 tournament analysis.

---

## Files

| File | Purpose |
|------|---------|
| `golf_dfs.db` | The database — **back this up after every tournament** |
| `db.py` | Core database engine (don't edit) |
| `import_tournament.py` | Weekly import script |
| `analyze.py` | Analysis and reporting |
| `seed_genesis.py` | One-time Genesis 2026 seed (already run) |

---

## Weekly Workflow

### Before the tournament (Tuesday)
```bash
# Import DK projections from DataGolf
python import_tournament.py dk-proj \
  --file draftkings_projections.csv \
  --tournament "Honda Classic" \
  --year 2026 \
  --date 2026-03-01

# Generate FanTeam predictions
python analyze.py predict \
  --file draftkings_projections.csv \
  --tournament "Honda Classic"
```

### After the tournament (Monday)
```bash
# Import DK actual ownership
python import_tournament.py dk-actual \
  --file draftkings_actual.csv \
  --tournament "Honda Classic" \
  --year 2026 \
  --date 2026-03-02

# Import FanTeam actual ownership (Excel with £2 and £10 sheets)
python import_tournament.py fanteam \
  --excel fanteam_ownership.xlsx \
  --sheet2 "£2" \
  --sheet10 "£10" \
  --tournament "Honda Classic" \
  --year 2026
```

---

## Analysis Commands

```bash
# Tournament overview — what data do we have?
python analyze.py summary

# Calibration: DK vs FanTeam by salary tier
python analyze.py calibration
python analyze.py calibration --tournament "Genesis Invitational"

# DK → FanTeam correlation per tournament
python analyze.py correlations

# Captain concentration per tournament
python analyze.py captains

# Full player history
python analyze.py player "Scottie Scheffler"
python analyze.py player "McIlroy"
```

---

## Input File Formats

### DK Actual (DataGolf CSV export)
Required columns: `player_name`, `salary`, `ownership`, `total_pts`, `fin_text`

```
player_name,salary,ownership,total_pts,fin_text,event_name,...
Scheffler, Scottie,14300,0.258,89.5,T12,The Genesis Invitational,...
```

### DK Projected (DataGolf projection CSV)
Required columns: `dk_name`, `dk_salary`, `projected_ownership`

```
dk_name,dk_salary,projected_ownership,total_points,...
Scottie Scheffler,14300,0.278,85.2,...
```

### FanTeam Excel
Two sheets in one Excel file (or two separate files).
Required columns: `Player`, `Price`, `Own %`, `С %` (captain), `Form`, `Score`

```
Player,Price,Own %,С %,Form,Score
Scottie Scheffler,28.6M,0.168,0.165,112.5,89.5
```

---

## Calibration Multipliers

Validated across Phoenix Open + Pebble Beach 2026 (2 tournaments, ~140 player-records):

| Salary Tier | £10 Multiplier | £2 Multiplier |
|-------------|---------------|--------------|
| $10,000+ | × 0.70 | × 0.72 |
| $9,000–$9,999 | × 0.90 | × 0.92 |
| $8,000–$8,999 | × 1.00 | × 1.00 |
| $7,000–$7,999 | × 1.08 | × 1.05 |
| Under $7,000 | × 1.10 | × 1.08 |

Apply to DK projected ownership to get FanTeam predicted ownership.

**Key insight:** DK systematically overprojects premium player ($10K+) ownership by ~7%.
FanTeam fields track DK actual closely (r = 0.89–0.95).

---

## Backing Up

After every tournament, copy `golf_dfs.db` to cloud storage:
```bash
cp golf_dfs.db ~/Dropbox/DFS/golf_dfs.db
# or
cp golf_dfs.db ~/Google\ Drive/DFS/golf_dfs.db
```

That single file is your entire database. Upload it when starting a new Claude session to restore all historical data.

---

## Current Database Contents

| Tournament | DK Proj | DK Actual | FT £2 | FT £10 |
|------------|---------|-----------|-------|--------|
| Sony Open 2026 | — | — | — | — |
| American Express 2026 | — | — | — | — |
| Farmers Insurance 2026 | — | — | — | — |
| Phoenix Open 2026 | ✓ | ✓ | ✓ | ✓ |
| Pebble Beach 2026 | ✓ | ✓ | ✓ | ✓ |
| Genesis Invitational 2026 | — | ✓ | ✓ | ✓ |

**To add historical data:** Re-upload the original source files and run the import commands above.
