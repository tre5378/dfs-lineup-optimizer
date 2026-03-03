"""
import_tournament.py
-------------------
Run this script each week to load new tournament data into golf_dfs.db.

Usage examples:
  python import_tournament.py --help
  python import_tournament.py dk-actual  --file dk_results.csv  --tournament "Genesis Invitational" --year 2026 --date 2026-02-16
  python import_tournament.py dk-proj    --file dk_proj.csv     --tournament "Honda Classic" --year 2026 --date 2026-02-27
  python import_tournament.py fanteam    --ft2 ft2.xlsx --ft10 ft10.xlsx --tournament "Genesis Invitational" --year 2026
  python import_tournament.py fanteam    --excel ownership.xlsx --sheet2 "£2" --sheet10 "£10" --tournament "Genesis Invitational" --year 2026
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from db import GolfDFSDatabase


def load_dk_actual(args, db):
    """Load DraftKings actual ownership CSV (DataGolf format)."""
    df = pd.read_csv(args.file)
    print(f"  Loaded {len(df)} rows from {args.file}")
    db.import_dk_actual(
        df,
        tournament_name=args.tournament,
        year=args.year,
        event_id=getattr(args, 'event_id', None),
        date=args.date,
        course=getattr(args, 'course', None),
    )


def load_dk_projected(args, db):
    """Load DraftKings projected ownership CSV (DataGolf projection format)."""
    df = pd.read_csv(args.file)
    print(f"  Loaded {len(df)} rows from {args.file}")
    db.import_dk_projected(
        df,
        tournament_name=args.tournament,
        year=args.year,
        event_id=getattr(args, 'event_id', None),
        date=args.date,
        course=getattr(args, 'course', None),
    )


def load_fanteam(args, db):
    """Load FanTeam ownership from Excel file with £2 and £10 sheets."""
    if hasattr(args, 'excel') and args.excel:
        sheet2  = getattr(args, 'sheet2',  '£2')
        sheet10 = getattr(args, 'sheet10', '£10')
        df_ft2  = pd.read_excel(args.excel, sheet_name=sheet2)
        df_ft10 = pd.read_excel(args.excel, sheet_name=sheet10)
        print(f"  Loaded £2={len(df_ft2)}, £10={len(df_ft10)} rows from {args.excel}")
    else:
        read = lambda f: pd.read_csv(f) if str(f).endswith('.csv') else pd.read_excel(f)
        df_ft2  = read(args.ft2)
        df_ft10 = read(args.ft10)
        print(f"  Loaded £2={len(df_ft2)}, £10={len(df_ft10)} rows")

    # Normalise column names
    for df in [df_ft2, df_ft10]:
        df.columns = [c.strip() for c in df.columns]

    db.import_fanteam(
        df_ft2, df_ft10,
        tournament_name=args.tournament,
        year=args.year,
    )


def main():
    parser = argparse.ArgumentParser(description="Import tournament data into golf_dfs.db")
    parser.add_argument("command", choices=["dk-actual", "dk-proj", "fanteam"],
                        help="Type of data to import")
    parser.add_argument("--db",         default="golf_dfs.db",  help="Path to database file")
    parser.add_argument("--tournament", required=True,           help="Tournament name")
    parser.add_argument("--year",       required=True, type=int, help="Year")
    parser.add_argument("--date",       default=None,            help="Tournament end date YYYY-MM-DD")
    parser.add_argument("--course",     default=None,            help="Course name")
    parser.add_argument("--event-id",   default=None, type=int,  help="DraftKings event ID")

    # dk-actual / dk-proj
    parser.add_argument("--file",   default=None, help="CSV file path (for dk-actual or dk-proj)")

    # fanteam (single Excel with two sheets)
    parser.add_argument("--excel",   default=None, help="Excel file with £2 and £10 sheets")
    parser.add_argument("--sheet2",  default="£2", help="Sheet name for £2 data (default: £2)")
    parser.add_argument("--sheet10", default="£10",help="Sheet name for £10 data (default: £10)")

    # fanteam (two separate Excel/CSV files)
    parser.add_argument("--ft2",  default=None, help="FanTeam £2 file")
    parser.add_argument("--ft10", default=None, help="FanTeam £10 file")

    args = parser.parse_args()

    db = GolfDFSDatabase(args.db)

    if args.command == "dk-actual":
        if not args.file:
            print("Error: --file required for dk-actual"); sys.exit(1)
        load_dk_actual(args, db)

    elif args.command == "dk-proj":
        if not args.file:
            print("Error: --file required for dk-proj"); sys.exit(1)
        load_dk_projected(args, db)

    elif args.command == "fanteam":
        if not (args.excel or (args.ft2 and args.ft10)):
            print("Error: provide --excel OR --ft2 + --ft10"); sys.exit(1)
        load_fanteam(args, db)

    db.close()
    print(f"\n✓ Done. Database saved: {args.db}")


if __name__ == "__main__":
    main()
