"""
fix_name_conflicts.py
---------------------
Resolves known FanTeam name ambiguities by registering explicit
mappings in the name_mappings table, then flags any remaining
suspicious salary ratios for manual review.

Run ONCE after initial data load, before re-importing FanTeam data.
Usage: python fix_name_conflicts.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from db import GolfDFSDatabase

db = GolfDFSDatabase("golf_dfs.db")

print("\nRegistering name mappings for known conflicts...")
print("="*60)

# Format: (FanTeam name as it appears, correct DK name)
# Add to this list any time a new conflict is discovered.
KNOWN_MAPPINGS = [
    # Griffin conflict: Ben Griffin vs Lanto Griffin
    ("Ben Griffin",     "Ben Griffin"),
    ("Lanto Griffin",   "Lanto Griffin"),

    # Hojgaard conflict
    ("Nicolai Hojgaard", "Nicolai Hojgaard"),
    ("Rasmus Hojgaard",  "Rasmus Hojgaard"),

    # Kim conflict (5 players)
    ("Tom Kim",         "Tom Kim"),
    ("Chan Kim",        "Chan Kim"),
    ("Michael Kim",     "Michael Kim"),
    ("S.H. Kim",        "S.H. Kim"),
    ("Si Woo Kim",      "Si Woo Kim"),

    # Brown conflict
    ("Blades Brown",    "Blades Brown"),
    ("Dan Brown",       "Dan Brown"),

    # Young conflict
    ("Cameron Young",   "Cameron Young"),
    ("Carson Young",    "Carson Young"),

    # Svensson conflict
    ("Adam Svensson",   "Adam Svensson"),
    ("Jesper Svensson", "Jesper Svensson"),

    # Lee conflict
    ("Min Woo Lee",     "Min Woo Lee"),
    ("S.T. Lee",        "S.T. Lee"),

    # Davis conflict
    ("Cam Davis",       "Cam Davis"),

    # McCarty conflict (same person, different name formats)
    ("Matt McCarty",    "Matt McCarty"),
]

for ft_name, dk_name in KNOWN_MAPPINGS:
    # Only register if DK player exists in DB
    row = db.conn.execute(
        "SELECT id FROM players WHERE dk_name=?", (dk_name,)
    ).fetchone()
    if row:
        db.add_name_mapping(ft_name, dk_name)
    else:
        print(f"  SKIP (not in DB yet): '{dk_name}'")

print("\n\nChecking for suspicious salary ratios (potential remaining mismatches)...")
print("="*60)
suspicious = db.audit_salary_ratios(low=1.7, high=2.3)
if suspicious.empty:
    print("✓ No suspicious ratios found — all data looks clean.")
else:
    print(f"Found {len(suspicious)} suspicious records:")
    print(suspicious[['player','tournament','dk_salary','ft_salary','ratio']].to_string(index=False))
    print("\nFor each of these, verify the player is correctly matched.")
    print("If wrong, add a mapping to KNOWN_MAPPINGS and re-run.")

print("\n\nName conflict risk zones (players sharing a last name):")
print("="*60)
conflicts = db.audit_name_conflicts()
print(conflicts.to_string(index=False))

db.close()
print("\n✓ Done. Re-run import_all.bat to reimport FanTeam data with correct matching.")
