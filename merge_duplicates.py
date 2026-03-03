"""
merge_duplicates.py
-------------------
Merges duplicate player records (same person, different name formats)
and registers canonical name mappings for future imports.

Run once: python merge_duplicates.py
"""

import sqlite3, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from db import GolfDFSDatabase

db = GolfDFSDatabase("golf_dfs.db")
conn = db.conn

def merge_players(keep_name: str, remove_name: str):
    """
    Merge remove_name into keep_name.
    All ownership records pointing to remove_name are reassigned to keep_name.
    Both name variants are registered as mappings to keep_name.
    """
    keep = conn.execute("SELECT id FROM players WHERE dk_name=?", (keep_name,)).fetchone()
    remove = conn.execute("SELECT id FROM players WHERE dk_name=?", (remove_name,)).fetchone()

    if not keep:
        print(f"  SKIP: '{keep_name}' not found"); return
    if not remove:
        print(f"  SKIP: '{remove_name}' not found (may already be merged)"); return

    keep_id, remove_id = keep["id"], remove["id"]

    # Reassign ownership records — but avoid unique constraint violations
    # (can't have two records for same tournament+player)
    ownership_conflicts = conn.execute("""
        SELECT o1.tournament_id FROM ownership o1
        JOIN ownership o2 ON o2.tournament_id = o1.tournament_id
        WHERE o1.player_id = ? AND o2.player_id = ?
    """, (remove_id, keep_id)).fetchall()

    conflict_tournaments = {r[0] for r in ownership_conflicts}

    if conflict_tournaments:
        # Delete the duplicate records (keep_id already has data for these tournaments)
        conn.execute(
            f"DELETE FROM ownership WHERE player_id=? AND tournament_id IN ({','.join('?'*len(conflict_tournaments))})",
            [remove_id] + list(conflict_tournaments)
        )

    # Reassign remaining ownership records
    conn.execute("UPDATE ownership SET player_id=? WHERE player_id=?", (keep_id, remove_id))

    # Remove the duplicate player record
    conn.execute("DELETE FROM players WHERE id=?", (remove_id,))
    conn.commit()

    # Register both name variants as mappings
    db.add_name_mapping(keep_name, keep_name)
    db.add_name_mapping(remove_name, keep_name)

    print(f"  ✓ Merged '{remove_name}' → '{keep_name}'")


print("\nMerging duplicate player records...")
print("="*60)

# Same person, different name formats
# Format: (name to KEEP, name to REMOVE/alias)
DUPLICATES = [
    ("Zach Bauchou",                "Zachary Bauchou"),
    ("Cam Davis",                   "Cameron Davis"),
    ("Adrien Dumont De Chassart",   "Adrien Dumont de Chassart"),
    ("John Keefer",                 "Johnny Keefer"),
    ("S.T. Lee",                    "Seung-taek Lee"),
    ("Haotong Li",                  "Hao-Tong Li"),
    ("Matt McCarty",                "Matthew McCarty"),
    ("Erik Van Rooyen",             "Erik van Rooyen"),
    # Same person, Echavarria (likely same player different format)
    ("Nico Echavarria",             "Nicolas Echavarria"),
]

for keep, remove in DUPLICATES:
    merge_players(keep, remove)

print("\n\nRegistering mappings for genuine conflicts (different players, same last name)...")
print("="*60)

GENUINE_CONFLICTS = [
    ("Ben Griffin",      "Ben Griffin"),
    ("Lanto Griffin",    "Lanto Griffin"),
    ("Nicolai Hojgaard", "Nicolai Hojgaard"),
    ("Rasmus Hojgaard",  "Rasmus Hojgaard"),
    ("Tom Kim",          "Tom Kim"),
    ("Chan Kim",         "Chan Kim"),
    ("Michael Kim",      "Michael Kim"),
    ("S.H. Kim",         "S.H. Kim"),
    ("Si Woo Kim",       "Si Woo Kim"),
    ("Blades Brown",     "Blades Brown"),
    ("Dan Brown",        "Dan Brown"),
    ("Cameron Young",    "Cameron Young"),
    ("Carson Young",     "Carson Young"),
    ("Adam Svensson",    "Adam Svensson"),
    ("Jesper Svensson",  "Jesper Svensson"),
    ("Min Woo Lee",      "Min Woo Lee"),
]

for ft_name, dk_name in GENUINE_CONFLICTS:
    row = conn.execute("SELECT id FROM players WHERE dk_name=?", (dk_name,)).fetchone()
    if row:
        db.add_name_mapping(ft_name, dk_name)
    else:
        print(f"  SKIP (not in DB): '{dk_name}'")

print("\n\nFinal name conflict check:")
print("="*60)
conflicts = db.audit_name_conflicts()
print(conflicts.to_string(index=False))

db.close()
print("\n✓ Done. Now re-run the FanTeam imports.")
