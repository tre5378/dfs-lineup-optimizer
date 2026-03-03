"""
fix_projected.py
----------------
One-time fix: corrects dk_projected_own values in the database.
  1. Divides values stored as percentages (e.g. 40.57) back to decimals (0.4057)
  2. Nulls out DataGolf placeholder values (>60%) which are not real projections

Run once: python fix_projected.py
Safe to run multiple times.
"""

import sqlite3, sys

db_path = "golf_dfs.db"
conn = sqlite3.connect(db_path)

# Step 1: fix values stored as percentages instead of decimals
before = conn.execute(
    "SELECT COUNT(*) FROM ownership WHERE dk_projected_own > 1.0"
).fetchone()[0]

if before > 0:
    print(f"Found {before} projected values stored as % not decimal — fixing...")
    conn.execute(
        "UPDATE ownership SET dk_projected_own = dk_projected_own / 100.0 "
        "WHERE dk_projected_own > 1.0"
    )
    conn.commit()
    print("  Done.")
else:
    print("Step 1: No percentage-format values found — skipping.")

# Step 2: null out DataGolf placeholder values (>60%)
bad = conn.execute(
    "SELECT COUNT(*) FROM ownership WHERE dk_projected_own > 0.60"
).fetchone()[0]

if bad > 0:
    print(f"Found {bad} placeholder projected values (>60%) — nulling out...")
    conn.execute("UPDATE ownership SET dk_projected_own = NULL WHERE dk_projected_own > 0.60")
    conn.commit()
    print("  Done.")
else:
    print("Step 2: No placeholder values found — skipping.")

# Verify with Scheffler
rows = conn.execute("""
    SELECT p.dk_name, t.name, o.dk_projected_own, o.dk_actual_own
    FROM ownership o
    JOIN players p ON p.id = o.player_id
    JOIN tournaments t ON t.id = o.tournament_id
    WHERE p.dk_name = 'Scottie Scheffler' AND o.dk_projected_own IS NOT NULL
    ORDER BY t.date
""").fetchall()

if rows:
    print("\nVerification - Scheffler projected vs actual:")
    for r in rows:
        print(f"  {r[1]:<25}  projected={r[2]:.3f}  actual={r[3]:.3f}")

conn.close()
print(f"\n✓ Done. Database: {db_path}")
