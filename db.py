"""
Golf DFS Ownership Database
SQLite-backed system for tracking DK and FanTeam ownership across tournaments.

Usage:
    from db import GolfDFSDatabase
    db = GolfDFSDatabase("golf_dfs.db")
"""

import sqlite3
import pandas as pd
import numpy as np
import re


class GolfDFSDatabase:
    def __init__(self, db_path: str = "golf_dfs.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()
        self._migrate_schema()
        print(f"✓ Connected to database: {db_path}")

    def _create_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS tournaments (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                event_id    INTEGER,
                date        TEXT,
                course      TEXT,
                year        INTEGER,
                UNIQUE(name, year)
            );

            CREATE TABLE IF NOT EXISTS players (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                dk_name     TEXT NOT NULL UNIQUE,
                alt_name    TEXT
            );

            -- Manual name mappings: FanTeam name -> DK player id
            -- Used to resolve ambiguous or mismatched names
            CREATE TABLE IF NOT EXISTS name_mappings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ft_name     TEXT NOT NULL UNIQUE,
                player_id   INTEGER NOT NULL,
                FOREIGN KEY (player_id) REFERENCES players(id)
            );

            CREATE TABLE IF NOT EXISTS ownership (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                tournament_id       INTEGER NOT NULL,
                player_id           INTEGER NOT NULL,
                dk_salary           INTEGER,
                ft_salary           REAL,
                dk_projected_own    REAL,
                dk_projected_pts    REAL,
                dk_projected_std    REAL,
                dk_actual_own       REAL,
                ft2_own             REAL,
                ft2_captain_own     REAL,
                ft10_own            REAL,
                ft10_captain_own    REAL,
                total_pts           REAL,
                finish              TEXT,
                form_score          REAL,
                FOREIGN KEY (tournament_id) REFERENCES tournaments(id),
                FOREIGN KEY (player_id)     REFERENCES players(id),
                UNIQUE(tournament_id, player_id)
            );

            CREATE TABLE IF NOT EXISTS calibration_snapshots (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                tournament_id       INTEGER NOT NULL,
                salary_tier         TEXT,
                n_players           INTEGER,
                dk_proj_avg         REAL,
                dk_actual_avg       REAL,
                ft2_avg             REAL,
                ft10_avg            REAL,
                proj_to_actual_diff REAL,
                dk_to_ft2_diff      REAL,
                dk_to_ft10_diff     REAL,
                FOREIGN KEY (tournament_id) REFERENCES tournaments(id)
            );
        """)
        self.conn.commit()

    def _migrate_schema(self):
        """Add columns and tables to existing databases."""
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(ownership)").fetchall()]
        if "ft_salary" not in cols:
            self.conn.execute("ALTER TABLE ownership ADD COLUMN ft_salary REAL")
            self.conn.commit()
        if "dk_projected_pts" not in cols:
            self.conn.execute("ALTER TABLE ownership ADD COLUMN dk_projected_pts REAL")
            self.conn.execute("ALTER TABLE ownership ADD COLUMN dk_projected_std REAL")
            self.conn.commit()
        # name_mappings table handled by CREATE TABLE IF NOT EXISTS above

    # ------------------------------------------------------------------
    # UPSERT HELPERS
    # ------------------------------------------------------------------

    def upsert_tournament(self, name, year, event_id=None, date=None, course=None) -> int:
        self.conn.execute(
            "INSERT INTO tournaments (name, year, event_id, date, course) VALUES (?,?,?,?,?) "
            "ON CONFLICT(name, year) DO UPDATE SET event_id=excluded.event_id, "
            "date=excluded.date, course=excluded.course",
            (name, year, event_id, date, course)
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id FROM tournaments WHERE name=? AND year=?", (name, year)
        ).fetchone()
        return row["id"]

    def upsert_player(self, dk_name, alt_name=None) -> int:
        self.conn.execute(
            "INSERT INTO players (dk_name, alt_name) VALUES (?,?) "
            "ON CONFLICT(dk_name) DO UPDATE SET alt_name=COALESCE(excluded.alt_name, alt_name)",
            (dk_name, alt_name)
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id FROM players WHERE dk_name=?", (dk_name,)
        ).fetchone()
        return row["id"]

    def add_name_mapping(self, ft_name: str, dk_name: str):
        """
        Manually map a FanTeam player name to a DK player name.
        Use this to resolve ambiguous or mismatched names.
        Example: db.add_name_mapping("Ben Griffin", "Ben Griffin")  # ensure correct match
        """
        row = self.conn.execute(
            "SELECT id FROM players WHERE dk_name=?", (dk_name,)
        ).fetchone()
        if not row:
            print(f"  Warning: DK player '{dk_name}' not found in players table.")
            return
        self.conn.execute(
            "INSERT INTO name_mappings (ft_name, player_id) VALUES (?,?) "
            "ON CONFLICT(ft_name) DO UPDATE SET player_id=excluded.player_id",
            (ft_name, row["id"])
        )
        self.conn.commit()
        print(f"  ✓ Mapped '{ft_name}' → '{dk_name}'")

    def upsert_ownership(self, tournament_id, player_id, **kwargs):
        allowed = {
            "dk_salary", "ft_salary", "dk_projected_own", "dk_projected_pts", "dk_projected_std", "dk_actual_own",
            "ft2_own", "ft2_captain_own", "ft10_own", "ft10_captain_own",
            "total_pts", "finish", "form_score"
        }
        data = {k: v for k, v in kwargs.items() if k in allowed and v is not None}

        existing = self.conn.execute(
            "SELECT id FROM ownership WHERE tournament_id=? AND player_id=?",
            (tournament_id, player_id)
        ).fetchone()

        if existing:
            if data:
                set_clause = ", ".join(f"{k}=?" for k in data)
                self.conn.execute(
                    f"UPDATE ownership SET {set_clause} WHERE tournament_id=? AND player_id=?",
                    list(data.values()) + [tournament_id, player_id]
                )
        else:
            cols = ["tournament_id", "player_id"] + list(data.keys())
            vals = [tournament_id, player_id] + list(data.values())
            placeholders = ",".join("?" for _ in vals)
            self.conn.execute(
                f"INSERT INTO ownership ({','.join(cols)}) VALUES ({placeholders})", vals
            )
        self.conn.commit()

    # ------------------------------------------------------------------
    # NORMALISATION HELPERS
    # ------------------------------------------------------------------

    def _normalise_pct(self, val) -> float:
        """Convert '9.68%' or '1.29%' or 0.0968 to a consistent 0-1 float."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        if isinstance(val, str):
            val = val.strip()
            is_pct_string = val.endswith('%')
            val = val.rstrip('%')
            try:
                val = float(val)
            except ValueError:
                return None
            if is_pct_string:
                val = val / 100
            return float(val)
        if val > 1.5:
            val = val / 100
        return float(val)

    def _parse_ft_salary(self, val) -> float:
        """Convert FanTeam price string to numeric. '29.4M' -> 29.4"""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        val = str(val).strip().upper().replace('M', '').replace('£', '')
        try:
            return float(val)
        except ValueError:
            return None

    def _clean_dk_name(self, name: str) -> str:
        """Convert 'Last, First' to 'First Last'."""
        parts = str(name).split(", ")
        return f"{parts[1]} {parts[0]}" if len(parts) == 2 else name

    @staticmethod
    def _normalise_name(name: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace for fuzzy comparison."""
        name = name.lower().strip()
        name = re.sub(r"[^a-z\s]", "", name)
        name = re.sub(r"\s+", " ", name)
        return name

    # ------------------------------------------------------------------
    # PLAYER MATCHING
    # ------------------------------------------------------------------

    def _match_player(self, ft_name: str, unmatched_log: list) -> int | None:
        """
        Match a FanTeam player name to a player_id using a three-tier strategy:

        1. Check name_mappings table (manual overrides — highest priority)
        2. Full normalised name match against dk_name
        3. Last name match — ONLY if exactly one player has that last name

        If last name is ambiguous (multiple players), logs a warning and returns None.
        Never guesses when there is ambiguity.
        """
        ft_name = ft_name.strip()
        norm_ft = self._normalise_name(ft_name)

        # Tier 1: manual mapping
        row = self.conn.execute(
            "SELECT player_id FROM name_mappings WHERE LOWER(ft_name)=?",
            (ft_name.lower(),)
        ).fetchone()
        if row:
            return row["player_id"]

        # Tier 2: full name match (normalised)
        all_players = self.conn.execute("SELECT id, dk_name FROM players").fetchall()
        for p in all_players:
            if self._normalise_name(p["dk_name"]) == norm_ft:
                return p["id"]

        # Tier 3: last name only — safe only if unique
        last = norm_ft.split()[-1]
        matches = [
            p for p in all_players
            if self._normalise_name(p["dk_name"]).split()[-1] == last
        ]

        if len(matches) == 1:
            return matches[0]["id"]
        elif len(matches) > 1:
            # Ambiguous — log and skip rather than guess
            names = [m["dk_name"] for m in matches]
            unmatched_log.append(
                f"  ⚠ AMBIGUOUS: FanTeam '{ft_name}' matches multiple players: {names}"
            )
            return None
        else:
            # No match found — insert as new player
            unmatched_log.append(
                f"  ℹ NEW PLAYER: '{ft_name}' not found in DB, inserted as new player"
            )
            return self.upsert_player(ft_name)

    def audit_name_conflicts(self) -> pd.DataFrame:
        """
        Return all players sharing a last name — these are the mismatch risk zones.
        """
        sql = """
        SELECT 
            LOWER(SUBSTR(dk_name, INSTR(dk_name, ' ')+1)) AS last_name,
            GROUP_CONCAT(dk_name, ' | ') AS players,
            COUNT(*) AS n
        FROM players
        GROUP BY last_name
        HAVING n > 1
        ORDER BY last_name
        """
        return pd.read_sql_query(sql, self.conn)

    def audit_salary_ratios(self, low: float = 1.7, high: float = 2.3) -> pd.DataFrame:
        """
        Flag records where FT/DK salary ratio is outside expected range.
        Ratios outside 1.7-2.3 likely indicate a name mismatch.
        """
        sql = f"""
        SELECT
            p.dk_name AS player,
            t.name AS tournament,
            o.dk_salary,
            o.ft_salary,
            ROUND(o.ft_salary / (o.dk_salary / 1000.0), 3) AS ratio,
            o.dk_actual_own,
            o.ft2_own
        FROM ownership o
        JOIN players p ON p.id = o.player_id
        JOIN tournaments t ON t.id = o.tournament_id
        WHERE o.dk_salary IS NOT NULL AND o.ft_salary IS NOT NULL
        AND (o.ft_salary / (o.dk_salary / 1000.0) < {low}
             OR o.ft_salary / (o.dk_salary / 1000.0) > {high})
        ORDER BY ratio
        """
        return pd.read_sql_query(sql, self.conn)

    # ------------------------------------------------------------------
    # IMPORT FROM DATAFRAMES
    # ------------------------------------------------------------------

    def import_dk_actual(self, df: pd.DataFrame, tournament_name: str, year: int,
                         event_id: int = None, date: str = None, course: str = None):
        t_id = self.upsert_tournament(tournament_name, year, event_id, date, course)
        count = 0
        for _, row in df.iterrows():
            name = self._clean_dk_name(row.get("player_name", row.get("Player", "")))
            p_id = self.upsert_player(name)
            self.upsert_ownership(
                t_id, p_id,
                dk_salary=int(row["salary"]) if pd.notna(row.get("salary")) else None,
                dk_actual_own=float(row["ownership"]) if pd.notna(row.get("ownership")) else None,
                total_pts=float(row["total_pts"]) if pd.notna(row.get("total_pts")) else None,
                finish=str(row["fin_text"]) if pd.notna(row.get("fin_text")) else None,
            )
            count += 1
        print(f"  ✓ DK Actual imported: {count} players → {tournament_name} {year}")
        return t_id

    def import_dk_projected(self, df: pd.DataFrame, tournament_name: str, year: int,
                            event_id: int = None, date: str = None, course: str = None):
        t_id = self.upsert_tournament(tournament_name, year, event_id, date, course)
        count = 0
        for _, row in df.iterrows():
            name = row.get("dk_name", row.get("player_name", ""))
            if ", " in str(name):
                name = self._clean_dk_name(name)
            p_id = self.upsert_player(name)
            proj = self._normalise_pct(row.get("projected_ownership"))
            if proj is not None and proj > 0.60:
                proj = None
            pts = float(row["total_points"]) if pd.notna(row.get("total_points")) else None
            std = float(row["std_dev"]) if pd.notna(row.get("std_dev")) else None
            self.upsert_ownership(
                t_id, p_id,
                dk_salary=int(row["dk_salary"]) if pd.notna(row.get("dk_salary")) else None,
                dk_projected_own=proj,
                dk_projected_pts=pts,
                dk_projected_std=std,
            )
            count += 1
        print(f"  ✓ DK Projected imported: {count} players → {tournament_name} {year}")
        return t_id

    def import_fanteam(self, df_ft2: pd.DataFrame, df_ft10: pd.DataFrame,
                       tournament_name: str, year: int):
        """
        Import FanTeam actual ownership for both £2 and £10 contests.
        Uses robust three-tier name matching. Ambiguous last names are
        flagged and skipped rather than guessed.
        """
        t_id = self.upsert_tournament(tournament_name, year)
        unmatched_log = []

        count2 = count10 = skipped2 = skipped10 = 0

        for _, row in df_ft2.iterrows():
            p_id = self._match_player(str(row["Player"]), unmatched_log)
            if p_id is None:
                skipped2 += 1
                continue
            own    = self._normalise_pct(row.get("Own %"))
            cap    = self._normalise_pct(row.get("С %"))
            form   = float(row["Form"]) if pd.notna(row.get("Form")) else None
            ft_sal = self._parse_ft_salary(row.get("Price"))
            self.upsert_ownership(t_id, p_id, ft_salary=ft_sal,
                                  ft2_own=own, ft2_captain_own=cap, form_score=form)
            count2 += 1

        for _, row in df_ft10.iterrows():
            p_id = self._match_player(str(row["Player"]), unmatched_log)
            if p_id is None:
                skipped10 += 1
                continue
            own = self._normalise_pct(row.get("Own %"))
            cap = self._normalise_pct(row.get("С %"))
            self.upsert_ownership(t_id, p_id, ft10_own=own, ft10_captain_own=cap)
            count10 += 1

        print(f"  ✓ FanTeam imported: £2={count2} players, £10={count10} players → {tournament_name} {year}")
        if skipped2 or skipped10:
            print(f"  ⚠ Skipped (ambiguous names): £2={skipped2}, £10={skipped10}")
        if unmatched_log:
            print("  Match log:")
            for msg in unmatched_log:
                print(f"    {msg}")
        return t_id

    # ------------------------------------------------------------------
    # ANALYSIS QUERIES
    # ------------------------------------------------------------------

    def get_tournament_summary(self) -> pd.DataFrame:
        sql = """
        SELECT
            t.name, t.year, t.date,
            COUNT(o.id)                                                      AS n_players,
            SUM(CASE WHEN o.dk_projected_own IS NOT NULL THEN 1 ELSE 0 END) AS has_dk_proj,
            SUM(CASE WHEN o.dk_projected_pts IS NOT NULL THEN 1 ELSE 0 END) AS has_dk_pts,
            SUM(CASE WHEN o.dk_actual_own    IS NOT NULL THEN 1 ELSE 0 END) AS has_dk_actual,
            SUM(CASE WHEN o.ft2_own          IS NOT NULL THEN 1 ELSE 0 END) AS has_ft2,
            SUM(CASE WHEN o.ft10_own         IS NOT NULL THEN 1 ELSE 0 END) AS has_ft10,
            SUM(CASE WHEN o.ft_salary        IS NOT NULL THEN 1 ELSE 0 END) AS has_ft_salary
        FROM tournaments t
        LEFT JOIN ownership o ON o.tournament_id = t.id
        GROUP BY t.id
        ORDER BY t.date
        """
        return pd.read_sql_query(sql, self.conn)

    def get_ownership_data(self, tournament_name: str = None, year: int = None) -> pd.DataFrame:
        where, params = [], []
        if tournament_name:
            where.append("t.name = ?"); params.append(tournament_name)
        if year:
            where.append("t.year = ?"); params.append(year)
        where_clause = "WHERE " + " AND ".join(where) if where else ""
        sql = f"""
        SELECT
            t.name AS tournament, t.year, t.date,
            p.dk_name AS player,
            o.dk_salary, o.ft_salary,
            o.dk_projected_own, o.dk_projected_pts, o.dk_projected_std, o.dk_actual_own,
            o.ft2_own, o.ft2_captain_own,
            o.ft10_own, o.ft10_captain_own,
            o.total_pts, o.finish, o.form_score
        FROM ownership o
        JOIN tournaments t ON t.id = o.tournament_id
        JOIN players p ON p.id = o.player_id
        {where_clause}
        ORDER BY t.date, o.dk_actual_own DESC
        """
        return pd.read_sql_query(sql, self.conn, params=params)

    def _ft_tier(self, ft_salary) -> str:
        if pd.isna(ft_salary):    return "Unknown"
        if ft_salary >= 20:       return "1_£20M+"
        elif ft_salary >= 18:     return "2_£18-20M"
        elif ft_salary >= 16:     return "3_£16-18M"
        elif ft_salary >= 14:     return "4_£14-16M"
        else:                     return "5_<£14M"

    def _dk_tier(self, dk_salary) -> str:
        if pd.isna(dk_salary):      return "Unknown"
        if dk_salary >= 10000:      return "1_$10K+"
        elif dk_salary >= 9000:     return "2_$9K-10K"
        elif dk_salary >= 8000:     return "3_$8K-9K"
        elif dk_salary >= 7000:     return "4_$7K-8K"
        else:                       return "5_<$7K"

    def calibration_analysis(self, tournament_name: str = None,
                             use_ft_salary: bool = False) -> pd.DataFrame:
        df = self.get_ownership_data(tournament_name)
        df = df[df["dk_actual_own"].notna() & (df["ft2_own"].notna() | df["ft10_own"].notna())]
        if use_ft_salary:
            df["tier"] = df["ft_salary"].apply(self._ft_tier)
        else:
            df["tier"] = df["dk_salary"].apply(self._dk_tier)
        rows = []
        for tier_key, g in df.groupby("tier"):
            label = tier_key[2:]
            row = {"tier": label, "n": len(g), "dk_actual_avg": g["dk_actual_own"].mean()}
            if g["ft2_own"].notna().sum() > 0:
                row["ft2_avg"]  = g["ft2_own"].mean()
                row["ft2_diff"] = row["ft2_avg"] - row["dk_actual_avg"]
            if g["ft10_own"].notna().sum() > 0:
                row["ft10_avg"]  = g["ft10_own"].mean()
                row["ft10_diff"] = row["ft10_avg"] - row["dk_actual_avg"]
            if g["dk_projected_own"].notna().sum() > 0:
                row["dk_proj_avg"] = g["dk_projected_own"].mean()
                row["proj_bias"]   = row["dk_actual_avg"] - row["dk_proj_avg"]
            rows.append(row)
        return pd.DataFrame(rows)

    def salary_comparison(self) -> pd.DataFrame:
        sql = """
        SELECT
            p.dk_name AS player, t.name AS tournament,
            o.dk_salary, o.ft_salary,
            ROUND(o.ft_salary / (o.dk_salary / 1000.0), 3) AS ft_to_dk_ratio,
            o.dk_actual_own, o.ft2_own, o.total_pts
        FROM ownership o
        JOIN players p ON p.id = o.player_id
        JOIN tournaments t ON t.id = o.tournament_id
        WHERE o.dk_salary IS NOT NULL AND o.ft_salary IS NOT NULL
        ORDER BY ft_to_dk_ratio DESC
        """
        return pd.read_sql_query(sql, self.conn)

    def correlations(self) -> pd.DataFrame:
        rows = []
        for t_name, g in self.get_ownership_data().groupby("tournament"):
            row = {"tournament": t_name}
            g2  = g.dropna(subset=["dk_actual_own", "ft2_own"])
            g10 = g.dropna(subset=["dk_actual_own", "ft10_own"])
            if len(g2) > 5:
                row["r_dk_ft2"] = round(g2["dk_actual_own"].corr(g2["ft2_own"]), 3)
                row["n_ft2"]    = len(g2)
            if len(g10) > 5:
                row["r_dk_ft10"] = round(g10["dk_actual_own"].corr(g10["ft10_own"]), 3)
                row["n_ft10"]    = len(g10)
            rows.append(row)
        return pd.DataFrame(rows)

    def captain_concentration(self) -> pd.DataFrame:
        rows = []
        df = self.get_ownership_data()
        for t_name, g in df.groupby("tournament"):
            row = {"tournament": t_name}
            if g["ft2_captain_own"].notna().sum() > 0:
                row["ft2_top3_cap"]    = round(g["ft2_captain_own"].nlargest(3).sum(), 4)
                row["ft2_top_captain"] = g.loc[g["ft2_captain_own"].idxmax(), "player"]
            if g["ft10_captain_own"].notna().sum() > 0:
                row["ft10_top3_cap"]    = round(g["ft10_captain_own"].nlargest(3).sum(), 4)
                row["ft10_top_captain"] = g.loc[g["ft10_captain_own"].idxmax(), "player"]
            rows.append(row)
        return pd.DataFrame(rows)

    def player_history(self, player_name: str) -> pd.DataFrame:
        sql = """
        SELECT
            t.name AS tournament, t.date,
            o.dk_salary, o.ft_salary,
            o.dk_projected_own, o.dk_projected_pts, o.dk_projected_std, o.dk_actual_own,
            o.ft2_own, o.ft2_captain_own,
            o.ft10_own, o.ft10_captain_own,
            o.total_pts, o.finish
        FROM ownership o
        JOIN tournaments t ON t.id = o.tournament_id
        JOIN players p ON p.id = o.player_id
        WHERE p.dk_name LIKE ?
        ORDER BY t.date
        """
        return pd.read_sql_query(sql, self.conn, params=[f"%{player_name}%"])

    def close(self):
        self.conn.close()
