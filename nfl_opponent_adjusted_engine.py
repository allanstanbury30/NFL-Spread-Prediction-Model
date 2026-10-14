"""Opponent-adjusted NFL EPA/play and Success Rate engine.

Purpose
-------
Build time-aware team efficiency snapshots from nflverse play-by-play data loaded
through nflreadpy. For a target week W, the snapshot uses only plays from weeks
strictly less than W. This makes the output safe to merge into historical
kickoff-time model rows without leaking future results.

Model
-----
For each completed team-game offensive observation, fit two weighted ridge models:

    EPA/play     = intercept + offense_team_effect + defense_opponent_effect
    Success Rate = intercept + offense_team_effect + defense_opponent_effect

Rows are weighted by qualifying play count and, optionally, an exponential
recency decay. Ridge regularization shrinks team effects toward zero, which means
team estimates regress toward the league-average intercept when samples are small.

Interpretation
--------------
- adj_off_epa: expected EPA/play for that offense vs an average defense.
  Higher is better.
- adj_def_epa_allowed: expected EPA/play allowed by that defense vs an average
  offense. Lower is better.
- adj_off_success_rate: expected success rate vs an average defense.
  Higher is better.
- adj_def_success_rate_allowed: expected success rate allowed vs an average
  offense. Lower is better.

The current-season snapshot engine is intentionally separate from prior-season
blending/continuity logic. That layer can consume these snapshots afterward.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import nflreadpy as nfl


REQUIRED_PBP_COLUMNS = {
    "game_id",
    "season",
    "season_type",
    "week",
    "posteam",
    "defteam",
    "play_type",
    "epa",
}


@dataclass(frozen=True)
class EngineConfig:
    """Configuration for the opponent-adjustment engine."""

    rolling_weeks: Optional[int] = 6
    recency_half_life_weeks: Optional[float] = 3.0
    ridge_alpha_epa: float = 200.0
    ridge_alpha_success: float = 200.0
    season_types: tuple[str, ...] = ("REG",)
    exclude_kneels: bool = True
    exclude_spikes: bool = True
    exclude_no_plays: bool = True


# ---------------------------------------------------------------------------
# 1. DATA INGESTION
# ---------------------------------------------------------------------------


def load_pbp(seasons: Iterable[int]) -> pd.DataFrame:
    """Load nflverse play-by-play with nflreadpy and convert Polars -> pandas."""

    seasons = sorted({int(s) for s in seasons})
    if not seasons:
        raise ValueError("At least one season is required.")

    print(f"Loading nflverse PBP for seasons: {seasons}")
    pbp_polars = nfl.load_pbp(seasons)
    pbp = pbp_polars.to_pandas()

    missing = REQUIRED_PBP_COLUMNS.difference(pbp.columns)
    if missing:
        raise KeyError(
            "nflverse PBP is missing required columns: "
            + ", ".join(sorted(missing))
        )

    return pbp


# ---------------------------------------------------------------------------
# 2. PLAY FILTERING + RAW TEAM-GAME METRICS
# ---------------------------------------------------------------------------


def prepare_scrimmage_plays(
    pbp: pd.DataFrame,
    config: EngineConfig,
) -> pd.DataFrame:
    """Keep meaningful run/pass plays and derive success = EPA > 0.

    Notes
    -----
    - Sacks are normally represented within nflfastR passing plays and therefore
      remain in the sample.
    - QB scrambles remain in the sample.
    - No-plays, kneels, and spikes can be excluded through config flags.
    """

    df = pbp.copy()

    df = df[df["season_type"].isin(config.season_types)].copy()
    df = df[df["play_type"].isin(["run", "pass"])].copy()
    df = df[df["posteam"].notna() & df["defteam"].notna()].copy()
    df = df[df["epa"].notna()].copy()

    if config.exclude_no_plays and "no_play" in df.columns:
        df = df[df["no_play"].fillna(0).astype(float) != 1].copy()

    if config.exclude_kneels and "qb_kneel" in df.columns:
        df = df[df["qb_kneel"].fillna(0).astype(float) != 1].copy()

    if config.exclude_spikes and "qb_spike" in df.columns:
        df = df[df["qb_spike"].fillna(0).astype(float) != 1].copy()

    # The requested definition of success is simply positive EPA.
    df["success"] = (df["epa"] > 0).astype(float)

    keep = [
        "game_id",
        "season",
        "week",
        "posteam",
        "defteam",
        "epa",
        "success",
    ]
    return df[keep].reset_index(drop=True)


def build_team_game_metrics(plays: pd.DataFrame) -> pd.DataFrame:
    """Create one row per team offense per game plus mirrored defensive metrics.

    The regression itself uses the offensive matchup columns:
      team -> offense
      opponent -> defense faced

    Defensive raw metrics are attached for inspection and validation.
    """

    offense = (
        plays.groupby(
            ["season", "week", "game_id", "posteam", "defteam"],
            as_index=False,
        )
        .agg(
            plays=("epa", "size"),
            raw_off_epa=("epa", "mean"),
            raw_off_success_rate=("success", "mean"),
        )
        .rename(columns={"posteam": "team", "defteam": "opponent"})
    )

    # The offense's realized efficiency is the opponent defense's realized
    # efficiency allowed in the same matchup.
    defense = offense[
        [
            "season",
            "week",
            "game_id",
            "team",
            "opponent",
            "plays",
            "raw_off_epa",
            "raw_off_success_rate",
        ]
    ].rename(
        columns={
            "team": "opponent",
            "opponent": "team",
            "plays": "def_plays",
            "raw_off_epa": "raw_def_epa_allowed",
            "raw_off_success_rate": "raw_def_success_rate_allowed",
        }
    )

    team_game = offense.merge(
        defense,
        on=["season", "week", "game_id", "team", "opponent"],
        how="left",
        validate="one_to_one",
    )

    if team_game[["def_plays", "raw_def_epa_allowed"]].isna().any().any():
        raise ValueError(
            "Could not create complete mirrored defensive team-game metrics. "
            "Inspect the PBP filtering rules and game identifiers."
        )

    return team_game.sort_values(
        ["season", "week", "game_id", "team"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. TIME-AWARE HISTORY SELECTION
# ---------------------------------------------------------------------------


def select_pre_target_history(
    team_game: pd.DataFrame,
    season: int,
    as_of_week: int,
    config: EngineConfig,
) -> pd.DataFrame:
    """Return only data that existed before the target week.

    Example: as_of_week=6 can only use Weeks 1-5. Even an earlier game in Week 6
    is excluded. This is deliberately conservative and prevents same-week leakage.
    """

    hist = team_game[
        (team_game["season"] == int(season))
        & (team_game["week"] < int(as_of_week))
    ].copy()

    if config.rolling_weeks is not None:
        if config.rolling_weeks <= 0:
            raise ValueError("rolling_weeks must be positive or None.")

        first_week = max(1, int(as_of_week) - config.rolling_weeks)
        hist = hist[hist["week"] >= first_week].copy()

    if hist.empty:
        return hist

    max_source_week = int(hist["week"].max())
    if max_source_week >= int(as_of_week):
        raise AssertionError(
            f"Leakage detected: source Week {max_source_week} used for "
            f"target Week {as_of_week}."
        )

    # The immediately preceding week has age 0. Older weeks decay from there.
    hist["age_weeks"] = (int(as_of_week) - 1) - hist["week"].astype(int)

    if config.recency_half_life_weeks is None:
        hist["recency_weight"] = 1.0
    else:
        if config.recency_half_life_weeks <= 0:
            raise ValueError("recency_half_life_weeks must be positive or None.")
        hist["recency_weight"] = np.power(
            0.5,
            hist["age_weeks"] / config.recency_half_life_weeks,
        )

    # Weighted regression uses both sample size and recency.
    hist["regression_weight"] = hist["plays"] * hist["recency_weight"]

    return hist.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. OPPONENT ADJUSTMENT VIA SIMULTANEOUS RIDGE REGRESSION
# ---------------------------------------------------------------------------


def _design_matrix(
    hist: pd.DataFrame,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Build offense-team and defense-opponent one-hot blocks.

    X = [offense team indicators | defense team indicators]
    """

    teams = sorted(set(hist["team"]).union(hist["opponent"]))
    team_to_idx = {team: i for i, team in enumerate(teams)}

    n_rows = len(hist)
    n_teams = len(teams)
    X = np.zeros((n_rows, 2 * n_teams), dtype=np.float64)

    row_idx = np.arange(n_rows)
    off_idx = hist["team"].map(team_to_idx).to_numpy(dtype=int)
    def_idx = hist["opponent"].map(team_to_idx).to_numpy(dtype=int)

    X[row_idx, off_idx] = 1.0
    X[row_idx, n_teams + def_idx] = 1.0

    return X, teams, team_to_idx


def _fit_ridge_metric(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    n_teams: int,
    alpha: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit ridge and re-center offense/defense effects around zero.

    Re-centering keeps predictions identical while making the intercept represent
    a league-average offense vs a league-average defense.
    """

    if alpha < 0:
        raise ValueError("Ridge alpha must be non-negative.")

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y, sample_weight=sample_weight)

    off = model.coef_[:n_teams].astype(float)
    defense = model.coef_[n_teams:].astype(float)
    intercept = float(model.intercept_)

    off_mean = float(off.mean())
    def_mean = float(defense.mean())

    # Preserve every fitted prediction after centering.
    intercept = intercept + off_mean + def_mean
    off = off - off_mean
    defense = defense - def_mean

    return intercept, off, defense


def _weighted_team_raw_metrics(hist: pd.DataFrame) -> pd.DataFrame:
    """Calculate recency/play-weighted raw offense and defense for diagnostics."""

    def weighted_mean(group: pd.DataFrame, value_col: str, weight_col: str) -> float:
        values = group[value_col].to_numpy(dtype=float)
        weights = group[weight_col].to_numpy(dtype=float)
        return float(np.average(values, weights=weights))

    offense_rows = []
    for team, g in hist.groupby("team", sort=True):
        offense_rows.append(
            {
                "team": team,
                "raw_weighted_off_epa": weighted_mean(
                    g, "raw_off_epa", "regression_weight"
                ),
                "raw_weighted_off_success_rate": weighted_mean(
                    g, "raw_off_success_rate", "regression_weight"
                ),
                "off_games": int(g["game_id"].nunique()),
                "off_plays": int(g["plays"].sum()),
            }
        )

    defense_view = hist.rename(
        columns={
            "opponent": "defense_team",
            "raw_off_epa": "epa_allowed",
            "raw_off_success_rate": "success_allowed",
        }
    )

    defense_rows = []
    for team, g in defense_view.groupby("defense_team", sort=True):
        defense_rows.append(
            {
                "team": team,
                "raw_weighted_def_epa_allowed": weighted_mean(
                    g, "epa_allowed", "regression_weight"
                ),
                "raw_weighted_def_success_rate_allowed": weighted_mean(
                    g, "success_allowed", "regression_weight"
                ),
                "def_games": int(g["game_id"].nunique()),
                "def_plays": int(g["plays"].sum()),
            }
        )

    return pd.DataFrame(offense_rows).merge(
        pd.DataFrame(defense_rows),
        on="team",
        how="outer",
        validate="one_to_one",
    )


def fit_opponent_adjusted_snapshot(
    team_game: pd.DataFrame,
    season: int,
    as_of_week: int,
    config: EngineConfig,
) -> pd.DataFrame:
    """Fit a leakage-safe opponent-adjusted snapshot for one season/week.

    For target Week W, only Weeks < W are eligible.
    """

    hist = select_pre_target_history(
        team_game=team_game,
        season=season,
        as_of_week=as_of_week,
        config=config,
    )

    if hist.empty:
        return pd.DataFrame()

    X, teams, _ = _design_matrix(hist)
    n_teams = len(teams)
    sample_weight = hist["regression_weight"].to_numpy(dtype=float)

    epa_intercept, off_epa_effect, def_epa_effect = _fit_ridge_metric(
        X=X,
        y=hist["raw_off_epa"].to_numpy(dtype=float),
        sample_weight=sample_weight,
        n_teams=n_teams,
        alpha=config.ridge_alpha_epa,
    )

    success_intercept, off_success_effect, def_success_effect = _fit_ridge_metric(
        X=X,
        y=hist["raw_off_success_rate"].to_numpy(dtype=float),
        sample_weight=sample_weight,
        n_teams=n_teams,
        alpha=config.ridge_alpha_success,
    )

    ratings = pd.DataFrame(
        {
            "team": teams,
            # Expected offense vs an average defense. Higher is better.
            "adj_off_epa": epa_intercept + off_epa_effect,
            "adj_off_success_rate": success_intercept + off_success_effect,
            # Expected allowed vs an average offense. Lower is better.
            "adj_def_epa_allowed": epa_intercept + def_epa_effect,
            "adj_def_success_rate_allowed": (
                success_intercept + def_success_effect
            ),
            # Zero-centered latent strength effects, useful as model features.
            "off_epa_strength": off_epa_effect,
            "def_epa_strength": -def_epa_effect,
            "off_success_strength": off_success_effect,
            "def_success_strength": -def_success_effect,
        }
    )

    # Linear ridge on success rate normally stays in range, but clipping the
    # displayed probability-like fields prevents tiny numerical excursions.
    ratings["adj_off_success_rate"] = ratings["adj_off_success_rate"].clip(0, 1)
    ratings["adj_def_success_rate_allowed"] = ratings[
        "adj_def_success_rate_allowed"
    ].clip(0, 1)

    raw = _weighted_team_raw_metrics(hist)
    ratings = ratings.merge(raw, on="team", how="left", validate="one_to_one")

    ratings.insert(0, "season", int(season))
    ratings.insert(1, "as_of_week", int(as_of_week))
    ratings.insert(2, "data_through_week", int(hist["week"].max()))
    ratings["league_epa_baseline"] = epa_intercept
    ratings["league_success_baseline"] = success_intercept

    # Final local leakage assertion for this snapshot.
    if not (ratings["data_through_week"] < ratings["as_of_week"]).all():
        raise AssertionError("Time-aware constraint failed in snapshot output.")

    return ratings.sort_values("team").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. BUILD ALL WEEKLY SNAPSHOTS
# ---------------------------------------------------------------------------


def build_weekly_snapshots(
    team_game: pd.DataFrame,
    seasons: Iterable[int],
    config: EngineConfig,
    include_end_of_season_snapshot: bool = True,
) -> pd.DataFrame:
    """Build leakage-safe snapshots keyed by season/as_of_week/team.

    Week 1 has no current-season observations and is intentionally omitted.
    Your prior-season/continuity layer should supply the Week 1 baseline.
    """

    snapshots: list[pd.DataFrame] = []

    for season in sorted({int(s) for s in seasons}):
        season_rows = team_game[team_game["season"] == season]
        if season_rows.empty:
            print(f"Warning: no team-game data for {season}; skipping.")
            continue

        max_week = int(season_rows["week"].max())
        last_target_week = max_week + (1 if include_end_of_season_snapshot else 0)

        for as_of_week in range(2, last_target_week + 1):
            snapshot = fit_opponent_adjusted_snapshot(
                team_game=team_game,
                season=season,
                as_of_week=as_of_week,
                config=config,
            )
            if not snapshot.empty:
                snapshots.append(snapshot)

    if not snapshots:
        return pd.DataFrame()

    result = pd.concat(snapshots, ignore_index=True)

    if not (result["data_through_week"] < result["as_of_week"]).all():
        raise AssertionError("A weekly snapshot contains future-week data.")

    return result


# ---------------------------------------------------------------------------
# 6. VERIFICATION / SANITY CHECKS
# ---------------------------------------------------------------------------


def print_ranking_block(
    title: str,
    df: pd.DataFrame,
    columns: list[str],
) -> None:
    print("\n" + title)
    print("-" * len(title))
    display = df[columns].copy()

    for col in display.columns:
        if "success" in col:
            display[col] = (100 * display[col]).map(lambda x: f"{x:6.2f}%")
        elif "epa" in col:
            display[col] = display[col].map(lambda x: f"{x:+.4f}")

    print(display.to_string(index=False))


def validate_season_rankings(
    team_game: pd.DataFrame,
    season: int = 2024,
    top_n: int = 5,
    ridge_alpha_epa: float = 200.0,
    ridge_alpha_success: float = 200.0,
) -> pd.DataFrame:
    """Print end-of-regular-season top/bottom adjusted offense and defense.

    For easier manual football-reality validation, this function intentionally
    uses the full completed regular season rather than only the rolling six-week
    production window. It still respects chronology because its target is the
    notional week immediately after the final regular-season week.
    """

    season_rows = team_game[team_game["season"] == int(season)]
    if season_rows.empty:
        raise ValueError(f"No data found for season {season}.")

    final_week = int(season_rows["week"].max())
    as_of_week = final_week + 1

    validation_config = EngineConfig(
        rolling_weeks=None,
        recency_half_life_weeks=None,
        ridge_alpha_epa=ridge_alpha_epa,
        ridge_alpha_success=ridge_alpha_success,
    )

    ratings = fit_opponent_adjusted_snapshot(
        team_game=team_game,
        season=season,
        as_of_week=as_of_week,
        config=validation_config,
    )

    print(
        f"\nVALIDATION: {season} regular season | "
        f"snapshot as of Week {as_of_week} | "
        f"data through Week {final_week} only"
    )

    if int(ratings["data_through_week"].max()) >= as_of_week:
        raise AssertionError("Leakage check failed during validation.")

    offense_cols = ["team", "adj_off_epa", "adj_off_success_rate"]
    defense_cols = [
        "team",
        "adj_def_epa_allowed",
        "adj_def_success_rate_allowed",
    ]

    top_off = ratings.nlargest(top_n, "adj_off_epa")
    bottom_off = ratings.nsmallest(top_n, "adj_off_epa")

    # Lower EPA allowed is better defense.
    top_def = ratings.nsmallest(top_n, "adj_def_epa_allowed")
    bottom_def = ratings.nlargest(top_n, "adj_def_epa_allowed")

    print_ranking_block(f"TOP {top_n} ADJUSTED OFFENSES", top_off, offense_cols)
    print_ranking_block(
        f"BOTTOM {top_n} ADJUSTED OFFENSES", bottom_off, offense_cols
    )
    print_ranking_block(f"TOP {top_n} ADJUSTED DEFENSES", top_def, defense_cols)
    print_ranking_block(
        f"BOTTOM {top_n} ADJUSTED DEFENSES", bottom_def, defense_cols
    )

    # Extra sanity checks: ridge effects are centered around zero.
    checks = {
        "mean_off_epa_strength": ratings["off_epa_strength"].mean(),
        "mean_def_epa_strength": ratings["def_epa_strength"].mean(),
        "mean_off_success_strength": ratings["off_success_strength"].mean(),
        "mean_def_success_strength": ratings["def_success_strength"].mean(),
    }
    print("\nCENTERING CHECKS (should be approximately 0)")
    for name, value in checks.items():
        print(f"{name:28s}: {value:+.8f}")

    return ratings


# ---------------------------------------------------------------------------
# 7. COMMAND-LINE ENTRY POINT
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build time-aware opponent-adjusted NFL EPA/Success ratings."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2019, 2020, 2021, 2022, 2023, 2024],
        help="NFL seasons to load.",
    )
    parser.add_argument(
        "--verify-season",
        type=int,
        default=2024,
        help="Historical season for the printed ranking sanity check.",
    )
    parser.add_argument(
        "--rolling-weeks",
        type=int,
        default=6,
        help="Completed weeks retained in each production snapshot.",
    )
    parser.add_argument(
        "--half-life",
        type=float,
        default=3.0,
        help="Recency half-life in weeks. Use 0 to disable decay.",
    )
    parser.add_argument(
        "--alpha-epa",
        type=float,
        default=200.0,
        help="Ridge penalty for EPA model. Tune only with walk-forward CV.",
    )
    parser.add_argument(
        "--alpha-success",
        type=float,
        default=200.0,
        help="Ridge penalty for Success Rate model. Tune only with walk-forward CV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/opponent_adjusted_team_snapshots.parquet"),
        help="Output Parquet path for weekly snapshots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    half_life = None if args.half_life == 0 else args.half_life

    config = EngineConfig(
        rolling_weeks=args.rolling_weeks,
        recency_half_life_weeks=half_life,
        ridge_alpha_epa=args.alpha_epa,
        ridge_alpha_success=args.alpha_success,
    )

    pbp = load_pbp(args.seasons)
    plays = prepare_scrimmage_plays(pbp, config)
    team_game = build_team_game_metrics(plays)

    print(f"Qualifying plays: {len(plays):,}")
    print(f"Team-game rows:   {len(team_game):,}")

    snapshots = build_weekly_snapshots(
        team_game=team_game,
        seasons=args.seasons,
        config=config,
        include_end_of_season_snapshot=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(args.output, index=False)
    print(f"\nSaved {len(snapshots):,} team-week snapshots -> {args.output}")

    validate_season_rankings(
        team_game=team_game,
        season=args.verify_season,
        top_n=5,
        ridge_alpha_epa=args.alpha_epa,
        ridge_alpha_success=args.alpha_success,
    )


if __name__ == "__main__":
    main()
