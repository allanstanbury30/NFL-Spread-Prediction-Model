#!/usr/bin/env python3
"""
run_production_ridge_2026.py
============================

Recovery / Stabilization Ridge production backtest for the NFL Point Spreads
Analyzer.

Structural rollback
-------------------
The model is intentionally returned to the previously stable unified layout:

Component #1
    Unified opponent-adjusted EPA/play and Success Rate. No passing/rushing
    sub-unit split and no era-specific feature blocks.

Component #2
    Single prior-season/current-season continuity blend for HC / OC / QB
    changes. No multi-era baseline system.

Final Ridge
    Unified offensive EPA and defensive EPA remain separate linear inputs.
    Offensive EPA receives modestly greater post-standardization elasticity than
    defensive EPA, allowing Ridge to penalize the noisier defensive signal more
    aggressively without creating additional football sub-units.

Data-refinement layers
----------------------
1. Garbage-time ingestion filter
   Second-half scrimmage plays are removed when live win probability is above
   95% or below 5%. The filter is applied to PBP before EPA aggregation,
   turnover smoothing, opponent adjustment, and continuity blending.

2. Turnover-luck regression
   Game-level turnover differential is regressed 50% toward league mean before
   opponent-adjusted EPA is fit. The estimated EPA cost per turnover comes only
   from the initial pre-test training span.

3. Iterative strength-of-schedule smoother
   Each team-game observation receives a bounded weight based on the opponent's
   leakage-safe defensive EPA rank entering that specific week. Two iterations
   are used: the first estimates defense strength, the second re-estimates those
   ranks after the first schedule weights are applied. Strong-defense games are
   modestly upweighted; weak-defense games are modestly downweighted.

4. Modern home-field baseline
   The active-season intercept calibration uses a fixed +1.5-point modern NFL
   home-field baseline. Because the final model predicts residual versus the
   sportsbook market, the +1.5 value is used as the HOME-field reference for
   the adaptive intercept rather than blindly adding 1.5 points on top of a
   market line that already contains home-field information.

5. Active 2026 four-week tracker
   Before each 2026 target week, the script scans only the prior four completed
   2026 weeks (or fewer early in the season) and calculates home scoring margin,
   favorite/underdog ATS cover rates, and average total points. The sample is
   shrunk before producing a small intercept adjustment.

6. Key-number and spread safeguards
   Post-prediction bet qualification retains critical-number weights, including
   1.35x when the projection crosses 3 and 1.30x when it crosses 7. Heavy
   spreads above 7 increase Ridge alpha and explicitly shrink the predicted
   market residual.

7. Global variance squelch
   Final market-residual Ridge alpha defaults to 250.0. Target-season recency
   sample weighting remains available, but the model is intentionally kept
   linear and aggressively regularized.

Walk-forward / display
----------------------
Default backtest:
    initial train span: 2018-2023
    test seasons:       2024-2026

2024 trains on earlier history, 2025 expands through 2024, and 2026 expands
through 2025. The terminal prints 2024, 2025, and 2026 side-by-side.

Leakage protections
-------------------
- A target week never trains on the same or a later week.
- SOS rank for a Week W game uses only data from Weeks < W.
- Active 2026 environment for Week W uses only prior completed 2026 weeks.
- Turnover EPA conversion is estimated only from the initial training span.
- Key-number weighting affects bet selection only, never the regression target.

Example
-------
python run_production_ridge_2026.py \
    --continuity team_continuity_2018_2026.csv \
    --train-start 2018 \
    --train-end 2023 \
    --test-start 2024 \
    --test-end 2026 \
    --ridge-alpha 250 \
    --thresholds 1.0 1.5 2.0 2.5 3.0 \
    --output-dir data/production_ridge_2026
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# GLOBALS
# ---------------------------------------------------------------------------

BLENDED_FEATURES: Tuple[str, ...] = (
    "blended_adj_off_epa",
    "blended_adj_def_epa_allowed",
    "blended_adj_off_success_rate",
    "blended_adj_def_success_rate_allowed",
)

# Canonicalization is mostly for joining external market / continuity files to
# nflverse history across relocations and common abbreviation variants.
TEAM_ALIASES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "LAR": "LA",
    "JAX": "JAC",
    "WSH": "WAS",
}


@dataclass(frozen=True)
class BacktestConfig:
    # Initial training span. Testing expands chronologically.
    train_start: int = 2018
    train_end: int = 2023
    test_start: int = 2024
    test_end: int = 2026

    # Component #1 unified EPA / Success Rate.
    rolling_weeks: int = 6
    recency_half_life_weeks: float = 3.0
    ridge_alpha_team_epa: float = 200.0
    ridge_alpha_team_success: float = 200.0

    # Garbage-time PBP ingestion filter.
    garbage_time_wp_low: float = 0.05
    garbage_time_wp_high: float = 0.95

    # Turnover luck regression.
    turnover_regression_strength: float = 0.50
    turnover_epa_min: float = 1.0
    turnover_epa_max: float = 6.0

    # Iterative opponent-defense-rank SOS weighting.
    sos_iterations: int = 2
    sos_rank_weight_strength: float = 0.15
    sos_min_weight: float = 0.85
    sos_max_weight: float = 1.15

    # Final aggressively regularized market-residual Ridge.
    model_ridge_alpha: float = 250.0

    # Unified asymmetric offensive/defensive EPA elasticity.
    offensive_epa_elasticity: float = 1.25
    defensive_epa_elasticity: float = 0.90

    # Target-season recency sample weighting retained from stable framework.
    recent_game_weight: float = 2.0
    recent_game_count: int = 2
    recent_weight_half_life_games: float = 2.0

    # Heavy-spread variance control.
    heavy_spread_threshold: float = 7.0
    alpha_growth_per_point: float = 0.25
    max_alpha_multiplier: float = 3.0

    # Light closing-market safety anchor.
    market_anchor_weight: float = 0.05

    # Active-season rolling window.
    market_bias_lookback_weeks: int = 4
    active_season: int = 2026
    active_trend_prior_games: float = 64.0

    # Modern HFA reference used by the active intercept calibration.
    modern_hfa_points: float = 1.50
    active_hfa_weight: float = 0.50
    active_cover_balance_points: float = 3.0
    active_total_scale_min: float = 0.90
    active_total_scale_max: float = 1.10
    active_max_intercept_shift: float = 1.25

    min_train_games: int = 500
    max_train_seasons: Optional[int] = None

    continuity_mode: str = "global"
    hard_zero_on_any_change: bool = False
    allow_missing_continuity: bool = False

    default_american_odds: float = -110.0


# ---------------------------------------------------------------------------
# COMPONENT LOADING
# ---------------------------------------------------------------------------

def _import_module_from_path(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Component not found: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _find_component(
    explicit_path: Optional[str],
    candidates: Sequence[str],
    script_dir: Path,
):
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Explicit component path does not exist: {path}")
        return path

    search_dirs = [
        script_dir,
        Path.cwd(),
        Path("/mnt/data"),
    ]

    for directory in search_dirs:
        for name in candidates:
            candidate = directory / name
            if candidate.exists():
                return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate component. Tried:\n  "
        + "\n  ".join(
            str(directory / name)
            for directory in search_dirs
            for name in candidates
        )
    )


def load_components(
    component1_path: Optional[str] = None,
    component2_path: Optional[str] = None,
    component3_path: Optional[str] = None,
):
    """
    Load the three component modules.

    Component #1 supports both the filename used in this conversation and the
    shorter filename mentioned in the project spec.
    """
    script_dir = Path(__file__).resolve().parent

    c1_path = _find_component(
        component1_path,
        candidates=(
            "opponent_adjustment_engine.py",
            "nfl_opponent_adjusted_engine.py",
        ),
        script_dir=script_dir,
    )
    c2_path = _find_component(
        component2_path,
        candidates=("prior_season_continuity_blender.py",),
        script_dir=script_dir,
    )
    c3_path = _find_component(
        component3_path,
        candidates=("nfl_spread_prediction_engine.py", "nfl_spread_prediction_engine_option3.py"),
        script_dir=script_dir,
    )

    print("Using components:")
    print(f"  #1 {c1_path}")
    print(f"  #2 {c2_path}")
    print(f"  #3 {c3_path}")

    c1 = _import_module_from_path("nfl_component1", c1_path)
    c2 = _import_module_from_path("nfl_component2", c2_path)
    c3 = _import_module_from_path("nfl_component3", c3_path)

    return c1, c2, c3


# ---------------------------------------------------------------------------
# GENERIC IO / NORMALIZATION
# ---------------------------------------------------------------------------

def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported input format: {path}. Use CSV or Parquet.")


def load_context_feature_module():
    """
    Import schedule_context_features.py from the same directory as this script
    (or the current working directory as fallback).
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "schedule_context_features.py",
        Path.cwd() / "schedule_context_features.py",
        Path("/mnt/data/schedule_context_features.py"),
    ]

    for path in candidates:
        if path.exists():
            return _import_module_from_path(
                "nfl_schedule_context_features",
                path.resolve(),
            )

    raise FileNotFoundError(
        "Could not locate schedule_context_features.py. "
        "Place it beside run_backtest.py."
    )


def canonical_team(value):
    if pd.isna(value):
        return value
    team = str(value).strip().upper()
    return TEAM_ALIASES.get(team, team)


def normalize_team_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(canonical_team)
    return out


def american_to_raw_probability(odds) -> float:
    if pd.isna(odds):
        return np.nan

    odds = float(odds)

    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    if odds > 0:
        return 100.0 / (odds + 100.0)

    return np.nan


def add_devigged_moneyline_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market_home_win_prob / market_away_win_prob if both closing ML columns
    are available. These are permitted for the kickoff-time model.
    """
    out = df.copy()

    if not {"close_home_ml", "close_away_ml"}.issubset(out.columns):
        return out

    hp = out["close_home_ml"].map(american_to_raw_probability)
    ap = out["close_away_ml"].map(american_to_raw_probability)
    total = hp + ap

    valid = total.notna() & (total > 0)

    out["market_home_win_prob"] = np.where(valid, hp / total, np.nan)
    out["market_away_win_prob"] = np.where(valid, ap / total, np.nan)

    return out


# ---------------------------------------------------------------------------
# NFLVERSE DATA INGESTION
# ---------------------------------------------------------------------------

def load_nflverse_schedules(seasons: Sequence[int]) -> pd.DataFrame:
    """
    Load schedule/game data with nflreadpy.
    """
    try:
        import nflreadpy as nfl
    except ImportError as exc:
        raise ImportError(
            "nflreadpy is required. Install it with:\n"
            "    pip install nflreadpy"
        ) from exc

    print(f"Loading nflverse schedules: {min(seasons)}-{max(seasons)}")
    schedules = nfl.load_schedules(list(seasons))

    if hasattr(schedules, "to_pandas"):
        schedules = schedules.to_pandas()

    return schedules


def _build_kickoff_utc(schedule: pd.DataFrame) -> pd.Series:
    """
    Convert nflverse gameday + gametime (Eastern) into UTC.

    If a kickoff_utc column already exists, use it.
    """
    if "kickoff_utc" in schedule.columns:
        return pd.to_datetime(schedule["kickoff_utc"], utc=True, errors="coerce")

    if not {"gameday", "gametime"}.issubset(schedule.columns):
        raise ValueError(
            "Schedule needs either kickoff_utc or both gameday and gametime."
        )

    local = pd.to_datetime(
        schedule["gameday"].astype(str)
        + " "
        + schedule["gametime"].astype(str),
        errors="coerce",
    )

    # nflverse historical kickoff times are represented in U.S. Eastern time.
    localized = local.dt.tz_localize(
        "America/New_York",
        ambiguous="NaT",
        nonexistent="shift_forward",
    )

    return localized.dt.tz_convert("UTC")


def prepare_schedule_table(
    schedule: pd.DataFrame,
    seasons: Sequence[int],
) -> pd.DataFrame:
    """
    Normalize nflverse schedule data into Component #3's expected format.
    """
    df = schedule.copy()

    required = {
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Schedule is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df[df["season"].isin(list(seasons))].copy()

    if "game_type" in df.columns:
        df = df[df["game_type"] == "REG"].copy()

    df = normalize_team_columns(df, ["home_team", "away_team"])
    df["kickoff_utc"] = _build_kickoff_utc(df)

    # Context features.
    if "home_rest" in df.columns and "home_rest_days" not in df.columns:
        df["home_rest_days"] = pd.to_numeric(df["home_rest"], errors="coerce")

    if "away_rest" in df.columns and "away_rest_days" not in df.columns:
        df["away_rest_days"] = pd.to_numeric(df["away_rest"], errors="coerce")

    if "div_game" in df.columns and "division_game" not in df.columns:
        df["division_game"] = (
            pd.to_numeric(df["div_game"], errors="coerce").fillna(0).astype(int)
        )

    # nflverse schedule market fallback.
    #
    # nflverse spread_line uses the opposite sign from Component #3:
    # a positive nflverse spread_line means the HOME team is favored.
    # Therefore:
    #
    #     nflverse +3.5  -> Component #3 home spread -3.5
    #
    if "spread_line" in df.columns:
        df["close_spread_home"] = -pd.to_numeric(
            df["spread_line"],
            errors="coerce",
        )

    if "total_line" in df.columns:
        df["close_total"] = pd.to_numeric(
            df["total_line"],
            errors="coerce",
        )

    if "home_moneyline" in df.columns:
        df["close_home_ml"] = pd.to_numeric(
            df["home_moneyline"],
            errors="coerce",
        )

    if "away_moneyline" in df.columns:
        df["close_away_ml"] = pd.to_numeric(
            df["away_moneyline"],
            errors="coerce",
        )

    if "home_spread_odds" in df.columns:
        df["bet_american_odds_home"] = pd.to_numeric(
            df["home_spread_odds"],
            errors="coerce",
        )

    if "away_spread_odds" in df.columns:
        df["bet_american_odds_away"] = pd.to_numeric(
            df["away_spread_odds"],
            errors="coerce",
        )

    # Historical rows only.
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# EXTERNAL MARKET DATA
# ---------------------------------------------------------------------------

def normalize_market_table(market: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize an external market archive.

    For safety, an external market table must provide close_spread_home directly.
    We intentionally do NOT guess the sign convention of a generic "spread"
    column.
    """
    df = market.copy()

    rename_aliases = {
        "home_close_spread": "close_spread_home",
        "closing_spread_home": "close_spread_home",
        "closing_total": "close_total",
        "home_close_ml": "close_home_ml",
        "away_close_ml": "close_away_ml",
        "closing_home_ml": "close_home_ml",
        "closing_away_ml": "close_away_ml",
    }

    for old, new in rename_aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if "close_spread_home" not in df.columns:
        raise ValueError(
            "External market data must contain close_spread_home "
            "(home favorite = negative). This avoids silently guessing a "
            "sportsbook archive's spread sign convention."
        )

    df["close_spread_home"] = pd.to_numeric(
        df["close_spread_home"],
        errors="coerce",
    )

    for col in (
        "close_total",
        "close_home_ml",
        "close_away_ml",
        "bet_american_odds_home",
        "bet_american_odds_away",
        "bet_spread_home",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = normalize_team_columns(df, ["home_team", "away_team"])

    if "closing_line_timestamp_utc" in df.columns:
        df["closing_line_timestamp_utc"] = pd.to_datetime(
            df["closing_line_timestamp_utc"],
            utc=True,
            errors="coerce",
        )

    return df


def merge_external_market(
    games: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """
    External market values override schedule market fields.

    Join priority:
      1) game_id
      2) season/week/home_team/away_team
    """
    games = games.copy()
    market = normalize_market_table(market)

    market_feature_cols = [
        c for c in (
            "close_spread_home",
            "close_total",
            "close_home_ml",
            "close_away_ml",
            "bet_american_odds_home",
            "bet_american_odds_away",
            "bet_spread_home",
            "closing_line_timestamp_utc",
        )
        if c in market.columns
    ]

    # External source should overwrite schedule fallback values.
    games = games.drop(
        columns=[c for c in market_feature_cols if c in games.columns],
        errors="ignore",
    )

    if (
        "game_id" in market.columns
        and market["game_id"].notna().any()
    ):
        keep = ["game_id", *market_feature_cols]
        market_join = market[keep].drop_duplicates("game_id")

        out = games.merge(
            market_join,
            on="game_id",
            how="left",
            validate="one_to_one",
        )
    else:
        keys = ["season", "week", "home_team", "away_team"]
        missing = set(keys) - set(market.columns)
        if missing:
            raise ValueError(
                "Market file without game_id must contain: "
                + ", ".join(keys)
            )

        keep = [*keys, *market_feature_cols]
        market_join = market[keep].drop_duplicates(keys)

        out = games.merge(
            market_join,
            on=keys,
            how="left",
            validate="one_to_one",
        )

    if "closing_line_timestamp_utc" in out.columns:
        valid_ts = (
            out["closing_line_timestamp_utc"].notna()
            & out["kickoff_utc"].notna()
        )
        leakage = valid_ts & (
            out["closing_line_timestamp_utc"] > out["kickoff_utc"]
        )
        if leakage.any():
            bad = out.loc[
                leakage,
                ["game_id", "closing_line_timestamp_utc", "kickoff_utc"],
            ]
            raise ValueError(
                "Closing-line leakage detected: market timestamp after kickoff.\n"
                + bad.head(10).to_string(index=False)
            )

    return out


# ---------------------------------------------------------------------------
# CONTINUITY DATA
# ---------------------------------------------------------------------------

def load_continuity_table(
    path: str | Path,
    *,
    seasons: Sequence[int],
) -> pd.DataFrame:
    df = read_table(path)

    required = {
        "season",
        "team",
        "hc_retained",
        "oc_retained",
        "qb_retained",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Continuity table missing columns: "
            + ", ".join(sorted(missing))
        )

    df = normalize_team_columns(df, ["team"])
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df = df[df["season"].isin(list(seasons))].copy()

    if df.duplicated(["season", "team"]).any():
        dupes = df[df.duplicated(["season", "team"], keep=False)]
        raise ValueError(
            "Duplicate season/team continuity rows:\n"
            + dupes.sort_values(["season", "team"]).head(20).to_string(index=False)
        )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# COMPONENT #1 DATA REFINEMENT: GARBAGE TIME + ITERATIVE SOS
# ---------------------------------------------------------------------------

def apply_garbage_time_ingestion_filter(
    pbp: pd.DataFrame,
    *,
    wp_low: float = 0.05,
    wp_high: float = 0.95,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Remove second-half plays when live win probability is extreme.

    nflfastR/nflverse normally exposes `wp` on a 0-1 scale. `home_wp` is used
    only as a fallback. Because the cutoff is symmetric around 0.50, either
    team-oriented probability is valid for identifying already-decided games.

    Only Q3/Q4 are treated as the second half. Overtime is deliberately left
    untouched.
    """
    if not 0.0 <= float(wp_low) < float(wp_high) <= 1.0:
        raise ValueError("Garbage-time WP bounds must satisfy 0 <= low < high <= 1.")

    out = pbp.copy()

    wp_col = next(
        (c for c in ("wp", "home_wp") if c in out.columns),
        None,
    )
    if wp_col is None:
        raise KeyError(
            "Garbage-time filter requires nflverse `wp` or `home_wp`."
        )
    if "qtr" not in out.columns:
        raise KeyError("Garbage-time filter requires nflverse `qtr`.")

    wp = pd.to_numeric(out[wp_col], errors="coerce")
    # Defensive compatibility if a provider exports percentages instead of 0-1.
    finite = wp[np.isfinite(wp)]
    if not finite.empty and float(finite.max()) > 1.5:
        wp = wp / 100.0

    qtr = pd.to_numeric(out["qtr"], errors="coerce")
    second_half = qtr.isin([3, 4])
    extreme_wp = (wp > float(wp_high)) | (wp < float(wp_low))
    remove = second_half & extreme_wp & wp.notna()

    removed = int(remove.sum())
    eligible_second_half = int((second_half & wp.notna()).sum())

    filtered = out.loc[~remove].copy().reset_index(drop=True)

    diagnostics = {
        "wp_column": wp_col,
        "plays_before": int(len(out)),
        "plays_removed": removed,
        "plays_after": int(len(filtered)),
        "second_half_wp_plays": eligible_second_half,
        "removed_pct_of_all": (
            float(removed / len(out)) if len(out) else 0.0
        ),
        "wp_low": float(wp_low),
        "wp_high": float(wp_high),
    }
    return filtered, diagnostics


def _install_sos_history_weight_hook(c1) -> None:
    """
    Extend Component #1's existing sample weight with a leakage-safe SOS weight.

    Component #1 already computes:
        plays * recency_weight

    The hook changes this to:
        plays * recency_weight * sos_weight

    without changing the component's API or adding predictive columns.
    """
    if hasattr(c1, "_production_original_select_pre_target_history"):
        return

    original = c1.select_pre_target_history
    c1._production_original_select_pre_target_history = original

    def select_with_sos(team_game, season, as_of_week, config):
        hist = original(team_game, season, as_of_week, config)
        if hist.empty:
            return hist

        if "sos_weight" in hist.columns:
            sos = pd.to_numeric(
                hist["sos_weight"],
                errors="coerce",
            ).fillna(1.0)
            hist["regression_weight"] = (
                pd.to_numeric(
                    hist["regression_weight"],
                    errors="coerce",
                ).fillna(0.0)
                * sos
            )
        return hist

    c1.select_pre_target_history = select_with_sos


def _defensive_rank_weights(
    snapshot: pd.DataFrame,
    *,
    strength: float,
    min_weight: float,
    max_weight: float,
) -> Dict[str, Tuple[float, float]]:
    """
    Map defense -> (rank_percentile, observation_weight).

    Lower adjusted EPA allowed is better. The best defense receives a positive
    centered rank and therefore a modest observation upweight; the weakest
    defense receives a matching downweight.
    """
    if snapshot.empty:
        return {}

    ratings = snapshot[["team", "adj_def_epa_allowed"]].dropna().copy()
    if ratings.empty:
        return {}

    ratings = ratings.sort_values(
        ["adj_def_epa_allowed", "team"],
        ascending=[True, True],
    ).reset_index(drop=True)

    n = len(ratings)
    if n == 1:
        ratings["rank_percentile"] = 0.50
    else:
        ratings["rank_percentile"] = (
            np.arange(n, dtype=float) / float(n - 1)
        )

    # best defense => +1, worst defense => -1
    centered_strength = 1.0 - 2.0 * ratings["rank_percentile"]
    ratings["sos_weight"] = np.clip(
        1.0 + float(strength) * centered_strength,
        float(min_weight),
        float(max_weight),
    )

    return {
        str(row.team): (
            float(row.rank_percentile),
            float(row.sos_weight),
        )
        for row in ratings.itertuples(index=False)
    }


def apply_iterative_sos_smoother(
    c1,
    team_game: pd.DataFrame,
    *,
    config: BacktestConfig,
    snapshot_config,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Iteratively weight each game by opponent defensive rank ENTERING that week.

    Iteration 1 ranks defenses from unweighted history. Iteration 2 recomputes
    those ranks after the first set of schedule weights has been applied. All
    snapshots for Week W use only Weeks < W through Component #1's existing
    leakage-safe selector.
    """
    if int(config.sos_iterations) < 1:
        raise ValueError("sos_iterations must be >= 1.")
    if not 0.0 <= float(config.sos_rank_weight_strength) <= 0.50:
        raise ValueError("sos_rank_weight_strength must be between 0 and 0.50.")
    if not 0 < float(config.sos_min_weight) <= 1.0:
        raise ValueError("sos_min_weight must be in (0, 1].")
    if float(config.sos_max_weight) < 1.0:
        raise ValueError("sos_max_weight must be >= 1.")

    out = team_game.copy().reset_index(drop=True)
    out["sos_weight"] = 1.0
    out["sos_def_rank_percentile"] = 0.50

    _install_sos_history_weight_hook(c1)

    seasons = sorted(
        int(s) for s in out["season"].dropna().unique()
    )

    for _iteration in range(int(config.sos_iterations)):
        next_weight = pd.Series(1.0, index=out.index, dtype=float)
        next_rank = pd.Series(0.50, index=out.index, dtype=float)

        for season in seasons:
            season_rows = out[out["season"] == season]
            weeks = sorted(
                int(w) for w in season_rows["week"].dropna().unique()
            )

            for week in weeks:
                row_mask = (
                    (out["season"] == season)
                    & (out["week"] == week)
                )

                snapshot = c1.fit_opponent_adjusted_snapshot(
                    team_game=out,
                    season=season,
                    as_of_week=week,
                    config=snapshot_config,
                )

                rank_map = _defensive_rank_weights(
                    snapshot,
                    strength=config.sos_rank_weight_strength,
                    min_weight=config.sos_min_weight,
                    max_weight=config.sos_max_weight,
                )

                if not rank_map:
                    continue

                opponents = out.loc[row_mask, "opponent"].astype(str)
                next_weight.loc[row_mask] = opponents.map(
                    lambda team: rank_map.get(team, (0.50, 1.0))[1]
                ).to_numpy(dtype=float)
                next_rank.loc[row_mask] = opponents.map(
                    lambda team: rank_map.get(team, (0.50, 1.0))[0]
                ).to_numpy(dtype=float)

        out["sos_weight"] = next_weight.clip(
            lower=float(config.sos_min_weight),
            upper=float(config.sos_max_weight),
        )
        out["sos_def_rank_percentile"] = next_rank.clip(0.0, 1.0)

    diagnostics = {
        "iterations": int(config.sos_iterations),
        "mean_weight": float(out["sos_weight"].mean()),
        "min_weight": float(out["sos_weight"].min()),
        "max_weight": float(out["sos_weight"].max()),
        "non_neutral_rows": int((~np.isclose(out["sos_weight"], 1.0)).sum()),
    }
    return out, diagnostics


# ---------------------------------------------------------------------------
# TURNOVER LUCK REGRESSION
# ---------------------------------------------------------------------------

def _turnover_play_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """Return regular-season scrimmage plays with a robust turnover flag."""
    required = {
        "game_id", "season", "week", "season_type", "posteam", "defteam",
        "play_type", "epa",
    }
    missing = required - set(pbp.columns)
    if missing:
        raise ValueError(
            "PBP missing turnover-filter columns: "
            + ", ".join(sorted(missing))
        )

    df = pbp.copy()
    df = df[df["season_type"] == "REG"].copy()
    df = df[df["play_type"].isin(["run", "pass"])].copy()
    df = df[df["posteam"].notna() & df["defteam"].notna()].copy()
    df = df[df["epa"].notna()].copy()

    if "no_play" in df.columns:
        df = df[df["no_play"].fillna(0).astype(float) != 1].copy()
    if "qb_kneel" in df.columns:
        df = df[df["qb_kneel"].fillna(0).astype(float) != 1].copy()
    if "qb_spike" in df.columns:
        df = df[df["qb_spike"].fillna(0).astype(float) != 1].copy()

    interception = (
        pd.to_numeric(df["interception"], errors="coerce").fillna(0.0)
        if "interception" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    fumble_lost = (
        pd.to_numeric(df["fumble_lost"], errors="coerce").fillna(0.0)
        if "fumble_lost" in df.columns
        else pd.Series(0.0, index=df.index)
    )

    df["turnover_flag"] = (
        (interception > 0) | (fumble_lost > 0)
    ).astype(float)

    return df


def estimate_turnover_epa_cost(
    pbp: pd.DataFrame,
    *,
    training_end_season: int,
    min_cost: float = 1.0,
    max_cost: float = 6.0,
) -> float:
    """
    Estimate a robust EPA cost per turnover from PRE-TEST history only.

    Median negative offensive EPA on turnover plays is used to avoid a few
    extreme return plays dominating the conversion constant.
    """
    df = _turnover_play_frame(pbp)
    df = df[df["season"] <= int(training_end_season)].copy()
    turns = df[df["turnover_flag"] > 0].copy()

    if turns.empty:
        return 3.0

    costs = (-pd.to_numeric(turns["epa"], errors="coerce")).clip(lower=0.0)
    costs = costs[costs.notna()]
    if costs.empty:
        return 3.0

    value = float(costs.median())
    return float(np.clip(value, float(min_cost), float(max_cost)))


def build_team_game_turnover_differential(
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    """Build one turnover-differential row per team/game."""
    df = _turnover_play_frame(pbp)
    events = df[df["turnover_flag"] > 0].copy()

    base = pd.concat(
        [
            df[["game_id", "season", "week", "posteam"]].rename(
                columns={"posteam": "team"}
            ),
            df[["game_id", "season", "week", "defteam"]].rename(
                columns={"defteam": "team"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates(["game_id", "season", "week", "team"])

    if events.empty:
        base["giveaways"] = 0.0
        base["takeaways"] = 0.0
    else:
        giveaways = (
            events.groupby(
                ["game_id", "season", "week", "posteam"],
                as_index=False,
            )["turnover_flag"]
            .sum()
            .rename(
                columns={
                    "posteam": "team",
                    "turnover_flag": "giveaways",
                }
            )
        )
        takeaways = (
            events.groupby(
                ["game_id", "season", "week", "defteam"],
                as_index=False,
            )["turnover_flag"]
            .sum()
            .rename(
                columns={
                    "defteam": "team",
                    "turnover_flag": "takeaways",
                }
            )
        )
        base = base.merge(
            giveaways,
            on=["game_id", "season", "week", "team"],
            how="left",
            validate="one_to_one",
        )
        base = base.merge(
            takeaways,
            on=["game_id", "season", "week", "team"],
            how="left",
            validate="one_to_one",
        )
        base["giveaways"] = base["giveaways"].fillna(0.0)
        base["takeaways"] = base["takeaways"].fillna(0.0)

    base["turnover_diff"] = base["takeaways"] - base["giveaways"]
    return base


def apply_turnover_luck_regression(
    team_game: pd.DataFrame,
    pbp: pd.DataFrame,
    *,
    training_end_season: int,
    regression_strength: float = 0.50,
    turnover_epa_min: float = 1.0,
    turnover_epa_max: float = 6.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Regress game-level turnover differential toward league mean before
    Component #1 ratings are fit.

    If actual turnover differential is +2 and league mean is ~0, a 50%
    regression treats +1 turnover as persistent signal and +1 as luck to remove.
    The luck EPA is split symmetrically across the two teams' offensive EPA
    observations, so the total game-level net correction equals the regressed
    turnover-luck estimate while preserving separate offense/defense modeling.
    """
    strength = float(regression_strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("turnover_regression_strength must be between 0 and 1.")

    out = team_game.copy()
    turnover = build_team_game_turnover_differential(pbp)
    turnover = normalize_team_columns(turnover, ["team"])

    league_mean = float(turnover["turnover_diff"].mean()) if len(turnover) else 0.0
    turnover_epa_cost = estimate_turnover_epa_cost(
        pbp,
        training_end_season=training_end_season,
        min_cost=turnover_epa_min,
        max_cost=turnover_epa_max,
    )

    out = out.merge(
        turnover[
            [
                "game_id", "season", "week", "team",
                "giveaways", "takeaways", "turnover_diff",
            ]
        ],
        on=["game_id", "season", "week", "team"],
        how="left",
        validate="one_to_one",
    )

    for col in ("giveaways", "takeaways", "turnover_diff"):
        out[col] = out[col].fillna(0.0)

    out["turnover_diff_regressed"] = (
        league_mean
        + (1.0 - strength) * (out["turnover_diff"] - league_mean)
    )
    out["turnover_diff_luck_removed"] = (
        out["turnover_diff"] - out["turnover_diff_regressed"]
    )

    # Split the game-level lucky EPA advantage across both opponent rows.
    # Each team's opposite turnover differential creates the matching reverse
    # correction on the other offensive row.
    out["turnover_luck_epa_points_removed"] = (
        0.5
        * out["turnover_diff_luck_removed"]
        * turnover_epa_cost
    )

    safe_plays = pd.to_numeric(out["plays"], errors="coerce").clip(lower=1.0)
    out["raw_off_epa_observed"] = out["raw_off_epa"]
    out["turnover_luck_epa_per_play_adjustment"] = (
        out["turnover_luck_epa_points_removed"] / safe_plays
    )
    out["raw_off_epa"] = (
        out["raw_off_epa_observed"]
        - out["turnover_luck_epa_per_play_adjustment"]
    )

    diagnostics = {
        "league_turnover_diff_mean": league_mean,
        "turnover_epa_cost": turnover_epa_cost,
        "turnover_regression_strength": strength,
    }
    return out, diagnostics


# ---------------------------------------------------------------------------
# COMPONENT #1 ORCHESTRATION
# ---------------------------------------------------------------------------

def build_component1_outputs(
    c1,
    *,
    pbp: pd.DataFrame,
    all_feature_seasons: Sequence[int],
    config: BacktestConfig,
):
    """
    Build:
      A) current-season rolling opponent-adjusted snapshots
      B) full-season prior-year baseline snapshots

    Prior-season baselines intentionally use the entire completed regular season
    rather than the current-season rolling 6-week window.
    """
    print("\n[1/5] COMPONENT #1 - Opponent-adjusted EPA / Success Rate")

    current_cfg = c1.EngineConfig(
        rolling_weeks=config.rolling_weeks,
        recency_half_life_weeks=config.recency_half_life_weeks,
        ridge_alpha_epa=config.ridge_alpha_team_epa,
        ridge_alpha_success=config.ridge_alpha_team_success,
        season_types=("REG",),
    )

    baseline_cfg = c1.EngineConfig(
        rolling_weeks=None,
        recency_half_life_weeks=None,
        ridge_alpha_epa=config.ridge_alpha_team_epa,
        ridge_alpha_success=config.ridge_alpha_team_success,
        season_types=("REG",),
    )

    filtered_pbp, garbage_diag = apply_garbage_time_ingestion_filter(
        pbp,
        wp_low=config.garbage_time_wp_low,
        wp_high=config.garbage_time_wp_high,
    )
    print(
        "  Garbage-time filter:  "
        f"removed {garbage_diag['plays_removed']:,} second-half plays "
        f"outside {100*config.garbage_time_wp_low:.0f}%/"
        f"{100*config.garbage_time_wp_high:.0f}% WP"
    )

    plays = c1.prepare_scrimmage_plays(filtered_pbp, current_cfg)
    team_game = c1.build_team_game_metrics(plays)

    # Normalize franchise abbreviations before fitting so season-over-season
    # joins remain stable.
    team_game = normalize_team_columns(team_game, ["team", "opponent"])

    # Turnover-luck filter occurs before opponent adjustment and continuity blend.
    team_game, turnover_diag = apply_turnover_luck_regression(
        team_game,
        filtered_pbp,
        training_end_season=config.train_end,
        regression_strength=config.turnover_regression_strength,
        turnover_epa_min=config.turnover_epa_min,
        turnover_epa_max=config.turnover_epa_max,
    )
    print(
        "  Turnover luck filter: "
        f"{100*config.turnover_regression_strength:.0f}% regression | "
        f"EPA/turnover={turnover_diag['turnover_epa_cost']:.3f}"
    )

    # Two-pass leakage-safe SOS smoothing. The installed Component #1 hook
    # multiplies its existing play-count/recency sample weight by sos_weight.
    team_game, sos_diag = apply_iterative_sos_smoother(
        c1,
        team_game,
        config=config,
        snapshot_config=current_cfg,
    )
    print(
        "  Iterative SOS smoother:"
        f" {sos_diag['iterations']} passes | "
        f"weight range {sos_diag['min_weight']:.3f}-"
        f"{sos_diag['max_weight']:.3f}"
    )

    current_snapshots = c1.build_weekly_snapshots(
        team_game=team_game,
        seasons=all_feature_seasons,
        config=current_cfg,
        include_end_of_season_snapshot=True,
    )

    if current_snapshots.empty:
        raise RuntimeError("Component #1 produced no current-season snapshots.")

    if not (
        current_snapshots["data_through_week"]
        < current_snapshots["as_of_week"]
    ).all():
        raise AssertionError("Component #1 leakage check failed.")

    # Full completed-season prior baselines.
    prior_baselines: dict[int, pd.DataFrame] = {}

    for source_season in sorted(set(all_feature_seasons)):
        season_rows = team_game[team_game["season"] == source_season]
        if season_rows.empty:
            continue

        max_week = int(season_rows["week"].max())

        baseline = c1.fit_opponent_adjusted_snapshot(
            team_game=team_game,
            season=source_season,
            as_of_week=max_week + 1,
            config=baseline_cfg,
        )

        if baseline.empty:
            continue

        baseline = normalize_team_columns(baseline, ["team"])

        prior_baselines[source_season + 1] = baseline[
            ["team", *BLENDED_FEATURES_TO_COMPONENT1()]
        ].copy()

    print(
        f"  Current snapshots: {len(current_snapshots):,} team-week rows"
    )
    print(
        f"  Prior baselines:   {len(prior_baselines):,} target seasons"
    )

    return team_game, current_snapshots, prior_baselines


def BLENDED_FEATURES_TO_COMPONENT1() -> Tuple[str, ...]:
    """
    The raw Component #1 field names consumed by Component #2.
    """
    return (
        "adj_off_epa",
        "adj_def_epa_allowed",
        "adj_off_success_rate",
        "adj_def_success_rate_allowed",
    )


# ---------------------------------------------------------------------------
# COMPONENT #2 ORCHESTRATION
# ---------------------------------------------------------------------------

def validate_continuity_coverage(
    schedules: pd.DataFrame,
    continuity: pd.DataFrame,
    target_seasons: Sequence[int],
    *,
    allow_missing: bool,
):
    active = pd.concat(
        [
            schedules[["season", "home_team"]].rename(
                columns={"home_team": "team"}
            ),
            schedules[["season", "away_team"]].rename(
                columns={"away_team": "team"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()

    active = active[active["season"].isin(list(target_seasons))]

    known = continuity[["season", "team"]].drop_duplicates()

    check = active.merge(
        known.assign(has_continuity=1),
        on=["season", "team"],
        how="left",
    )

    missing = check[check["has_continuity"].isna()][["season", "team"]]

    if not missing.empty:
        message = (
            "Continuity table is missing active team-season rows:\n"
            + missing.sort_values(["season", "team"]).to_string(index=False)
        )
        if allow_missing:
            print("WARNING:", message)
        else:
            raise ValueError(message)


def build_component2_blends(
    c2,
    *,
    current_snapshots: pd.DataFrame,
    prior_baselines: Mapping[int, pd.DataFrame],
    schedules: pd.DataFrame,
    continuity: pd.DataFrame,
    target_seasons: Sequence[int],
    config: BacktestConfig,
) -> pd.DataFrame:
    """
    Build one blended rating row per season/week/team.
    """
    print("\n[2/5] COMPONENT #2 - Prior-season / continuity blending")

    blend_cfg = c2.BlendConfig(
        week1_prior_weight=0.80,
        prior_weight_zero_week=8,
        continuity_mode=config.continuity_mode,
        hard_zero_on_any_change=config.hard_zero_on_any_change,
        require_continuity_record=not config.allow_missing_continuity,
    )

    rows: List[pd.DataFrame] = []

    for season in sorted(target_seasons):
        if season not in prior_baselines:
            raise ValueError(
                f"No completed prior-season baseline found for target {season}. "
                f"Load PBP for at least {season - 1}."
            )

        prior = prior_baselines[season]

        season_schedule = schedules[schedules["season"] == season]
        weeks = sorted(
            int(w)
            for w in season_schedule["week"].dropna().unique()
            if int(w) >= 1
        )

        season_continuity = continuity[continuity["season"] == season].copy()

        for week in weeks:
            blended = c2.blend_team_ratings(
                current_snapshots=current_snapshots,
                prior_baseline=prior,
                continuity=season_continuity,
                season=season,
                week=week,
                metrics=BLENDED_FEATURES_TO_COMPONENT1(),
                fallback_baseline=None,
                config=blend_cfg,
            )

            rows.append(blended)

        print(
            f"  {season}: blended {len(weeks)} regular-season weeks"
        )

    if not rows:
        raise RuntimeError("Component #2 produced no blended ratings.")

    blended_all = pd.concat(rows, ignore_index=True)

    # One rating row per team/week.
    if blended_all.duplicated(["season", "week", "team"]).any():
        raise AssertionError("Duplicate season/week/team rows after Component #2.")

    return blended_all


# ---------------------------------------------------------------------------
# FINAL GAME TABLE
# ---------------------------------------------------------------------------

def build_game_table(
    schedules: pd.DataFrame,
    *,
    model_seasons: Sequence[int],
    external_market: Optional[pd.DataFrame],
) -> pd.DataFrame:
    games = schedules[
        schedules["season"].isin(list(model_seasons))
    ].copy()

    if external_market is not None:
        games = merge_external_market(games, external_market)

    if "close_spread_home" not in games.columns:
        raise ValueError(
            "No closing spread available. Supply --market with "
            "close_spread_home or use schedule data containing spread_line."
        )

    # Kickoff-time model wagers are placed at the close.
    games["bet_spread_home"] = games["close_spread_home"]

    # Closing spread odds if available; otherwise Component #3 will use -110.
    games = add_devigged_moneyline_probabilities(games)

    # Drop rows unusable for a completed historical backtest.
    before = len(games)
    games = games.dropna(
        subset=[
            "kickoff_utc",
            "home_score",
            "away_score",
            "close_spread_home",
        ]
    ).copy()

    dropped = before - len(games)
    if dropped:
        print(
            f"WARNING: dropped {dropped} games missing kickoff, score, or close."
        )

    return games.sort_values(
        ["season", "week", "kickoff_utc", "game_id"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# MARKET CALIBRATION FEATURES
# ---------------------------------------------------------------------------

KEY_NUMBER_BONUSES: Dict[float, float] = {
    3.0: 0.35,
    7.0: 0.30,
    4.0: 0.15,
    6.0: 0.12,
    10.0: 0.10,
}


def _favorite_underdog_cover_results(
    games: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series]:
    """
    Return boolean/NaN favorite-cover and underdog-cover outcomes.

    Pick'em games and ATS pushes are excluded (NaN).
    """
    spread = pd.to_numeric(
        games["close_spread_home"],
        errors="coerce",
    )
    home_margin = pd.to_numeric(
        games["home_margin"],
        errors="coerce",
    )
    ats_home = home_margin + spread

    favorite_cover = pd.Series(np.nan, index=games.index, dtype=float)
    underdog_cover = pd.Series(np.nan, index=games.index, dtype=float)

    home_favorite = spread < 0
    away_favorite = spread > 0
    push = np.isclose(
        ats_home.to_numpy(dtype=float),
        0.0,
        equal_nan=False,
    )

    # Home favorite: home ATS win = favorite cover.
    mask = home_favorite & ~push
    favorite_cover.loc[mask] = (
        ats_home.loc[mask] > 0
    ).astype(float)
    underdog_cover.loc[mask] = 1.0 - favorite_cover.loc[mask]

    # Away favorite: home ATS loss = away favorite cover.
    mask = away_favorite & ~push
    favorite_cover.loc[mask] = (
        ats_home.loc[mask] < 0
    ).astype(float)
    underdog_cover.loc[mask] = 1.0 - favorite_cover.loc[mask]

    return favorite_cover, underdog_cover


def calculate_environment_metrics(
    games: pd.DataFrame,
) -> Dict[str, float]:
    """
    Calculate league environment metrics from completed games.
    """
    required = {
        "home_margin",
        "home_score",
        "away_score",
        "close_spread_home",
    }
    missing = required - set(games.columns)
    if missing:
        raise ValueError(
            "Environment metrics missing columns: "
            + ", ".join(sorted(missing))
        )

    completed = games.dropna(
        subset=[
            "home_margin",
            "home_score",
            "away_score",
            "close_spread_home",
        ]
    ).copy()

    if completed.empty:
        return {
            "games": 0,
            "home_field_margin": np.nan,
            "favorite_cover_pct": np.nan,
            "underdog_cover_pct": np.nan,
            "average_total_points": np.nan,
        }

    favorite_cover, underdog_cover = (
        _favorite_underdog_cover_results(completed)
    )

    return {
        "games": int(len(completed)),
        "home_field_margin": float(
            pd.to_numeric(
                completed["home_margin"],
                errors="coerce",
            ).mean()
        ),
        "favorite_cover_pct": float(favorite_cover.mean()),
        "underdog_cover_pct": float(underdog_cover.mean()),
        "average_total_points": float(
            (
                pd.to_numeric(
                    completed["home_score"],
                    errors="coerce",
                )
                + pd.to_numeric(
                    completed["away_score"],
                    errors="coerce",
                )
            ).mean()
        ),
    }


def calculate_active_2026_trends(
    dataset: pd.DataFrame,
    *,
    active_season: int = 2026,
    target_week: Optional[int] = None,
    lookback_weeks: int = 4,
) -> Dict[str, float]:
    """
    Calculate the active-season environment from PRIOR completed weeks only.

    For target Week W, the tracker uses at most Weeks W-4 ... W-1. This makes
    the 2026 calibration adaptive while preserving strict target-week leakage
    protection. If target_week is omitted, the function reports the latest
    available four-week window for audit/output purposes.
    """
    if int(lookback_weeks) < 1:
        raise ValueError("lookback_weeks must be >= 1.")

    season_rows = dataset[
        dataset["season"] == int(active_season)
    ].copy()

    completed_weeks = sorted(
        int(w)
        for w in season_rows.loc[
            season_rows["home_margin"].notna(),
            "week",
        ].dropna().unique()
    )

    if target_week is None:
        target_week = (max(completed_weeks) + 1) if completed_weeks else 1

    end_week = int(target_week) - 1
    start_week = max(1, end_week - int(lookback_weeks) + 1)

    active = season_rows[
        (season_rows["week"] >= start_week)
        & (season_rows["week"] <= end_week)
    ].copy()

    metrics = calculate_environment_metrics(active)
    metrics["season"] = int(active_season)
    metrics["target_week"] = int(target_week)
    metrics["week_start"] = int(start_week)
    metrics["week_end"] = int(end_week)
    metrics["lookback_weeks"] = int(lookback_weeks)

    return metrics


def calculate_historical_environment_baseline(
    dataset: pd.DataFrame,
    *,
    before_season: int = 2026,
) -> Dict[str, float]:
    """
    Historical comparison baseline using only seasons before the active year.
    """
    historical = dataset[
        dataset["season"] < int(before_season)
    ].copy()

    metrics = calculate_environment_metrics(historical)
    metrics["before_season"] = int(before_season)
    return metrics


def print_active_trend_report(
    active: Mapping[str, float],
    historical: Mapping[str, float],
    *,
    modern_hfa_points: float = 1.50,
) -> None:
    print("\nACTIVE 2026 FOUR-WEEK ENVIRONMENT TRACKER")
    print("-" * 72)

    if not active.get("games", 0):
        print("  No prior completed active-season games available.")
        return

    print(
        f"  Target week:            {int(active['target_week'])}"
    )
    print(
        f"  Active sample:          {int(active['games'])} games "
        f"(Weeks {int(active['week_start'])}-{int(active['week_end'])})"
    )
    print(
        f"  Home field margin:      {active['home_field_margin']:+.3f} "
        f"(modern baseline {float(modern_hfa_points):+.3f})"
    )
    print(
        f"  Favorite cover rate:    {100*active['favorite_cover_pct']:.2f}% "
        f"(historical {100*historical['favorite_cover_pct']:.2f}%)"
    )
    print(
        f"  Underdog cover rate:    {100*active['underdog_cover_pct']:.2f}% "
        f"(historical {100*historical['underdog_cover_pct']:.2f}%)"
    )
    print(
        f"  Avg total points/game:  {active['average_total_points']:.3f} "
        f"(historical {historical['average_total_points']:.3f})"
    )


def active_2026_intercept_shift(
    close_spread_home: float,
    active: Mapping[str, float],
    historical: Mapping[str, float],
    *,
    modern_hfa_points: float = 1.50,
    prior_games: float = 64.0,
    hfa_weight: float = 0.50,
    cover_balance_points: float = 3.0,
    total_scale_min: float = 0.90,
    total_scale_max: float = 1.10,
    max_abs_shift: float = 1.25,
) -> float:
    """
    Produce a small HOME-margin adaptive shift from the prior four active weeks.

    The HFA component is explicitly referenced to +1.5 points, not a historical
    +2.5-style assumption. The market already prices home field, so this is a
    *delta* calibration rather than a second unconditional +1.5 addition.
    """
    n = float(active.get("games", 0) or 0)
    if n <= 0:
        return 0.0

    required_values = [
        active.get("home_field_margin"),
        active.get("favorite_cover_pct"),
        active.get("underdog_cover_pct"),
        active.get("average_total_points"),
        historical.get("favorite_cover_pct"),
        historical.get("underdog_cover_pct"),
        historical.get("average_total_points"),
    ]
    if any(pd.isna(v) for v in required_values):
        return 0.0

    confidence = n / (n + float(prior_games))

    hfa_delta = (
        float(active["home_field_margin"])
        - float(modern_hfa_points)
    )
    hfa_component = float(hfa_weight) * hfa_delta

    active_balance = (
        float(active["favorite_cover_pct"])
        - float(active["underdog_cover_pct"])
    )
    historical_balance = (
        float(historical["favorite_cover_pct"])
        - float(historical["underdog_cover_pct"])
    )
    balance_delta = active_balance - historical_balance

    spread = float(close_spread_home)
    if spread < 0:
        favorite_home_sign = 1.0
    elif spread > 0:
        favorite_home_sign = -1.0
    else:
        favorite_home_sign = 0.0

    cover_component = (
        float(cover_balance_points)
        * balance_delta
        * favorite_home_sign
    )

    historical_total = float(historical["average_total_points"])
    if historical_total > 0:
        total_scale = (
            float(active["average_total_points"])
            / historical_total
        )
    else:
        total_scale = 1.0

    total_scale = float(
        np.clip(
            total_scale,
            float(total_scale_min),
            float(total_scale_max),
        )
    )

    raw_shift = (hfa_component + cover_component) * total_scale
    shrunk_shift = confidence * raw_shift

    return float(
        np.clip(
            shrunk_shift,
            -float(max_abs_shift),
            float(max_abs_shift),
        )
    )


def effective_ridge_alpha(
    base_alpha: float,
    abs_market_spread: float,
    *,
    threshold: float = 7.0,
    growth_per_point: float = 0.25,
    max_multiplier: float = 3.0,
) -> float:
    """
    Dynamically increase L2 shrinkage as the market spread grows beyond 7.

    Example with base_alpha=100:
        |spread| = 7 -> alpha 100
        |spread| = 8 -> alpha 125
        |spread| = 10 -> alpha 175

    The multiplier is capped to prevent pathological regularization.
    """
    if base_alpha <= 0:
        raise ValueError("base_alpha must be > 0.")
    if threshold < 0:
        raise ValueError("heavy-spread threshold cannot be negative.")
    if growth_per_point < 0:
        raise ValueError("alpha growth cannot be negative.")
    if max_multiplier < 1:
        raise ValueError("max alpha multiplier must be >= 1.")

    spread = abs(float(abs_market_spread))
    excess = max(0.0, spread - float(threshold))

    multiplier = min(
        1.0 + float(growth_per_point) * excess,
        float(max_multiplier),
    )

    return float(base_alpha) * multiplier


def crossed_key_numbers(
    market_implied_margin: float,
    predicted_margin: float,
) -> List[float]:
    """
    Return absolute NFL key numbers crossed between market and model margins.

    Signed thresholds are considered so +3 and -3 are treated correctly from
    the home-margin perspective.
    """
    start = float(market_implied_margin)
    end = float(predicted_margin)

    if np.isclose(start, end):
        return []

    lo = min(start, end)
    hi = max(start, end)

    crossed: List[float] = []

    for key in KEY_NUMBER_BONUSES:
        for signed_key in (-float(key), float(key)):
            # Include a key if the market starts on it and the model moves off
            # it, or if the interval passes through it.
            if lo <= signed_key <= hi:
                crossed.append(float(key))
                break

    return sorted(set(crossed))


def key_number_value_weight(
    market_implied_margin: float,
    predicted_margin: float,
) -> Tuple[float, str]:
    """
    Apply only the highest-value crossed key-number bonus.

    This makes a 1-point edge through 3 more valuable than a similar 1-point
    move in a non-key region, without allowing multiple keys to compound into
    an extreme decision score.
    """
    crossed = crossed_key_numbers(
        market_implied_margin,
        predicted_margin,
    )

    if not crossed:
        return 1.0, ""

    best_key = max(
        crossed,
        key=lambda k: KEY_NUMBER_BONUSES[k],
    )

    return (
        1.0 + KEY_NUMBER_BONUSES[best_key],
        ",".join(str(int(k)) for k in crossed),
    )


def add_key_number_edge_calibration(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add post-prediction decision-edge calibration.

    raw_model_edge_home remains a real point-spread difference.
    decision_edge_home is a weighted decision score used only for thresholding.
    It MUST NOT replace predicted_home_margin when calculating MAE.
    """
    df = predictions.copy()

    market_margin = -pd.to_numeric(
        df["bet_spread_home"],
        errors="coerce",
    )

    raw_edge = (
        pd.to_numeric(df["predicted_home_margin"], errors="coerce")
        + pd.to_numeric(df["bet_spread_home"], errors="coerce")
    )

    weights = []
    crossed_strings = []

    for book_margin, model_margin in zip(
        market_margin,
        df["predicted_home_margin"],
    ):
        weight, crossed = key_number_value_weight(
            book_margin,
            model_margin,
        )
        weights.append(weight)
        crossed_strings.append(crossed)

    df["raw_edge_at_bet"] = raw_edge
    df["key_number_value_weight"] = weights
    df["key_numbers_crossed"] = crossed_strings
    df["decision_edge_home"] = (
        df["raw_edge_at_bet"]
        * df["key_number_value_weight"]
    )

    return df


# ---------------------------------------------------------------------------
# COMPONENT #3 / WEEKLY WALK-FORWARD
# ---------------------------------------------------------------------------

def choose_model_features(c3, dataset: pd.DataFrame) -> List[str]:
    """
    Build a stable linear feature set with independent offensive and defensive
    unit terms.

    Highly collinear mixed EPA composites are intentionally removed so Ridge can
    estimate offensive and defensive elasticity separately rather than hiding
    both units inside a single symmetric net-strength feature.
    """
    features = list(c3.default_model_features(dataset))

    # Remove duplicate/legacy contextual aliases when canonical fields exist.
    if "net_rest" in dataset.columns:
        features = [f for f in features if f != "rest_diff"]
    if "is_divisional" in dataset.columns:
        features = [f for f in features if f != "division_game"]

    # Remove mixed EPA composites. Keep raw unit EPA and separate off/def diffs.
    mixed_epa = {
        "net_epa_strength_diff",
        "home_off_vs_away_def_epa",
        "away_off_vs_home_def_epa",
    }
    features = [f for f in features if f not in mixed_epa]

    # Likewise remove the mixed success composite but keep independent unit
    # success-rate differences.
    features = [
        f for f in features
        if f not in {
            "net_success_strength_diff",
            "home_off_vs_away_def_success",
            "away_off_vs_home_def_success",
        }
    ]

    for extra in (
        "market_home_win_prob",
        "market_away_win_prob",
        "net_rest",
        "travel_distance_diff",
        "is_divisional",
        "tz_shifted",
    ):
        if extra in dataset.columns and extra not in features:
            features.append(extra)

    return features


class FeatureElasticityTransformer(BaseEstimator, TransformerMixin):
    """
    Multiply standardized columns by feature-specific elasticity factors.

    With Ridge, scaling a standardized feature by w > 1 reduces the effective
    L2 penalty on that original feature by roughly 1/w^2. This gives offensive
    EPA more coefficient freedom without hard-coding a coefficient sign/value.
    """
    def __init__(self, weights: Sequence[float]):
        self.weights = tuple(float(w) for w in weights)

    def fit(self, X, y=None):
        if X.shape[1] != len(self.weights):
            raise ValueError(
                "Feature elasticity vector length does not match model matrix."
            )
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) * np.asarray(self.weights, dtype=float)


def build_feature_elasticity_weights(
    feature_columns: Sequence[str],
    *,
    offensive_epa_elasticity: float,
    defensive_epa_elasticity: float,
) -> List[float]:
    """Create per-column penalty-elasticity multipliers."""
    off_names = {
        "home_blended_adj_off_epa",
        "away_blended_adj_off_epa",
        "off_epa_diff",
    }
    def_names = {
        "home_blended_adj_def_epa_allowed",
        "away_blended_adj_def_epa_allowed",
        "def_epa_diff",
    }

    weights: List[float] = []
    for feature in feature_columns:
        if feature in off_names:
            weights.append(float(offensive_epa_elasticity))
        elif feature in def_names:
            weights.append(float(defensive_epa_elasticity))
        else:
            weights.append(1.0)
    return weights


def make_ridge_residual_model(
    feature_columns: Sequence[str],
    *,
    alpha: float,
    offensive_epa_elasticity: float = 1.25,
    defensive_epa_elasticity: float = 0.90,
) -> Pipeline:
    """
    Standardized, asymmetric L2-regularized market-residual model.
    """
    if alpha <= 0:
        raise ValueError("Ridge alpha must be > 0.")
    if offensive_epa_elasticity <= 0 or defensive_epa_elasticity <= 0:
        raise ValueError("Feature elasticities must be > 0.")

    elasticity = build_feature_elasticity_weights(
        feature_columns,
        offensive_epa_elasticity=offensive_epa_elasticity,
        defensive_epa_elasticity=defensive_epa_elasticity,
    )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("elasticity", FeatureElasticityTransformer(elasticity)),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def build_exponential_training_weights(
    train: pd.DataFrame,
    *,
    target_season: int,
    target_week: int,
    recent_game_weight: float = 2.0,
    recent_game_count: int = 2,
    half_life_games: float = 2.0,
) -> np.ndarray:
    """
    Target-week-aware sample weights for the final Ridge fit.

    Prior-season rows receive 1.0x. In the current target season, the two most
    recent completed weeks receive 2.0x. Older current-season rows decay
    exponentially back toward 1.0x.
    """
    if recent_game_weight < 1.0:
        raise ValueError("recent_game_weight must be >= 1.0.")
    if recent_game_count < 1:
        raise ValueError("recent_game_count must be >= 1.")
    if half_life_games <= 0:
        raise ValueError("recent_weight_half_life_games must be > 0.")

    weights = np.ones(len(train), dtype=float)
    same_season = train["season"].astype(int).to_numpy() == int(target_season)
    if not same_season.any():
        return weights

    weeks = pd.to_numeric(train["week"], errors="coerce").to_numpy(dtype=float)
    age = (int(target_week) - 1) - weeks

    valid = same_season & np.isfinite(age) & (age >= 0)
    if not valid.any():
        return weights

    # Last N completed weeks receive the full recent multiplier.
    recent = valid & (age < int(recent_game_count))
    weights[recent] = float(recent_game_weight)

    # Older same-season weeks decay from the full recent multiplier toward 1.
    older = valid & ~recent
    if older.any():
        excess_age = age[older] - (int(recent_game_count) - 1)
        premium = float(recent_game_weight) - 1.0
        weights[older] = 1.0 + premium * np.power(
            0.5,
            excess_age / float(half_life_games),
        )

    return weights


def apply_market_anchor(
    raw_predicted_home_margin,
    close_spread_home,
    *,
    anchor_weight: float = 0.05,
) -> np.ndarray:
    """
    Lightly regress the Ridge projection toward the sportsbook close.

    close_spread_home convention:
        Home -3.5 -> -3.5

    Therefore the market-implied HOME margin is:
        -close_spread_home
    """
    if not 0.0 <= float(anchor_weight) <= 1.0:
        raise ValueError("market_anchor_weight must be between 0 and 1.")

    raw = np.asarray(raw_predicted_home_margin, dtype=float)
    spread = np.asarray(close_spread_home, dtype=float)
    market_margin = -spread

    return (
        (1.0 - float(anchor_weight)) * raw
        + float(anchor_weight) * market_margin
    )


def weekly_walk_forward_predict(
    c3,
    dataset: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    test_start: int,
    test_end: int,
    config: BacktestConfig,
) -> pd.DataFrame:
    """
    Weekly calibrated Ridge walk-forward simulation.

    The model predicts the realized margin RESIDUAL relative to the closing
    market. Higher alpha therefore means a smaller departure from the market.
    """
    print("\n[4/5] CALIBRATED RIDGE - Weekly walk-forward model")
    print(f"  Base Ridge alpha:       {config.model_ridge_alpha}")
    print(
        f"  Garbage-time filter:    Q3/Q4 WP < {config.garbage_time_wp_low:.2f} "
        f"or > {config.garbage_time_wp_high:.2f}"
    )
    print(
        f"  SOS smoother:           {config.sos_iterations} iterations, "
        f"rank strength={config.sos_rank_weight_strength:.2f}"
    )
    print(f"  Modern HFA baseline:    {config.modern_hfa_points:+.2f}")
    print(
        f"  EPA elasticity:         offense={config.offensive_epa_elasticity:.2f} "
        f"defense={config.defensive_epa_elasticity:.2f}"
    )
    print(
        f"  Recent-game weight:     {config.recent_game_weight:.2f}x "
        f"for last {config.recent_game_count} completed weeks"
    )
    print(f"  Heavy-spread threshold: {config.heavy_spread_threshold}")
    print(f"  Alpha growth / point:   {config.alpha_growth_per_point:.3f}")
    print(f"  Max alpha multiplier:   {config.max_alpha_multiplier:.2f}x")
    print(f"  Market anchor weight:   {config.market_anchor_weight:.3f}")
    print(
        f"  Market-bias lookback:  "
        f" {config.market_bias_lookback_weeks} weeks"
    )

    df = dataset.copy().sort_values(
        ["season", "week", "kickoff_utc", "game_id"]
    ).reset_index(drop=True)

    if "market_implied_home_margin" not in df.columns:
        df["market_implied_home_margin"] = -df["close_spread_home"]

    df["market_residual_target"] = (
        df["home_margin"]
        - df["market_implied_home_margin"]
    )

    historical_environment = calculate_historical_environment_baseline(
        df,
        before_season=config.active_season,
    )

    prediction_blocks: List[pd.DataFrame] = []

    for season in range(test_start, test_end + 1):
        season_test = df[df["season"] == season]

        if season_test.empty:
            print(f"  WARNING: no model rows for test season {season}")
            continue

        weeks = sorted(
            int(w)
            for w in season_test["week"].dropna().unique()
        )

        for week in weeks:
            test = df[
                (df["season"] == season)
                & (df["week"] == week)
            ].copy()

            if season == config.active_season:
                active_trends = calculate_active_2026_trends(
                    df,
                    active_season=config.active_season,
                    target_week=week,
                    lookback_weeks=config.market_bias_lookback_weeks,
                )
                print_active_trend_report(
                    active_trends,
                    historical_environment,
                    modern_hfa_points=config.modern_hfa_points,
                )
            else:
                active_trends = {
                    "games": 0,
                    "home_field_margin": np.nan,
                    "favorite_cover_pct": np.nan,
                    "underdog_cover_pct": np.nan,
                    "average_total_points": np.nan,
                    "target_week": week,
                    "week_start": max(1, week - config.market_bias_lookback_weeks),
                    "week_end": week - 1,
                }

            train = df[
                (df["season"] < season)
                | (
                    (df["season"] == season)
                    & (df["week"] < week)
                )
            ].copy()

            if config.max_train_seasons is not None and not train.empty:
                min_season = season - config.max_train_seasons
                train = train[
                    train["season"] >= min_season
                ].copy()

            train = train.dropna(
                subset=["home_margin", "market_residual_target"]
            )

            if len(train) < config.min_train_games:
                print(
                    f"  SKIP {season} W{week}: "
                    f"{len(train)} training games < "
                    f"{config.min_train_games}"
                )
                continue

            same_or_future_week = (
                (train["season"] > season)
                | (
                    (train["season"] == season)
                    & (train["week"] >= week)
                )
            )
            if same_or_future_week.any():
                raise AssertionError(
                    f"Leakage in calibrated Ridge training "
                    f"for {season} Week {week}."
                )

            # Alpha is known before kickoff because it depends only on the
            # sportsbook's closing spread.
            test["effective_ridge_alpha"] = (
                test["close_spread_home"]
                .abs()
                .map(
                    lambda spread: effective_ridge_alpha(
                        config.model_ridge_alpha,
                        spread,
                        threshold=config.heavy_spread_threshold,
                        growth_per_point=config.alpha_growth_per_point,
                        max_multiplier=config.max_alpha_multiplier,
                    )
                )
            )

            test["raw_model_market_residual"] = np.nan

            # Fit once per alpha tier found in this target week.
            for alpha, idx in test.groupby(
                "effective_ridge_alpha",
                sort=True,
            ).groups.items():
                model = make_ridge_residual_model(
                    feature_columns,
                    alpha=float(alpha),
                    offensive_epa_elasticity=config.offensive_epa_elasticity,
                    defensive_epa_elasticity=config.defensive_epa_elasticity,
                )

                recency_weights = build_exponential_training_weights(
                    train,
                    target_season=season,
                    target_week=week,
                    recent_game_weight=config.recent_game_weight,
                    recent_game_count=config.recent_game_count,
                    half_life_games=config.recent_weight_half_life_games,
                )

                model.fit(
                    train[list(feature_columns)],
                    train["market_residual_target"].astype(float),
                    ridge__sample_weight=recency_weights,
                )

                residual_pred = model.predict(
                    test.loc[list(idx), list(feature_columns)]
                )

                test.loc[
                    list(idx),
                    "raw_model_market_residual",
                ] = residual_pred

            # Explicitly shrink the market-residual correction as alpha rises.
            # This makes the market-variance penalty act on the full correction,
            # including Ridge's unpenalized intercept.
            test["variance_residual_shrink"] = (
                config.model_ridge_alpha
                / test["effective_ridge_alpha"]
            ).clip(upper=1.0)

            test["variance_adjusted_market_residual"] = (
                test["raw_model_market_residual"]
                * test["variance_residual_shrink"]
            )

            raw_margin = (
                test["market_implied_home_margin"].to_numpy(dtype=float)
                + test[
                    "variance_adjusted_market_residual"
                ].to_numpy(dtype=float)
            )

            # Active 2026 rolling-four-week environment nudge. The tracker
            # contains only weeks strictly before this target week.
            test["active_2026_intercept_shift"] = 0.0

            active_mask = (
                (test["season"] == config.active_season)
                & (float(active_trends.get("games", 0) or 0) > 0)
            )

            if active_mask.any():
                test.loc[
                    active_mask,
                    "active_2026_intercept_shift",
                ] = test.loc[
                    active_mask,
                    "close_spread_home",
                ].map(
                    lambda spread: active_2026_intercept_shift(
                        spread,
                        active_trends,
                        historical_environment,
                        modern_hfa_points=config.modern_hfa_points,
                        prior_games=config.active_trend_prior_games,
                        hfa_weight=config.active_hfa_weight,
                        cover_balance_points=config.active_cover_balance_points,
                        total_scale_min=config.active_total_scale_min,
                        total_scale_max=config.active_total_scale_max,
                        max_abs_shift=config.active_max_intercept_shift,
                    )
                )

            trend_adjusted_raw_margin = (
                raw_margin
                + test[
                    "active_2026_intercept_shift"
                ].to_numpy(dtype=float)
            )

            # Existing 5% safety anchor.
            pred = apply_market_anchor(
                raw_predicted_home_margin=trend_adjusted_raw_margin,
                close_spread_home=test[
                    "close_spread_home"
                ].to_numpy(dtype=float),
                anchor_weight=config.market_anchor_weight,
            )

            keep = [
                c for c in (
                    "game_id",
                    "season",
                    "week",
                    "kickoff_utc",
                    "home_team",
                    "away_team",
                    "home_score",
                    "away_score",
                    "home_margin",
                    "close_spread_home",
                    "bet_spread_home",
                    "close_total",
                    "bet_american_odds_home",
                    "bet_american_odds_away",
                    "rolling_dog_cover_rate_4w",
                    "rolling_dog_sample_size_4w",
                    "underdog_side_sign",
                    "market_bias_feature",
                    "effective_ridge_alpha",
                    "raw_model_market_residual",
                    "variance_residual_shrink",
                    "variance_adjusted_market_residual",
                    "active_2026_intercept_shift",
                )
                if c in test.columns
            ]

            block = test[keep].copy()

            block["market_implied_home_margin"] = (
                -block["close_spread_home"]
            )
            block["pre_active_model_home_margin"] = raw_margin
            block["raw_model_home_margin"] = trend_adjusted_raw_margin

            # Same active tracker metrics are attached to each row for audit.
            block["active_2026_tracker_games"] = active_trends.get(
                "games", 0
            )
            block["active_2026_tracker_week_start"] = active_trends.get(
                "week_start", np.nan
            )
            block["active_2026_tracker_week_end"] = active_trends.get(
                "week_end", np.nan
            )
            block["modern_hfa_baseline"] = config.modern_hfa_points
            block["active_2026_home_field_margin"] = active_trends.get(
                "home_field_margin", np.nan
            )
            block["active_2026_favorite_cover_pct"] = active_trends.get(
                "favorite_cover_pct", np.nan
            )
            block["active_2026_underdog_cover_pct"] = active_trends.get(
                "underdog_cover_pct", np.nan
            )
            block["active_2026_avg_total_points"] = active_trends.get(
                "average_total_points", np.nan
            )

            block["historical_home_field_margin"] = historical_environment.get(
                "home_field_margin", np.nan
            )
            block["historical_favorite_cover_pct"] = historical_environment.get(
                "favorite_cover_pct", np.nan
            )
            block["historical_underdog_cover_pct"] = historical_environment.get(
                "underdog_cover_pct", np.nan
            )
            block["historical_avg_total_points"] = historical_environment.get(
                "average_total_points", np.nan
            )

            block["market_anchor_weight"] = (
                config.market_anchor_weight
            )
            block["predicted_home_margin"] = pred

            block["anchored_market_residual"] = (
                block["predicted_home_margin"]
                - block["market_implied_home_margin"]
            )

            block["model_edge_home_close"] = (
                block["predicted_home_margin"]
                + block["close_spread_home"]
            )
            block["model_edge_home_bet"] = (
                block["predicted_home_margin"]
                + block["bet_spread_home"]
            )

            block["train_games"] = len(train)
            block["offensive_epa_elasticity"] = config.offensive_epa_elasticity
            block["defensive_epa_elasticity"] = config.defensive_epa_elasticity
            block["recent_game_weight"] = config.recent_game_weight
            block["model_type"] = "advanced_efficiency_ridge"

            prediction_blocks.append(block)

        predicted_games = sum(
            len(block)
            for block in prediction_blocks
            if (
                not block.empty
                and int(block["season"].iloc[0]) == season
            )
        )

        print(
            f"  {season}: "
            f"{predicted_games:,} out-of-sample predictions"
        )

    if not prediction_blocks:
        return pd.DataFrame()

    predictions = pd.concat(
        prediction_blocks,
        ignore_index=True,
    )

    # Key-number valuation is intentionally AFTER model prediction.
    predictions = add_key_number_edge_calibration(predictions)

    return predictions


def simulate_calibrated_ats_bets(
    c3,
    predictions: pd.DataFrame,
    *,
    edge_threshold: float,
    default_american_odds: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Simulate ATS bets using key-number-weighted DECISION edge.

    Grading, ROI, MAE, and raw margin predictions remain based on the actual
    unweighted football line. Key-number weighting only changes whether a bet
    qualifies for a threshold.
    """
    if predictions.empty:
        return pd.DataFrame(), {
            "edge_threshold": float(edge_threshold),
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate_ex_pushes": np.nan,
            "profit_units": 0.0,
            "roi": np.nan,
            "avg_clv_points": np.nan,
        }

    df = predictions.copy()

    if "decision_edge_home" not in df.columns:
        df = add_key_number_edge_calibration(df)

    df["bet_side"] = np.select(
        [
            df["decision_edge_home"] >= float(edge_threshold),
            df["decision_edge_home"] <= -float(edge_threshold),
        ],
        ["HOME", "AWAY"],
        default="PASS",
    )

    bets = df[df["bet_side"] != "PASS"].copy()

    if bets.empty:
        return bets, {
            "edge_threshold": float(edge_threshold),
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate_ex_pushes": np.nan,
            "profit_units": 0.0,
            "roi": np.nan,
            "avg_clv_points": np.nan,
        }

    bets["edge_at_bet"] = bets["raw_edge_at_bet"]

    bets["actual_home_ats_margin"] = (
        bets["home_margin"]
        + bets["bet_spread_home"]
    )

    side_sign = np.where(
        bets["bet_side"] == "HOME",
        1.0,
        -1.0,
    )

    bets["selected_side_ats_margin"] = (
        bets["actual_home_ats_margin"]
        * side_sign
    )

    bets["bet_result"] = np.where(
        np.isclose(bets["selected_side_ats_margin"], 0.0),
        "PUSH",
        np.where(
            bets["selected_side_ats_margin"] > 0,
            "WIN",
            "LOSS",
        ),
    )

    home_price = (
        bets["bet_american_odds_home"]
        if "bet_american_odds_home" in bets.columns
        else pd.Series(default_american_odds, index=bets.index)
    )
    away_price = (
        bets["bet_american_odds_away"]
        if "bet_american_odds_away" in bets.columns
        else pd.Series(default_american_odds, index=bets.index)
    )

    bets["american_odds"] = np.where(
        bets["bet_side"] == "HOME",
        home_price,
        away_price,
    )

    bets["american_odds"] = (
        pd.to_numeric(
            bets["american_odds"],
            errors="coerce",
        )
        .fillna(default_american_odds)
    )

    bets["profit_units"] = 0.0

    win_mask = bets["bet_result"] == "WIN"
    loss_mask = bets["bet_result"] == "LOSS"

    bets.loc[win_mask, "profit_units"] = bets.loc[
        win_mask,
        "american_odds",
    ].map(c3.american_profit_per_unit)

    bets.loc[loss_mask, "profit_units"] = -1.0

    bets["clv_points"] = (
        (
            bets["bet_spread_home"]
            - bets["close_spread_home"]
        )
        * side_sign
    )

    wins = int((bets["bet_result"] == "WIN").sum())
    losses = int((bets["bet_result"] == "LOSS").sum())
    pushes = int((bets["bet_result"] == "PUSH").sum())
    graded = wins + losses

    return bets, {
        "edge_threshold": float(edge_threshold),
        "bets": int(len(bets)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate_ex_pushes": (
            float(wins / graded)
            if graded
            else np.nan
        ),
        "profit_units": float(
            bets["profit_units"].sum()
        ),
        "roi": float(
            bets["profit_units"].sum() / len(bets)
        ),
        "avg_clv_points": float(
            bets["clv_points"].mean()
        ),
        "avg_raw_abs_edge": float(
            bets["raw_edge_at_bet"].abs().mean()
        ),
        "avg_decision_abs_edge": float(
            bets["decision_edge_home"].abs().mean()
        ),
        "key_cross_rate": float(
            (bets["key_number_value_weight"] > 1.0).mean()
        ),
    }


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------

def threshold_performance_report(
    c3,
    predictions: pd.DataFrame,
    *,
    thresholds: Sequence[float],
    default_american_odds: float,
) -> pd.DataFrame:
    """
    ATS / ROI / MAE report at each minimum model-edge threshold.

    MAE is calculated on the games that qualified as bets at that threshold.
    """
    rows = []

    for threshold in thresholds:
        bets, summary = simulate_calibrated_ats_bets(
            c3,
            predictions,
            edge_threshold=float(threshold),
            default_american_odds=default_american_odds,
        )

        if bets.empty:
            selected_mae = np.nan
            avg_abs_edge = np.nan
        else:
            selected_mae = float(
                mean_absolute_error(
                    bets["home_margin"],
                    bets["predicted_home_margin"],
                )
            )
            avg_abs_edge = float(bets["edge_at_bet"].abs().mean())

        rows.append(
            {
                "edge_threshold": float(threshold),
                "bets": int(summary.get("bets", 0)),
                "wins": int(summary.get("wins", 0)),
                "losses": int(summary.get("losses", 0)),
                "pushes": int(summary.get("pushes", 0)),
                "ats_record": (
                    f"{int(summary.get('wins', 0))}-"
                    f"{int(summary.get('losses', 0))}-"
                    f"{int(summary.get('pushes', 0))}"
                ),
                "win_rate": summary.get("win_rate_ex_pushes", np.nan),
                "profit_units": summary.get("profit_units", 0.0),
                "roi": summary.get("roi", np.nan),
                "mae_selected_games": selected_mae,
                "avg_abs_model_edge": avg_abs_edge,
                "avg_clv_points": summary.get("avg_clv_points", np.nan),
                "avg_decision_abs_edge": summary.get(
                    "avg_decision_abs_edge", np.nan
                ),
                "key_cross_rate": summary.get("key_cross_rate", np.nan),
            }
        )

    return pd.DataFrame(rows)


def per_season_report(
    c3,
    predictions: pd.DataFrame,
    *,
    threshold: float,
    default_american_odds: float,
) -> pd.DataFrame:
    rows = []

    for season, group in predictions.groupby("season", sort=True):
        bets, summary = simulate_calibrated_ats_bets(
            c3,
            group,
            edge_threshold=threshold,
            default_american_odds=default_american_odds,
        )

        rows.append(
            {
                "season": int(season),
                "edge_threshold": float(threshold),
                "predictions": len(group),
                "bets": int(summary.get("bets", 0)),
                "ats_record": (
                    f"{int(summary.get('wins', 0))}-"
                    f"{int(summary.get('losses', 0))}-"
                    f"{int(summary.get('pushes', 0))}"
                ),
                "win_rate": summary.get("win_rate_ex_pushes", np.nan),
                "roi": summary.get("roi", np.nan),
                "mae_all_games": float(
                    mean_absolute_error(
                        group["home_margin"],
                        group["predicted_home_margin"],
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def print_report(
    c3,
    predictions: pd.DataFrame,
    threshold_report: pd.DataFrame,
    *,
    test_start: int,
    test_end: int,
):
    print("\n" + "=" * 88)
    print("NFL POINT SPREADS ANALYZER - OUT-OF-SAMPLE BACKTEST")
    print("=" * 88)
    print(f"Test seasons: {test_start}-{test_end}")
    print(f"Predictions:  {len(predictions):,}")

    if predictions.empty:
        print("No predictions were generated.")
        return

    overall = c3.regression_metrics(predictions)

    print(f"Overall margin MAE:  {overall['mae']:.3f} points")
    print(f"Overall margin RMSE: {overall['rmse']:.3f} points")

    print("\nPERFORMANCE BY MINIMUM MODEL EDGE")
    print("-" * 88)

    display = threshold_report.copy()
    display["win_rate"] = display["win_rate"].map(
        lambda x: f"{100*x:.2f}%" if pd.notna(x) else "N/A"
    )
    display["roi"] = display["roi"].map(
        lambda x: f"{100*x:.2f}%" if pd.notna(x) else "N/A"
    )
    display["profit_units"] = display["profit_units"].map(
        lambda x: f"{x:+.2f}"
    )
    display["mae_selected_games"] = display["mae_selected_games"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
    )
    display["avg_abs_model_edge"] = display["avg_abs_model_edge"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
    )
    display["avg_clv_points"] = display["avg_clv_points"].map(
        lambda x: f"{x:+.3f}" if pd.notna(x) else "N/A"
    )
    if "avg_decision_abs_edge" in display.columns:
        display["avg_decision_abs_edge"] = display[
            "avg_decision_abs_edge"
        ].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
        )
    if "key_cross_rate" in display.columns:
        display["key_cross_rate"] = display[
            "key_cross_rate"
        ].map(
            lambda x: f"{100*x:.1f}%" if pd.notna(x) else "N/A"
        )

    print(
        display[
            [
                "edge_threshold",
                "bets",
                "ats_record",
                "win_rate",
                "profit_units",
                "roi",
                "mae_selected_games",
                "avg_abs_model_edge",
                "avg_decision_abs_edge",
                "key_cross_rate",
                "avg_clv_points",
            ]
        ].to_string(index=False)
    )

    print("\nNOTE:")
    print(
        "  This is a kickoff-time model. When bet_spread_home equals the "
        "official closing spread, CLV is 0 by construction."
    )


def print_unified_multiyear_display(
    season_report: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Print 2024/2025/2026 in one horizontal terminal table."""
    wanted = [2024, 2025, 2026]
    seasons = [s for s in wanted if s in set(season_report["season"].astype(int))]
    if not seasons:
        return

    rows = []
    labels = [
        ("ATS Record", "ats_record"),
        ("Win Rate", "win_rate"),
        ("ROI", "roi"),
        ("Bets", "bets"),
        ("Predictions", "predictions"),
        ("MAE", "mae_all_games"),
    ]

    by_season = season_report.set_index("season")
    for label, col in labels:
        row = {"Metric": label}
        for season in seasons:
            value = by_season.loc[season, col]
            if col in {"win_rate", "roi"}:
                row[str(season)] = (
                    f"{100*float(value):.2f}%" if pd.notna(value) else "N/A"
                )
            elif col == "mae_all_games":
                row[str(season)] = (
                    f"{float(value):.3f}" if pd.notna(value) else "N/A"
                )
            else:
                row[str(season)] = str(value)
        rows.append(row)

    table = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print("UNIFIED OUT-OF-SAMPLE SEASON DISPLAY")
    print("=" * 92)
    print(table.to_string(index=False))

    # Margin MAE across all three seasons for context.
    selected = predictions[predictions["season"].isin(seasons)]
    if not selected.empty:
        combined_mae = mean_absolute_error(
            selected["home_margin"],
            selected["predicted_home_margin"],
        )
        print(f"\nCombined {seasons[0]}-{seasons[-1]} margin MAE: {combined_mae:.3f}")


# ---------------------------------------------------------------------------
# SAVE OUTPUTS
# ---------------------------------------------------------------------------

def save_outputs(
    output_dir: Path,
    *,
    current_snapshots: pd.DataFrame,
    blended_ratings: pd.DataFrame,
    model_dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold_report: pd.DataFrame,
    season_report: pd.DataFrame,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "component1_weekly_snapshots.parquet": current_snapshots,
        "component2_blended_ratings.parquet": blended_ratings,
        "model_dataset.parquet": model_dataset,
        "backtest_predictions.parquet": predictions,
    }

    for filename, frame in files.items():
        path = output_dir / filename
        frame.to_parquet(path, index=False)

    threshold_report.to_csv(
        output_dir / "threshold_report.csv",
        index=False,
    )
    season_report.to_csv(
        output_dir / "season_report.csv",
        index=False,
    )

    print("\nSaved:")
    for filename in files:
        print(f"  {output_dir / filename}")
    print(f"  {output_dir / 'threshold_report.csv'}")
    print(f"  {output_dir / 'season_report.csv'}")


# ---------------------------------------------------------------------------
# END-TO-END ORCHESTRATOR
# ---------------------------------------------------------------------------

def run_backtest(
    *,
    continuity_path: str,
    market_path: Optional[str],
    output_dir: str,
    thresholds: Sequence[float],
    config: BacktestConfig,
    component1_path: Optional[str] = None,
    component2_path: Optional[str] = None,
    component3_path: Optional[str] = None,
):
    # Validate season layout.
    if config.train_start > config.train_end:
        raise ValueError("train_start must be <= train_end.")
    if config.test_start > config.test_end:
        raise ValueError("test_start must be <= test_end.")
    if config.train_end >= config.test_start:
        raise ValueError(
            "Training and test spans overlap. "
            "Use train_end < test_start."
        )

    c1, c2, c3 = load_components(
        component1_path=component1_path,
        component2_path=component2_path,
        component3_path=component3_path,
    )

    model_seasons = list(
        range(config.train_start, config.test_end + 1)
    )

    # Need one extra historical season to supply the first target season's
    # prior-year baseline.
    feature_seasons = list(
        range(config.train_start - 1, config.test_end + 1)
    )

    print("\n[0/5] DATA INGESTION")
    print(
        f"  PBP seasons:   {feature_seasons[0]}-{feature_seasons[-1]}"
    )
    print(
        f"  Model seasons: {model_seasons[0]}-{model_seasons[-1]}"
    )
    print(
        f"  Train span:    {config.train_start}-{config.train_end}"
    )
    print(
        f"  Test span:     {config.test_start}-{config.test_end}"
    )

    # Component #1 owns nflreadpy PBP loading.
    pbp = c1.load_pbp(feature_seasons)

    raw_schedules = load_nflverse_schedules(model_seasons)
    schedules = prepare_schedule_table(
        raw_schedules,
        seasons=model_seasons,
    )

    # OPTION 2: scheduling / situational context.
    # These features use only information available before kickoff.
    context_features = load_context_feature_module()
    schedules = context_features.add_context_features(schedules)
    context_features.print_context_summary(schedules)

    continuity = load_continuity_table(
        continuity_path,
        seasons=model_seasons,
    )

    validate_continuity_coverage(
        schedules,
        continuity,
        target_seasons=model_seasons,
        allow_missing=config.allow_missing_continuity,
    )

    external_market = (
        read_table(market_path)
        if market_path is not None
        else None
    )

    # Components #1 and #2.
    _, current_snapshots, prior_baselines = build_component1_outputs(
        c1,
        pbp=pbp,
        all_feature_seasons=feature_seasons,
        config=config,
    )

    blended_ratings = build_component2_blends(
        c2,
        current_snapshots=current_snapshots,
        prior_baselines=prior_baselines,
        schedules=schedules,
        continuity=continuity,
        target_seasons=model_seasons,
        config=config,
    )

    # Final historical game + market table.
    print("\n[3/5] DATASET SYNTHESIS")
    games = build_game_table(
        schedules,
        model_seasons=model_seasons,
        external_market=external_market,
    )

    model_dataset = c3.synthesize_dataset(
        games=games,
        blended_team_ratings=blended_ratings,
        market=None,
    )

    required_context_features = {
        "net_rest",
        "travel_distance_diff",
        "is_divisional",
        "tz_shifted",
    }
    missing_context = required_context_features - set(model_dataset.columns)
    if missing_context:
        raise AssertionError(
            "Context features were lost before model_dataset creation: "
            + ", ".join(sorted(missing_context))
        )

    # Do not allow rows outside the requested model span.
    model_dataset = model_dataset[
        model_dataset["season"].between(
            config.train_start,
            config.test_end,
        )
    ].copy()

    features = choose_model_features(c3, model_dataset)

    # Warn rather than silently train if major features are entirely absent.
    print(f"  Historical games: {len(model_dataset):,}")
    print(f"  Model features:   {len(features):,}")

    missing_feature_rows = model_dataset[features].isna().all(axis=1).sum()
    if missing_feature_rows:
        raise ValueError(
            f"{missing_feature_rows} game rows have every model feature missing."
        )

    # Component #3 weekly walk-forward.
    predictions = weekly_walk_forward_predict(
        c3,
        model_dataset,
        feature_columns=features,
        test_start=config.test_start,
        test_end=config.test_end,
        config=config,
    )

    if predictions.empty:
        raise RuntimeError(
            "No out-of-sample predictions were generated. "
            "Check min_train_games and input coverage."
        )

    # Final leakage assertion: output must contain only requested test seasons.
    if not predictions["season"].between(
        config.test_start,
        config.test_end,
    ).all():
        raise AssertionError("Prediction output contains non-test seasons.")

    print("\n[5/5] REPORTING")

    active_trends = calculate_active_2026_trends(
        model_dataset,
        active_season=config.active_season,
        target_week=None,
        lookback_weeks=config.market_bias_lookback_weeks,
    )
    historical_environment = calculate_historical_environment_baseline(
        model_dataset,
        before_season=config.active_season,
    )

    environment_report = pd.DataFrame(
        [
            {"environment": "active_2026", **active_trends},
            {"environment": "historical_baseline", **historical_environment},
        ]
    )

    threshold_report = threshold_performance_report(
        c3,
        predictions,
        thresholds=thresholds,
        default_american_odds=config.default_american_odds,
    )

    primary_threshold = float(thresholds[0])

    season_report = per_season_report(
        c3,
        predictions,
        threshold=primary_threshold,
        default_american_odds=config.default_american_odds,
    )

    print_report(
        c3,
        predictions,
        threshold_report,
        test_start=config.test_start,
        test_end=config.test_end,
    )

    print(
        f"\nPER-SEASON PERFORMANCE @ {primary_threshold:.2f}-POINT EDGE"
    )
    print("-" * 88)

    season_display = season_report.copy()
    season_display["win_rate"] = season_display["win_rate"].map(
        lambda x: f"{100*x:.2f}%" if pd.notna(x) else "N/A"
    )
    season_display["roi"] = season_display["roi"].map(
        lambda x: f"{100*x:.2f}%" if pd.notna(x) else "N/A"
    )
    season_display["mae_all_games"] = season_display["mae_all_games"].map(
        lambda x: f"{x:.3f}"
    )
    print(season_display.to_string(index=False))

    print_unified_multiyear_display(
        season_report,
        predictions,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    environment_report.to_csv(
        Path(output_dir) / "active_2026_environment.csv",
        index=False,
    )

    save_outputs(
        Path(output_dir),
        current_snapshots=current_snapshots,
        blended_ratings=blended_ratings,
        model_dataset=model_dataset,
        predictions=predictions,
        threshold_report=threshold_report,
        season_report=season_report,
    )

    return {
        "current_snapshots": current_snapshots,
        "blended_ratings": blended_ratings,
        "model_dataset": model_dataset,
        "predictions": predictions,
        "threshold_report": threshold_report,
        "season_report": season_report,
        "active_2026_environment": environment_report,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete NFL Point Spreads Analyzer backtest."
    )

    parser.add_argument(
        "--continuity",
        required=True,
        help=(
            "CSV/Parquet continuity table with season, team, "
            "hc_retained, oc_retained, qb_retained."
        ),
    )

    parser.add_argument(
        "--market",
        default=None,
        help=(
            "Optional normalized external market CSV/Parquet. "
            "If omitted, nflverse schedule spread_line/total_line are used."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="data/production_ridge_2026",
    )

    parser.add_argument("--train-start", type=int, default=2018)
    parser.add_argument("--train-end", type=int, default=2023)
    parser.add_argument("--test-start", type=int, default=2024)
    parser.add_argument("--test-end", type=int, default=2026)

    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=250.0,
        help=(
            "L2 regularization strength for the final score-margin Ridge model. "
            "Higher values produce more coefficient shrinkage."
        ),
    )
    parser.add_argument(
        "--offensive-epa-elasticity",
        type=float,
        default=1.25,
        help="Post-standardization elasticity multiplier for offensive EPA features.",
    )
    parser.add_argument(
        "--defensive-epa-elasticity",
        type=float,
        default=0.90,
        help="Post-standardization elasticity multiplier for defensive EPA features.",
    )
    parser.add_argument(
        "--recent-game-weight",
        type=float,
        default=2.0,
        help="Weight assigned to the two most recent completed target-season weeks.",
    )
    parser.add_argument(
        "--recent-game-count",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--recent-weight-half-life-games",
        type=float,
        default=2.0,
        help="Half-life for older current-season sample-weight premium.",
    )
    parser.add_argument(
        "--garbage-time-wp-low",
        type=float,
        default=0.05,
        help="Remove Q3/Q4 plays when live WP is below this value.",
    )
    parser.add_argument(
        "--garbage-time-wp-high",
        type=float,
        default=0.95,
        help="Remove Q3/Q4 plays when live WP is above this value.",
    )
    parser.add_argument(
        "--sos-iterations",
        type=int,
        default=2,
        help="Number of leakage-safe defensive-rank SOS smoothing passes.",
    )
    parser.add_argument(
        "--sos-rank-weight-strength",
        type=float,
        default=0.15,
        help="Maximum centered rank influence before SOS weight clipping.",
    )

    parser.add_argument(
        "--turnover-regression-strength",
        type=float,
        default=0.50,
        help="Fraction of game-level turnover differential regressed toward league mean.",
    )

    parser.add_argument(
        "--market-anchor-weight",
        type=float,
        default=0.05,
        help=(
            "Final-margin weight assigned to the sportsbook-implied margin. "
            "Default 0.05 = 95%% Ridge / 5%% market."
        ),
    )
    parser.add_argument(
        "--heavy-spread-threshold",
        type=float,
        default=7.0,
        help="Spread magnitude above which final-model Ridge alpha increases.",
    )
    parser.add_argument(
        "--alpha-growth-per-point",
        type=float,
        default=0.25,
        help="Additional alpha multiplier per point above the heavy-spread threshold.",
    )
    parser.add_argument(
        "--max-alpha-multiplier",
        type=float,
        default=3.0,
        help="Cap on spread-dependent Ridge alpha scaling.",
    )
    parser.add_argument(
        "--market-bias-lookback-weeks",
        type=int,
        default=4,
        help=(
            "Rolling market-bias lookback window in completed weeks. "
            "Default: 4."
        ),
    )
    parser.add_argument(
        "--active-season",
        type=int,
        default=2026,
        help="NFL season used by the active-environment tracker.",
    )
    parser.add_argument(
        "--active-trend-prior-games",
        type=float,
        default=64.0,
        help="Pseudo-games used to shrink the rolling active-season trend.",
    )
    parser.add_argument(
        "--active-hfa-weight",
        type=float,
        default=0.50,
        help="Weight applied to active-vs-historical home-margin change.",
    )
    parser.add_argument(
        "--active-cover-balance-points",
        type=float,
        default=3.0,
        help="Point conversion for active favorite/underdog ATS balance.",
    )
    parser.add_argument(
        "--active-max-intercept-shift",
        type=float,
        default=1.25,
        help="Hard cap in points on the active-season intercept nudge.",
    )

    parser.add_argument(
        "--min-train-games",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--max-train-seasons",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--rolling-weeks",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--recency-half-life",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--team-ridge-alpha-epa",
        type=float,
        default=200.0,
    )
    parser.add_argument(
        "--team-ridge-alpha-success",
        type=float,
        default=200.0,
    )

    parser.add_argument(
        "--continuity-mode",
        choices=("global", "unit_specific"),
        default="global",
    )
    parser.add_argument(
        "--hard-zero-on-any-change",
        action="store_true",
    )
    parser.add_argument(
        "--allow-missing-continuity",
        action="store_true",
        help=(
            "If used, Component #2 treats missing continuity conservatively "
            "instead of failing."
        ),
    )

    parser.add_argument(
        "--default-american-odds",
        type=float,
        default=-110.0,
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 2.5, 3.0],
        help="Minimum absolute model edges to report.",
    )

    parser.add_argument(
        "--component1",
        default=None,
        help="Optional explicit path to Component #1.",
    )
    parser.add_argument(
        "--component2",
        default=None,
        help="Optional explicit path to Component #2.",
    )
    parser.add_argument(
        "--component3",
        default=None,
        help="Optional explicit path to Component #3.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = BacktestConfig(
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        rolling_weeks=args.rolling_weeks,
        recency_half_life_weeks=args.recency_half_life,
        ridge_alpha_team_epa=args.team_ridge_alpha_epa,
        ridge_alpha_team_success=args.team_ridge_alpha_success,
        garbage_time_wp_low=args.garbage_time_wp_low,
        garbage_time_wp_high=args.garbage_time_wp_high,
        turnover_regression_strength=args.turnover_regression_strength,
        sos_iterations=args.sos_iterations,
        sos_rank_weight_strength=args.sos_rank_weight_strength,
        model_ridge_alpha=args.ridge_alpha,
        offensive_epa_elasticity=args.offensive_epa_elasticity,
        defensive_epa_elasticity=args.defensive_epa_elasticity,
        recent_game_weight=args.recent_game_weight,
        recent_game_count=args.recent_game_count,
        recent_weight_half_life_games=args.recent_weight_half_life_games,
        heavy_spread_threshold=args.heavy_spread_threshold,
        alpha_growth_per_point=args.alpha_growth_per_point,
        max_alpha_multiplier=args.max_alpha_multiplier,
        market_anchor_weight=args.market_anchor_weight,
        market_bias_lookback_weeks=args.market_bias_lookback_weeks,
        active_season=args.active_season,
        active_trend_prior_games=args.active_trend_prior_games,
        active_hfa_weight=args.active_hfa_weight,
        active_cover_balance_points=args.active_cover_balance_points,
        active_max_intercept_shift=args.active_max_intercept_shift,
        min_train_games=args.min_train_games,
        max_train_seasons=args.max_train_seasons,
        continuity_mode=args.continuity_mode,
        hard_zero_on_any_change=args.hard_zero_on_any_change,
        allow_missing_continuity=args.allow_missing_continuity,
        default_american_odds=args.default_american_odds,
    )

    run_backtest(
        continuity_path=args.continuity,
        market_path=args.market,
        output_dir=args.output_dir,
        thresholds=args.thresholds,
        config=config,
        component1_path=args.component1,
        component2_path=args.component2,
        component3_path=args.component3,
    )


if __name__ == "__main__":
    main()
