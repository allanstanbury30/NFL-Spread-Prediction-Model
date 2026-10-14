"""
Component #2: Prior-Season + Continuity Blender
================================================

Purpose
-------
Blend current-season opponent-adjusted efficiency metrics from Component #1
with the prior season's baseline in a strictly time-aware way.

Core rules
----------
1. Week 1 starts at 80% prior season / 20% current-or-fallback baseline.
2. Prior-season scheduled weight decays linearly to 0% by Week 8.
3. Continuity changes (HC / OC / QB) reduce the *effective* prior-season weight.
4. Any weight removed by continuity DOES NOT automatically transfer to the
   tiny current-season sample. It falls back to a league/preseason baseline.
5. Missing current or prior metrics also fall back to the baseline.
6. Output includes diagnostics so every blended number is auditable.

Expected Component #1 columns
-----------------------------
season
as_of_week
team
adj_off_epa
adj_def_epa_allowed
adj_off_success_rate
adj_def_success_rate_allowed

The same metric columns are expected in the prior-season baseline table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union, Literal

import numpy as np
import pandas as pd


DEFAULT_METRICS: Tuple[str, ...] = (
    "adj_off_epa",
    "adj_def_epa_allowed",
    "adj_off_success_rate",
    "adj_def_success_rate_allowed",
)


@dataclass(frozen=True)
class BlendConfig:
    """
    Configuration for the prior/current-season blending process.

    The change multipliers are deliberately configurable. They are starting
    assumptions only and should later be tuned with walk-forward validation.
    """
    week1_prior_weight: float = 0.80
    prior_weight_zero_week: int = 8

    # Starting assumptions only. Tune later with walk-forward validation.
    hc_change_multiplier: float = 0.50
    oc_change_multiplier: float = 0.50
    qb_change_multiplier: float = 0.25

    # "global" follows the strict spec: HC/OC/QB changes discount both units.
    # "unit_specific" is available later:
    #   offense -> HC, OC, QB
    #   defense -> HC only (until DC continuity is added).
    continuity_mode: Literal["global", "unit_specific"] = "global"

    # Optional hard switch: if True, ANY HC/OC/QB change sets prior weight to 0.
    hard_zero_on_any_change: bool = False

    # Fail loudly if a team lacks continuity data.
    require_continuity_record: bool = True


def scheduled_prior_weight(week: int, config: BlendConfig = BlendConfig()) -> float:
    """
    Scheduled prior-season weight before continuity adjustment.

    Week 1 -> 0.80
    Week 8+ -> 0.00
    """
    if week < 1:
        raise ValueError("week must be >= 1")

    if week >= config.prior_weight_zero_week:
        return 0.0

    span = config.prior_weight_zero_week - 1
    remaining = config.prior_weight_zero_week - week
    return float(config.week1_prior_weight * (remaining / span))


def scheduled_current_weight(week: int, config: BlendConfig = BlendConfig()) -> float:
    """Current-season scheduled weight; complements the prior-season schedule."""
    return 1.0 - scheduled_prior_weight(week, config)


def _retained_to_bool(value) -> bool:
    """Normalize common retained/not-retained encodings to bool."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if pd.isna(value):
        raise ValueError("Continuity value is missing.")

    if isinstance(value, (int, float, np.integer, np.floating)):
        if value in (0, 1):
            return bool(value)

    text = str(value).strip().lower()
    truthy = {"1", "true", "t", "yes", "y", "retained", "same"}
    falsy = {"0", "false", "f", "no", "n", "changed", "new"}

    if text in truthy:
        return True
    if text in falsy:
        return False

    raise ValueError(f"Cannot interpret continuity value: {value!r}")


def normalize_continuity(
    continuity: Union[pd.DataFrame, Mapping[str, Mapping[str, object]]],
    season: Optional[int] = None,
) -> pd.DataFrame:
    """
    Accept either a DataFrame or a dictionary keyed by team.

    Required fields:
        team
        hc_retained
        oc_retained
        qb_retained
    """
    if isinstance(continuity, pd.DataFrame):
        df = continuity.copy()
    else:
        rows = []
        for team, flags in continuity.items():
            row = {"team": team, **dict(flags)}
            if season is not None:
                row["season"] = season
            rows.append(row)
        df = pd.DataFrame(rows)

    required = {"team", "hc_retained", "oc_retained", "qb_retained"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Continuity table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if season is not None and "season" in df.columns:
        df = df[df["season"] == season].copy()

    if df["team"].duplicated().any():
        dupes = df.loc[df["team"].duplicated(keep=False), "team"].tolist()
        raise ValueError(f"Duplicate continuity rows for team(s): {sorted(set(dupes))}")

    for col in ("hc_retained", "oc_retained", "qb_retained"):
        df[col] = df[col].map(_retained_to_bool)

    return df.reset_index(drop=True)


def continuity_multipliers(
    row: pd.Series,
    config: BlendConfig = BlendConfig(),
) -> Tuple[float, float]:
    """
    Return (offense_multiplier, defense_multiplier).

    In "global" mode:
        HC, OC, and QB continuity affect both units.

    In "unit_specific" mode:
        offense -> HC, OC, QB
        defense -> HC only
    """
    hc_retained = bool(row["hc_retained"])
    oc_retained = bool(row["oc_retained"])
    qb_retained = bool(row["qb_retained"])

    any_change = not (hc_retained and oc_retained and qb_retained)

    if config.hard_zero_on_any_change and any_change:
        return 0.0, 0.0

    hc_factor = 1.0 if hc_retained else config.hc_change_multiplier
    oc_factor = 1.0 if oc_retained else config.oc_change_multiplier
    qb_factor = 1.0 if qb_retained else config.qb_change_multiplier

    offense = hc_factor * oc_factor * qb_factor

    if config.continuity_mode == "global":
        defense = offense
    elif config.continuity_mode == "unit_specific":
        defense = hc_factor
    else:
        raise ValueError(
            f"Unknown continuity_mode={config.continuity_mode!r}. "
            "Use 'global' or 'unit_specific'."
        )

    return float(offense), float(defense)


def select_prior_season_baseline(
    snapshots: pd.DataFrame,
    target_season: int,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """
    Select the latest available snapshot from the immediately prior season,
    separately for each team.
    """
    prior_season = target_season - 1

    required = {"season", "team", "as_of_week", *metrics}
    missing = required - set(snapshots.columns)
    if missing:
        raise ValueError(
            "Snapshot table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    prior = snapshots[snapshots["season"] == prior_season].copy()
    if prior.empty:
        raise ValueError(f"No prior-season snapshots found for {prior_season}.")

    prior = (
        prior.sort_values(["team", "as_of_week"])
        .groupby("team", as_index=False)
        .tail(1)
    )

    return prior[["team", *metrics]].reset_index(drop=True)


def infer_league_baseline(
    prior_baseline: pd.DataFrame,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> Dict[str, float]:
    """
    Infer a time-safe fallback baseline from prior-season team averages.

    A separate preseason projection may be supplied instead, provided it is
    expressed on the same metric scale.
    """
    baseline = {}
    for metric in metrics:
        if metric not in prior_baseline.columns:
            raise ValueError(f"Prior baseline missing metric: {metric}")
        value = pd.to_numeric(prior_baseline[metric], errors="coerce").mean()
        if pd.isna(value):
            raise ValueError(f"Could not infer baseline for metric: {metric}")
        baseline[metric] = float(value)
    return baseline


def _blend_one_value(
    current_value: float,
    prior_value: float,
    fallback_value: float,
    scheduled_prior: float,
    scheduled_current: float,
    continuity_multiplier: float,
) -> Tuple[float, float, float, float]:
    """
    Blend one metric.

    Any prior-season weight removed by continuity falls to the fallback baseline
    rather than being transferred to a tiny current-season sample.
    """
    prior_available = pd.notna(prior_value)
    current_available = pd.notna(current_value)

    effective_prior = (
        scheduled_prior * continuity_multiplier if prior_available else 0.0
    )
    effective_current = scheduled_current if current_available else 0.0

    fallback_weight = 1.0 - effective_prior - effective_current
    fallback_weight = max(0.0, min(1.0, fallback_weight))

    blended = (
        effective_prior * (prior_value if prior_available else 0.0)
        + effective_current * (current_value if current_available else 0.0)
        + fallback_weight * fallback_value
    )

    return (
        float(blended),
        float(effective_prior),
        float(effective_current),
        float(fallback_weight),
    )


def blend_team_ratings(
    current_snapshots: pd.DataFrame,
    prior_baseline: pd.DataFrame,
    continuity: Union[pd.DataFrame, Mapping[str, Mapping[str, object]]],
    *,
    season: int,
    week: int,
    metrics: Sequence[str] = DEFAULT_METRICS,
    fallback_baseline: Optional[Mapping[str, float]] = None,
    config: BlendConfig = BlendConfig(),
) -> pd.DataFrame:
    """
    Produce blended offensive/defensive team ratings for one target game week.

    `current_snapshots` must be Component #1 output where the Week N snapshot
    contains information only through Week N-1.
    """
    if week < 1:
        raise ValueError("week must be >= 1")

    current_required = {"season", "as_of_week", "team", *metrics}
    missing = current_required - set(current_snapshots.columns)
    if missing:
        raise ValueError(
            "Current snapshot table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    prior_required = {"team", *metrics}
    missing = prior_required - set(prior_baseline.columns)
    if missing:
        raise ValueError(
            "Prior baseline table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    current = current_snapshots[
        (current_snapshots["season"] == season)
        & (current_snapshots["as_of_week"] == week)
    ].copy()

    cont = normalize_continuity(continuity, season=season)
    if "season" in cont.columns:
        cont = cont.drop(columns=["season"])

    team_universe = sorted(
        set(prior_baseline["team"].astype(str))
        | set(cont["team"].astype(str))
        | set(current["team"].astype(str))
    )

    base = pd.DataFrame({"team": team_universe})

    prior = prior_baseline[["team", *metrics]].copy()
    prior = prior.rename(columns={m: f"prior_{m}" for m in metrics})

    if current.empty:
        current = pd.DataFrame(columns=["team", *metrics])
    else:
        current = current[["team", *metrics]].copy()

    current = current.rename(columns={m: f"current_{m}" for m in metrics})

    out = (
        base
        .merge(prior, on="team", how="left", validate="one_to_one")
        .merge(current, on="team", how="left", validate="one_to_one")
        .merge(cont, on="team", how="left", validate="one_to_one")
    )

    if config.require_continuity_record:
        continuity_cols = ["hc_retained", "oc_retained", "qb_retained"]
        bad = out[continuity_cols].isna().any(axis=1)
        if bad.any():
            teams = out.loc[bad, "team"].tolist()
            raise ValueError(
                "Missing continuity records for team(s): " + ", ".join(teams)
            )
    else:
        for col in ("hc_retained", "oc_retained", "qb_retained"):
            out[col] = out[col].fillna(False)

    scheduled_prior = scheduled_prior_weight(week, config)
    scheduled_current = scheduled_current_weight(week, config)

    out.insert(0, "season", season)
    out.insert(1, "week", week)
    out["scheduled_prior_weight"] = scheduled_prior
    out["scheduled_current_weight"] = scheduled_current

    multipliers = out.apply(
        lambda r: continuity_multipliers(r, config),
        axis=1,
        result_type="expand",
    )
    multipliers.columns = ["off_continuity_multiplier", "def_continuity_multiplier"]
    out = pd.concat([out, multipliers], axis=1)

    if fallback_baseline is None:
        fallback_baseline = infer_league_baseline(prior_baseline, metrics)

    missing_fallback = set(metrics) - set(fallback_baseline.keys())
    if missing_fallback:
        raise ValueError(
            "Fallback baseline missing metric(s): "
            + ", ".join(sorted(missing_fallback))
        )

    offense_metrics = {"adj_off_epa", "adj_off_success_rate"}

    for metric in metrics:
        unit_multiplier_col = (
            "off_continuity_multiplier"
            if metric in offense_metrics
            else "def_continuity_multiplier"
        )

        blended_values = []
        prior_weights = []
        current_weights = []
        fallback_weights = []

        for _, row in out.iterrows():
            blended, wp, wc, wb = _blend_one_value(
                current_value=row[f"current_{metric}"],
                prior_value=row[f"prior_{metric}"],
                fallback_value=float(fallback_baseline[metric]),
                scheduled_prior=scheduled_prior,
                scheduled_current=scheduled_current,
                continuity_multiplier=float(row[unit_multiplier_col]),
            )

            blended_values.append(blended)
            prior_weights.append(wp)
            current_weights.append(wc)
            fallback_weights.append(wb)

        out[f"blended_{metric}"] = blended_values
        out[f"{metric}_effective_prior_weight"] = prior_weights
        out[f"{metric}_effective_current_weight"] = current_weights
        out[f"{metric}_fallback_weight"] = fallback_weights

        total_weight = (
            out[f"{metric}_effective_prior_weight"]
            + out[f"{metric}_effective_current_weight"]
            + out[f"{metric}_fallback_weight"]
        )
        if not np.allclose(total_weight.to_numpy(dtype=float), 1.0, atol=1e-10):
            raise AssertionError(f"Weights do not sum to 1 for metric {metric}")

    return out.sort_values("team").reset_index(drop=True)


def attach_blended_features_to_games(
    games: pd.DataFrame,
    blended_team_ratings: pd.DataFrame,
    *,
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
) -> pd.DataFrame:
    """
    Merge blended team ratings onto one-row-per-game training data.
    """
    required_game = {"season", "week", home_team_col, away_team_col}
    missing = required_game - set(games.columns)
    if missing:
        raise ValueError(
            "Games table missing required columns: "
            + ", ".join(sorted(missing))
        )

    blended_cols = [
        c for c in blended_team_ratings.columns if c.startswith("blended_")
    ]

    home = blended_team_ratings[
        ["season", "week", "team", *blended_cols]
    ].rename(
        columns={
            "team": home_team_col,
            **{c: f"home_{c}" for c in blended_cols},
        }
    )

    away = blended_team_ratings[
        ["season", "week", "team", *blended_cols]
    ].rename(
        columns={
            "team": away_team_col,
            **{c: f"away_{c}" for c in blended_cols},
        }
    )

    out = games.merge(
        home,
        on=["season", "week", home_team_col],
        how="left",
        validate="many_to_one",
    )

    out = out.merge(
        away,
        on=["season", "week", away_team_col],
        how="left",
        validate="many_to_one",
    )

    return out


def print_weight_schedule(config: BlendConfig = BlendConfig()) -> None:
    """Quick visual sanity check for the decay schedule."""
    rows = []
    for week in range(1, config.prior_weight_zero_week + 1):
        rows.append(
            {
                "week": week,
                "prior": scheduled_prior_weight(week, config),
                "current": scheduled_current_weight(week, config),
            }
        )

    table = pd.DataFrame(rows)
    print("\nPrior/current scheduled weights:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))


def demo() -> None:
    """Synthetic demonstration of continuity discounting."""
    season = 2026
    week = 3

    prior = pd.DataFrame(
        {
            "team": ["BUF", "NE", "CHI"],
            "adj_off_epa": [0.16, 0.02, -0.08],
            "adj_def_epa_allowed": [-0.06, -0.01, 0.04],
            "adj_off_success_rate": [0.49, 0.44, 0.40],
            "adj_def_success_rate_allowed": [0.41, 0.44, 0.47],
        }
    )

    current = pd.DataFrame(
        {
            "season": [season] * 3,
            "as_of_week": [week] * 3,
            "team": ["BUF", "NE", "CHI"],
            "adj_off_epa": [0.12, 0.08, 0.05],
            "adj_def_epa_allowed": [-0.04, -0.03, -0.02],
            "adj_off_success_rate": [0.47, 0.46, 0.45],
            "adj_def_success_rate_allowed": [0.42, 0.43, 0.42],
        }
    )

    continuity = {
        "BUF": {
            "hc_retained": True,
            "oc_retained": True,
            "qb_retained": True,
        },
        "NE": {
            "hc_retained": True,
            "oc_retained": True,
            "qb_retained": False,
        },
        "CHI": {
            "hc_retained": False,
            "oc_retained": False,
            "qb_retained": False,
        },
    }

    blended = blend_team_ratings(
        current_snapshots=current,
        prior_baseline=prior,
        continuity=continuity,
        season=season,
        week=week,
    )

    cols = [
        "team",
        "scheduled_prior_weight",
        "scheduled_current_weight",
        "off_continuity_multiplier",
        "blended_adj_off_epa",
        "blended_adj_def_epa_allowed",
        "blended_adj_off_success_rate",
        "blended_adj_def_success_rate_allowed",
    ]

    print_weight_schedule()
    print("\nSynthetic Week 3 blend:")
    print(blended[cols].to_string(index=False))


if __name__ == "__main__":
    demo()
