#!/usr/bin/env python3
"""
schedule_context_features.py
============================

Contextual scheduling / situational features for the NFL Point Spreads Analyzer.

Adds, before kickoff:
    net_rest
        Home rest days minus Away rest days.

    away_travel_miles
        Great-circle miles from the Away team's season-appropriate home venue
        to the game venue.

    home_travel_miles
        Normally 0 for true home games. Non-zero for recognized neutral-site
        games.

    travel_distance_diff
        away_travel_miles - home_travel_miles.
        Positive means the Away team traveled farther.

    is_divisional
        1 when both clubs are in the same NFL division, else 0.

    tz_shifted
        Absolute body-clock shift in hours between the Away team's home time
        zone and the game venue at kickoff. Because this uses actual UTC offsets
        at the kickoff timestamp, Arizona/DST is handled correctly.

Design
------
The module uses no outcome information. Every feature is available before
kickoff and is therefore safe for historical walk-forward forecasting.

Travel is an approximation based on great-circle stadium-to-stadium distance,
not actual charter routing. This is intentional: it is stable, reproducible,
and available historically without post-game data.

Historical venue overrides are included for:
    Raiders 2018-2019  -> Oakland Coliseum
    Chargers 2018-2019 -> Dignity Health Sports Park
    Rams 2018-2019     -> Los Angeles Memorial Coliseum

Common international / neutral NFL venues are also recognized when the
schedule's stadium field contains a known venue name.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import radians, sin, cos, asin, sqrt
from typing import Dict, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# TEAM / DIVISION METADATA
# ---------------------------------------------------------------------------

TEAM_ALIASES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "LAR": "LA",
    "JAX": "JAC",
    "WSH": "WAS",
    "ARZ": "ARI",
}

DIVISIONS: Dict[str, str] = {
    # AFC East
    "BUF": "AFC_EAST",
    "MIA": "AFC_EAST",
    "NE": "AFC_EAST",
    "NYJ": "AFC_EAST",

    # AFC North
    "BAL": "AFC_NORTH",
    "CIN": "AFC_NORTH",
    "CLE": "AFC_NORTH",
    "PIT": "AFC_NORTH",

    # AFC South
    "HOU": "AFC_SOUTH",
    "IND": "AFC_SOUTH",
    "JAC": "AFC_SOUTH",
    "TEN": "AFC_SOUTH",

    # AFC West
    "DEN": "AFC_WEST",
    "KC": "AFC_WEST",
    "LAC": "AFC_WEST",
    "LV": "AFC_WEST",

    # NFC East
    "DAL": "NFC_EAST",
    "NYG": "NFC_EAST",
    "PHI": "NFC_EAST",
    "WAS": "NFC_EAST",

    # NFC North
    "CHI": "NFC_NORTH",
    "DET": "NFC_NORTH",
    "GB": "NFC_NORTH",
    "MIN": "NFC_NORTH",

    # NFC South
    "ATL": "NFC_SOUTH",
    "CAR": "NFC_SOUTH",
    "NO": "NFC_SOUTH",
    "TB": "NFC_SOUTH",

    # NFC West
    "ARI": "NFC_WEST",
    "LA": "NFC_WEST",
    "SF": "NFC_WEST",
    "SEA": "NFC_WEST",
}


@dataclass(frozen=True)
class Venue:
    name: str
    latitude: float
    longitude: float
    timezone: str


# Approximate stadium coordinates are sufficient for stable great-circle
# distance features. Shared stadiums intentionally share coordinates.
CURRENT_HOME_VENUES: Dict[str, Venue] = {
    "ARI": Venue("State Farm Stadium", 33.5276, -112.2626, "America/Phoenix"),
    "ATL": Venue("Mercedes-Benz Stadium", 33.7554, -84.4008, "America/New_York"),
    "BAL": Venue("M&T Bank Stadium", 39.2780, -76.6227, "America/New_York"),
    "BUF": Venue("Highmark Stadium", 42.7738, -78.7868, "America/New_York"),
    "CAR": Venue("Bank of America Stadium", 35.2258, -80.8528, "America/New_York"),
    "CHI": Venue("Soldier Field", 41.8623, -87.6167, "America/Chicago"),
    "CIN": Venue("Paycor Stadium", 39.0954, -84.5160, "America/New_York"),
    "CLE": Venue("Cleveland Browns Stadium", 41.5061, -81.6995, "America/New_York"),
    "DAL": Venue("AT&T Stadium", 32.7473, -97.0945, "America/Chicago"),
    "DEN": Venue("Empower Field at Mile High", 39.7439, -105.0201, "America/Denver"),
    "DET": Venue("Ford Field", 42.3400, -83.0456, "America/New_York"),
    "GB": Venue("Lambeau Field", 44.5013, -88.0622, "America/Chicago"),
    "HOU": Venue("NRG Stadium", 29.6847, -95.4107, "America/Chicago"),
    "IND": Venue("Lucas Oil Stadium", 39.7601, -86.1639, "America/New_York"),
    "JAC": Venue("EverBank Stadium", 30.3239, -81.6373, "America/New_York"),
    "KC": Venue("Arrowhead Stadium", 39.0489, -94.4839, "America/Chicago"),
    "LAC": Venue("SoFi Stadium", 33.9535, -118.3392, "America/Los_Angeles"),
    "LA": Venue("SoFi Stadium", 33.9535, -118.3392, "America/Los_Angeles"),
    "LV": Venue("Allegiant Stadium", 36.0908, -115.1830, "America/Los_Angeles"),
    "MIA": Venue("Hard Rock Stadium", 25.9580, -80.2389, "America/New_York"),
    "MIN": Venue("U.S. Bank Stadium", 44.9738, -93.2581, "America/Chicago"),
    "NE": Venue("Gillette Stadium", 42.0909, -71.2643, "America/New_York"),
    "NO": Venue("Caesars Superdome", 29.9511, -90.0812, "America/Chicago"),
    "NYG": Venue("MetLife Stadium", 40.8135, -74.0745, "America/New_York"),
    "NYJ": Venue("MetLife Stadium", 40.8135, -74.0745, "America/New_York"),
    "PHI": Venue("Lincoln Financial Field", 39.9008, -75.1675, "America/New_York"),
    "PIT": Venue("Acrisure Stadium", 40.4468, -80.0158, "America/New_York"),
    "SEA": Venue("Lumen Field", 47.5952, -122.3316, "America/Los_Angeles"),
    "SF": Venue("Levi's Stadium", 37.4030, -121.9700, "America/Los_Angeles"),
    "TB": Venue("Raymond James Stadium", 27.9759, -82.5033, "America/New_York"),
    "TEN": Venue("Nissan Stadium", 36.1665, -86.7713, "America/Chicago"),
    "WAS": Venue("Northwest Stadium", 38.9078, -76.8645, "America/New_York"),
}


# Historical team-home venue overrides for our 2018-2025 backtest.
# Key = (team, first_season, last_season)
HISTORICAL_HOME_VENUES: Dict[Tuple[str, int, int], Venue] = {
    ("LV", 2018, 2019): Venue(
        "Oakland-Alameda County Coliseum",
        37.7516,
        -122.2005,
        "America/Los_Angeles",
    ),
    ("LAC", 2018, 2019): Venue(
        "Dignity Health Sports Park",
        33.8644,
        -118.2611,
        "America/Los_Angeles",
    ),
    ("LA", 2018, 2019): Venue(
        "Los Angeles Memorial Coliseum",
        34.0141,
        -118.2879,
        "America/Los_Angeles",
    ),
}


# Known neutral / international venues. Matching is substring-based after
# lowercasing the schedule stadium name.
NEUTRAL_VENUE_PATTERNS: Dict[str, Venue] = {
    "tottenham": Venue(
        "Tottenham Hotspur Stadium",
        51.6043,
        -0.0664,
        "Europe/London",
    ),
    "wembley": Venue(
        "Wembley Stadium",
        51.5560,
        -0.2796,
        "Europe/London",
    ),
    "azteca": Venue(
        "Estadio Azteca",
        19.3029,
        -99.1505,
        "America/Mexico_City",
    ),
    "allianz arena": Venue(
        "Allianz Arena",
        48.2188,
        11.6247,
        "Europe/Berlin",
    ),
    "deutsche bank": Venue(
        "Deutsche Bank Park",
        50.0686,
        8.6455,
        "Europe/Berlin",
    ),
    "neo quimica": Venue(
        "Neo Quimica Arena",
        -23.5453,
        -46.4742,
        "America/Sao_Paulo",
    ),
    "corinthians": Venue(
        "Neo Quimica Arena",
        -23.5453,
        -46.4742,
        "America/Sao_Paulo",
    ),
    "croke park": Venue(
        "Croke Park",
        53.3607,
        -6.2511,
        "Europe/Dublin",
    ),
    "bernabeu": Venue(
        "Santiago Bernabeu Stadium",
        40.4531,
        -3.6883,
        "Europe/Madrid",
    ),
    "olympiastadion berlin": Venue(
        "Olympiastadion Berlin",
        52.5147,
        13.2395,
        "Europe/Berlin",
    ),
}


def canonical_team(team: object) -> str:
    code = str(team).strip().upper()
    return TEAM_ALIASES.get(code, code)


def home_venue_for_team(team: str, season: int) -> Venue:
    """Return the season-appropriate home venue for a franchise."""
    team = canonical_team(team)
    season = int(season)

    for (hist_team, first, last), venue in HISTORICAL_HOME_VENUES.items():
        if team == hist_team and first <= season <= last:
            return venue

    try:
        return CURRENT_HOME_VENUES[team]
    except KeyError as exc:
        raise KeyError(f"No venue mapping for team {team!r}") from exc


def neutral_venue_from_stadium_name(stadium_name: object) -> Optional[Venue]:
    if stadium_name is None or pd.isna(stadium_name):
        return None

    text = str(stadium_name).strip().lower()

    for pattern, venue in NEUTRAL_VENUE_PATTERNS.items():
        if pattern in text:
            return venue

    return None


def game_venue(
    row: pd.Series,
    *,
    stadium_column_candidates: Sequence[str] = (
        "stadium",
        "stadium_name",
        "venue",
    ),
) -> Tuple[Venue, bool]:
    """
    Resolve actual game venue.

    A recognized neutral/international stadium overrides the home team's
    normal venue. Otherwise the season-appropriate home stadium is used.
    """
    for col in stadium_column_candidates:
        if col in row.index:
            neutral = neutral_venue_from_stadium_name(row[col])
            if neutral is not None:
                return neutral, True

    home = canonical_team(row["home_team"])
    season = int(row["season"])
    return home_venue_for_team(home, season), False


def haversine_miles(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Great-circle distance between two latitude/longitude pairs."""
    earth_radius_miles = 3958.7613

    p1 = radians(float(lat1))
    p2 = radians(float(lat2))
    dphi = radians(float(lat2) - float(lat1))
    dlambda = radians(float(lon2) - float(lon1))

    a = (
        sin(dphi / 2.0) ** 2
        + cos(p1) * cos(p2) * sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * asin(sqrt(a))

    return earth_radius_miles * c


def utc_offset_hours(
    kickoff_utc: pd.Timestamp,
    timezone_name: str,
) -> float:
    """
    Actual UTC offset at kickoff.

    This makes the body-clock feature DST-aware. Arizona therefore behaves
    differently from Denver during daylight-saving months.
    """
    ts = pd.Timestamp(kickoff_utc)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    local = ts.to_pydatetime().astimezone(ZoneInfo(timezone_name))
    offset = local.utcoffset()

    if offset is None:
        raise ValueError(
            f"Could not determine UTC offset for {timezone_name}"
        )

    return offset.total_seconds() / 3600.0


def calculate_timezone_shift(
    kickoff_utc: pd.Timestamp,
    origin_timezone: str,
    destination_timezone: str,
) -> float:
    """
    Absolute body-clock displacement in hours.

    For domestic U.S. games this normally evaluates to 0, 1, 2, or 3.
    International games may be larger.
    """
    origin_offset = utc_offset_hours(kickoff_utc, origin_timezone)
    destination_offset = utc_offset_hours(
        kickoff_utc,
        destination_timezone,
    )

    return float(abs(destination_offset - origin_offset))


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------

def _team_previous_game_rest(
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate fallback rest days from the immediately previous game in the
    same season, using kickoff timestamps only.

    Week 1 is left missing here and later receives a neutral default. If nflverse
    already provides home_rest / away_rest, those fields take precedence.
    """
    appearances = []

    base_cols = [
        "game_id",
        "season",
        "week",
        "kickoff_utc",
        "home_team",
        "away_team",
    ]
    base = schedules[base_cols].copy()

    home = base[
        ["game_id", "season", "week", "kickoff_utc", "home_team"]
    ].rename(columns={"home_team": "team"})

    away = base[
        ["game_id", "season", "week", "kickoff_utc", "away_team"]
    ].rename(columns={"away_team": "team"})

    appearances = pd.concat([home, away], ignore_index=True)
    appearances = appearances.sort_values(
        ["season", "team", "kickoff_utc", "game_id"]
    ).reset_index(drop=True)

    appearances["previous_kickoff_utc"] = appearances.groupby(
        ["season", "team"],
        sort=False,
    )["kickoff_utc"].shift(1)

    delta = (
        appearances["kickoff_utc"]
        - appearances["previous_kickoff_utc"]
    )

    # Calendar/elapsed days. The exact convention does not materially change
    # net_rest because the same convention is used on both sides.
    appearances["calculated_rest_days"] = (
        delta.dt.total_seconds() / 86400.0
    )

    return appearances[
        ["game_id", "team", "calculated_rest_days"]
    ]


def add_rest_features(
    schedules: pd.DataFrame,
    *,
    week1_default_rest_days: float = 7.0,
) -> pd.DataFrame:
    """
    Add:
        home_context_rest_days
        away_context_rest_days
        net_rest = home - away

    Existing nflverse home_rest_days / away_rest_days are preferred.
    """
    df = schedules.copy()

    if "kickoff_utc" not in df.columns:
        raise ValueError("kickoff_utc is required for contextual features.")

    fallback = _team_previous_game_rest(df)

    home_fallback = fallback.rename(
        columns={
            "team": "home_team",
            "calculated_rest_days": "home_rest_calculated",
        }
    )
    away_fallback = fallback.rename(
        columns={
            "team": "away_team",
            "calculated_rest_days": "away_rest_calculated",
        }
    )

    df = df.merge(
        home_fallback,
        on=["game_id", "home_team"],
        how="left",
        validate="one_to_one",
    )

    df = df.merge(
        away_fallback,
        on=["game_id", "away_team"],
        how="left",
        validate="one_to_one",
    )

    if "home_rest_days" in df.columns:
        official_home = pd.to_numeric(
            df["home_rest_days"],
            errors="coerce",
        )
    elif "home_rest" in df.columns:
        official_home = pd.to_numeric(
            df["home_rest"],
            errors="coerce",
        )
    else:
        official_home = pd.Series(np.nan, index=df.index)

    if "away_rest_days" in df.columns:
        official_away = pd.to_numeric(
            df["away_rest_days"],
            errors="coerce",
        )
    elif "away_rest" in df.columns:
        official_away = pd.to_numeric(
            df["away_rest"],
            errors="coerce",
        )
    else:
        official_away = pd.Series(np.nan, index=df.index)

    df["home_context_rest_days"] = (
        official_home
        .fillna(df["home_rest_calculated"])
        .fillna(float(week1_default_rest_days))
    )

    df["away_context_rest_days"] = (
        official_away
        .fillna(df["away_rest_calculated"])
        .fillna(float(week1_default_rest_days))
    )

    df["net_rest"] = (
        df["home_context_rest_days"]
        - df["away_context_rest_days"]
    )

    return df


# ---------------------------------------------------------------------------
# DIVISION / TRAVEL / TIMEZONE
# ---------------------------------------------------------------------------

def add_divisional_feature(schedules: pd.DataFrame) -> pd.DataFrame:
    df = schedules.copy()

    def same_division(row: pd.Series) -> int:
        home = canonical_team(row["home_team"])
        away = canonical_team(row["away_team"])

        if home not in DIVISIONS or away not in DIVISIONS:
            raise KeyError(
                f"Unknown division mapping: home={home}, away={away}"
            )

        return int(DIVISIONS[home] == DIVISIONS[away])

    df["is_divisional"] = df.apply(same_division, axis=1).astype(int)
    return df


def add_travel_timezone_features(
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add travel and time-zone features.

    For ordinary home games:
        home_travel_miles = 0
        travel_distance_diff == away_travel_miles

    For recognized neutral games:
        both teams' travel is estimated and the difference is used.
    """
    df = schedules.copy()

    away_miles = []
    home_miles = []
    travel_diff = []
    tz_shifted = []
    resolved_venue = []
    neutral_flags = []

    for _, row in df.iterrows():
        season = int(row["season"])
        home_team = canonical_team(row["home_team"])
        away_team = canonical_team(row["away_team"])

        home_origin = home_venue_for_team(home_team, season)
        away_origin = home_venue_for_team(away_team, season)
        destination, is_neutral = game_venue(row)

        away_distance = haversine_miles(
            away_origin.latitude,
            away_origin.longitude,
            destination.latitude,
            destination.longitude,
        )

        if is_neutral:
            home_distance = haversine_miles(
                home_origin.latitude,
                home_origin.longitude,
                destination.latitude,
                destination.longitude,
            )
        else:
            home_distance = 0.0

        shift = calculate_timezone_shift(
            row["kickoff_utc"],
            away_origin.timezone,
            destination.timezone,
        )

        away_miles.append(float(away_distance))
        home_miles.append(float(home_distance))
        travel_diff.append(float(away_distance - home_distance))
        tz_shifted.append(float(shift))
        resolved_venue.append(destination.name)
        neutral_flags.append(int(is_neutral))

    df["away_travel_miles"] = away_miles
    df["home_travel_miles"] = home_miles
    df["travel_distance_diff"] = travel_diff
    df["tz_shifted"] = tz_shifted

    # Audit columns; not automatically included as predictive features.
    df["context_resolved_venue"] = resolved_venue
    df["context_neutral_site"] = neutral_flags

    return df


def add_context_features(
    schedules: pd.DataFrame,
    *,
    week1_default_rest_days: float = 7.0,
) -> pd.DataFrame:
    """
    Master feature-extraction function.

    Required columns:
        game_id
        season
        week
        kickoff_utc
        home_team
        away_team

    Returns the original schedule table plus contextual features.
    """
    required = {
        "game_id",
        "season",
        "week",
        "kickoff_utc",
        "home_team",
        "away_team",
    }
    missing = required - set(schedules.columns)

    if missing:
        raise ValueError(
            "Schedule table is missing required contextual columns: "
            + ", ".join(sorted(missing))
        )

    df = schedules.copy()

    df["home_team"] = df["home_team"].map(canonical_team)
    df["away_team"] = df["away_team"].map(canonical_team)
    df["kickoff_utc"] = pd.to_datetime(
        df["kickoff_utc"],
        utc=True,
        errors="raise",
    )

    df = add_rest_features(
        df,
        week1_default_rest_days=week1_default_rest_days,
    )
    df = add_divisional_feature(df)
    df = add_travel_timezone_features(df)

    validate_context_features(df)

    return df


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def validate_context_features(df: pd.DataFrame) -> None:
    """Sanity checks designed to catch broken joins or mappings early."""
    required = {
        "net_rest",
        "away_travel_miles",
        "home_travel_miles",
        "travel_distance_diff",
        "is_divisional",
        "tz_shifted",
    }

    missing = required - set(df.columns)
    if missing:
        raise AssertionError(
            "Context extraction did not produce: "
            + ", ".join(sorted(missing))
        )

    if not df["is_divisional"].isin([0, 1]).all():
        raise AssertionError("is_divisional must contain only 0/1.")

    if (df["away_travel_miles"] < 0).any():
        raise AssertionError("Away travel distance cannot be negative.")

    if (df["home_travel_miles"] < 0).any():
        raise AssertionError("Home travel distance cannot be negative.")

    if (df["tz_shifted"] < 0).any():
        raise AssertionError("tz_shifted cannot be negative.")

    expected_diff = (
        df["away_travel_miles"]
        - df["home_travel_miles"]
    )
    if not np.allclose(
        df["travel_distance_diff"],
        expected_diff,
        atol=1e-8,
        equal_nan=True,
    ):
        raise AssertionError(
            "travel_distance_diff does not equal away - home travel."
        )


def print_context_summary(df: pd.DataFrame) -> None:
    """Useful QC output before a backtest."""
    print("\nCONTEXT FEATURE SUMMARY")
    print("-" * 72)

    cols = [
        "net_rest",
        "away_travel_miles",
        "travel_distance_diff",
        "is_divisional",
        "tz_shifted",
    ]

    print(
        df[cols]
        .describe()
        .transpose()
        .round(3)
        .to_string()
    )

    print("\nLongest away trips:")
    show = [
        "season",
        "week",
        "away_team",
        "home_team",
        "away_travel_miles",
        "tz_shifted",
        "context_resolved_venue",
    ]
    print(
        df.nlargest(10, "away_travel_miles")[show]
        .round({"away_travel_miles": 1, "tz_shifted": 1})
        .to_string(index=False)
    )


if __name__ == "__main__":
    print(
        "This module is intended to be imported by run_backtest.py.\n"
        "Call add_context_features(schedule_dataframe)."
    )
