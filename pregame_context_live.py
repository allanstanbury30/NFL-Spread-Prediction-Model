#!/usr/bin/env python3
"""
pregame_context_live.py
=======================

Read-only NFL pregame context collector for the V3/V6 model.

What it does
------------
* Fetches hourly stadium weather from Open-Meteo.
* Optionally fetches SportsDataIO injuries + depth charts + player news.
* Converts raw feeds into timestamped, structured pregame features.
* NEVER changes a model pick or adds/subtracts points by itself.

The point of this module is to collect honest forward data first.  After enough
history is available, the structured fields can be walk-forward tested as model
features without relying on subjective article sentiment.

Environment variables
---------------------
SPORTSDATAIO_API_KEY   Optional. If absent, weather still works and injury/news
                       fields are left unavailable rather than guessed.

SportsDataIO endpoint names occasionally differ by subscription/product.  This
module therefore tries a small set of documented/legacy-compatible resource
names and fails soft if a feed is unavailable to the account.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
SPORTSDATAIO_BASE = "https://api.sportsdata.io/v3/nfl"

# Coordinates are stadium-area coordinates, sufficient for hourly forecast
# lookup.  Roof types intentionally distinguish fixed indoor from retractable.
# Retractable roof status must be treated as UNKNOWN unless separately confirmed.
TEAM_STADIUM = {
    "ARI": (33.5276, -112.2626, "RETRACTABLE", "State Farm Stadium"),
    "ATL": (33.7554, -84.4008, "RETRACTABLE", "Mercedes-Benz Stadium"),
    "BAL": (39.2780, -76.6227, "OUTDOOR", "M&T Bank Stadium"),
    "BUF": (42.7738, -78.7868, "OUTDOOR", "Buffalo stadium area"),
    "CAR": (35.2258, -80.8528, "OUTDOOR", "Bank of America Stadium"),
    "CHI": (41.8623, -87.6167, "OUTDOOR", "Soldier Field"),
    "CIN": (39.0954, -84.5160, "OUTDOOR", "Paycor Stadium"),
    "CLE": (41.5061, -81.6995, "OUTDOOR", "Huntington Bank Field"),
    "DAL": (32.7473, -97.0945, "RETRACTABLE", "AT&T Stadium"),
    "DEN": (39.7439, -105.0201, "OUTDOOR", "Empower Field at Mile High"),
    "DET": (42.3400, -83.0456, "INDOOR", "Ford Field"),
    "GB":  (44.5013, -88.0622, "OUTDOOR", "Lambeau Field"),
    "HOU": (29.6847, -95.4107, "RETRACTABLE", "NRG Stadium"),
    "IND": (39.7601, -86.1639, "RETRACTABLE", "Lucas Oil Stadium"),
    "JAX": (30.3239, -81.6373, "OUTDOOR", "EverBank Stadium"),
    "KC":  (39.0489, -94.4839, "OUTDOOR", "GEHA Field at Arrowhead Stadium"),
    "LV":  (36.0908, -115.1830, "INDOOR", "Allegiant Stadium"),
    "LAC": (33.9535, -118.3392, "ROOFED_OPEN_SIDES", "SoFi Stadium"),
    "LAR": (33.9535, -118.3392, "ROOFED_OPEN_SIDES", "SoFi Stadium"),
    "MIA": (25.9580, -80.2389, "ROOFED_OPEN_SIDES", "Hard Rock Stadium"),
    "MIN": (44.9738, -93.2575, "INDOOR", "U.S. Bank Stadium"),
    "NE":  (42.0909, -71.2643, "OUTDOOR", "Gillette Stadium"),
    "NO":  (29.9511, -90.0812, "INDOOR", "Caesars Superdome"),
    "NYG": (40.8135, -74.0745, "OUTDOOR", "MetLife Stadium"),
    "NYJ": (40.8135, -74.0745, "OUTDOOR", "MetLife Stadium"),
    "PHI": (39.9008, -75.1675, "OUTDOOR", "Lincoln Financial Field"),
    "PIT": (40.4468, -80.0158, "OUTDOOR", "Acrisure Stadium"),
    "SEA": (47.5952, -122.3316, "OUTDOOR", "Lumen Field"),
    "SF":  (37.4030, -121.9698, "OUTDOOR", "Levi's Stadium"),
    "TB":  (27.9759, -82.5033, "OUTDOOR", "Raymond James Stadium"),
    "TEN": (36.1665, -86.7713, "OUTDOOR", "Nissan Stadium"),
    "WAS": (38.9077, -76.8645, "OUTDOOR", "Northwest Stadium"),
}

TEAM_ALIASES = {
    "JAC": "JAX", "WSH": "WAS", "LA": "LAR", "OAK": "LV", "SD": "LAC",
}

STATUS_SEVERITY = {
    "OUT": 1.00,
    "IR": 1.00,
    "INJURED RESERVE": 1.00,
    "DOUBTFUL": 0.85,
    "QUESTIONABLE": 0.45,
    "PROBABLE": 0.10,
    "ACTIVE": 0.00,
    "": 0.00,
}

OFFENSIVE_LINE = {"C", "G", "OG", "OT", "T", "OL", "LT", "RT", "LG", "RG"}
SKILL = {"QB", "RB", "FB", "WR", "TE"}
DEFENSE = {"CB", "DB", "S", "FS", "SS", "LB", "ILB", "OLB", "DE", "DL", "DT", "NT", "EDGE"}

CONCRETE_NEWS_PATTERNS = {
    "ruled_out": re.compile(r"\b(ruled out|will not play|won't play|inactive)\b", re.I),
    "expected_out": re.compile(r"\b(not expected to play|unlikely to play)\b", re.I),
    "expected_active": re.compile(r"\b(expected to play|cleared to play|will play|active)\b", re.I),
    "limited": re.compile(r"\b(limited|limited participant)\b", re.I),
    "full_practice": re.compile(r"\b(full participant|practiced in full)\b", re.I),
    "suspension": re.compile(r"\b(suspend|suspension)\b", re.I),
    "qb_change": re.compile(r"\b(start(?:ing)? quarterback|named the starter|benched|backup quarterback)\b", re.I),
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: Optional[datetime] = None) -> str:
    return (dt or utc_now()).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_team(team: str) -> str:
    t = str(team).strip().upper()
    return TEAM_ALIASES.get(t, t)


def _json_get(url: str, *, headers: Optional[dict] = None, timeout: int = 25):
    req = Request(url, headers={"User-Agent": "NFL-Pregame-Context/1.0", "Accept": "application/json", **(headers or {})})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body[:300]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc.reason}") from exc


def _safe_float(v, default=np.nan) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def fetch_open_meteo_weather(home_team: str, kickoff_utc: str) -> dict:
    team = canonical_team(home_team)
    stadium = TEAM_STADIUM.get(team)
    base = {
        "weather_available": False,
        "stadium_name": "",
        "stadium_roof_type": "UNKNOWN",
        "weather_forecast_time_utc": "",
        "temperature_f": np.nan,
        "apparent_temperature_f": np.nan,
        "relative_humidity_pct": np.nan,
        "precip_probability_pct": np.nan,
        "precipitation_in": np.nan,
        "rain_in": np.nan,
        "snowfall_in": np.nan,
        "wind_mph": np.nan,
        "wind_gust_mph": np.nan,
        "weather_code": np.nan,
        "weather_hours_from_kickoff": np.nan,
    }
    if stadium is None:
        return base

    lat, lon, roof, stadium_name = stadium
    base.update({"stadium_name": stadium_name, "stadium_roof_type": roof})

    kickoff = pd.to_datetime(kickoff_utc, utc=True, errors="coerce")
    if pd.isna(kickoff):
        return base

    hourly = [
        "temperature_2m", "apparent_temperature", "relative_humidity_2m",
        "precipitation_probability", "precipitation", "rain", "snowfall",
        "weather_code", "wind_speed_10m", "wind_gusts_10m",
    ]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "UTC",
        "forecast_days": 16,
    }
    payload = _json_get(OPEN_METEO_URL + "?" + urlencode(params))
    h = payload.get("hourly", {}) if isinstance(payload, dict) else {}
    times = pd.to_datetime(h.get("time", []), utc=True, errors="coerce")
    if len(times) == 0:
        return base

    delta = np.abs((times - kickoff).total_seconds())
    if len(delta) == 0 or np.all(pd.isna(delta)):
        return base
    i = int(np.nanargmin(delta))

    def val(name):
        arr = h.get(name, [])
        return arr[i] if i < len(arr) else np.nan

    base.update({
        "weather_available": True,
        "weather_forecast_time_utc": str(times[i].isoformat()),
        "temperature_f": _safe_float(val("temperature_2m")),
        "apparent_temperature_f": _safe_float(val("apparent_temperature")),
        "relative_humidity_pct": _safe_float(val("relative_humidity_2m")),
        "precip_probability_pct": _safe_float(val("precipitation_probability")),
        "precipitation_in": _safe_float(val("precipitation")),
        "rain_in": _safe_float(val("rain")),
        "snowfall_in": _safe_float(val("snowfall")),
        "wind_mph": _safe_float(val("wind_speed_10m")),
        "wind_gust_mph": _safe_float(val("wind_gusts_10m")),
        "weather_code": _safe_float(val("weather_code")),
        "weather_hours_from_kickoff": float(delta[i] / 3600.0),
    })
    return base


def _sportsdata_get_first(api_key: str, resource_candidates: Sequence[str], *, timeout: int = 25):
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    errors = []
    for resource in resource_candidates:
        url = f"{SPORTSDATAIO_BASE}/{resource.lstrip('/')}"
        try:
            return _json_get(url, headers=headers, timeout=timeout), resource
        except Exception as exc:
            errors.append(f"{resource}: {exc}")
    raise RuntimeError("SportsDataIO feed unavailable. Tried: " + " | ".join(errors))


def fetch_sportsdata_injuries_and_depth(api_key: str) -> Tuple[List[dict], List[dict], dict]:
    meta = {}
    injuries, inj_resource = _sportsdata_get_first(api_key, [
        "scores/json/Injuries",
        "scores/json/Injuries/All",
        "stats/json/Injuries",
    ])
    depth, depth_resource = _sportsdata_get_first(api_key, [
        "scores/json/DepthCharts",
        "scores/json/DepthCharts/All",
        "stats/json/DepthCharts",
    ])
    meta["injury_resource"] = inj_resource
    meta["depth_resource"] = depth_resource
    return list(injuries or []), list(depth or []), meta


def fetch_sportsdata_player_details_by_team(api_key: str, team: str) -> Tuple[List[dict], str]:
    t = canonical_team(team)
    # Different SportsDataIO products/versions have exposed this resource under
    # slightly different names.  We try candidates and record the winning path.
    candidates = [
        f"scores/json/PlayerDetailsByTeam/{quote(t)}",
        f"scores/json/PlayerDetails/{quote(t)}",
        f"scores/json/Players/{quote(t)}",
        f"stats/json/PlayerDetailsByTeam/{quote(t)}",
    ]
    payload, resource = _sportsdata_get_first(api_key, candidates)
    return list(payload or []), resource


def _flatten_depth(depth_rows: Sequence[dict]) -> pd.DataFrame:
    rows = []
    for obj in depth_rows:
        if not isinstance(obj, dict):
            continue
        # Some feeds return TeamDepthChart objects with nested lists.
        nested_seen = False
        for category_key, category_name in (("Offense", "OFF"), ("Defense", "DEF"), ("SpecialTeams", "ST")):
            nested = obj.get(category_key)
            if isinstance(nested, list):
                nested_seen = True
                for r in nested:
                    if isinstance(r, dict):
                        x = dict(r)
                        x.setdefault("PositionCategory", category_name)
                        rows.append(x)
        if not nested_seen and ("PlayerID" in obj or "DepthOrder" in obj):
            rows.append(dict(obj))
    return pd.DataFrame(rows)


def _build_depth_lookup(depth_rows: Sequence[dict]) -> Dict[str, dict]:
    df = _flatten_depth(depth_rows)
    out: Dict[str, dict] = {}
    if df.empty:
        return out
    for _, r in df.iterrows():
        pid = r.get("PlayerID")
        if pd.isna(pid):
            continue
        out[str(int(pid)) if isinstance(pid, (int, np.integer, float, np.floating)) and not pd.isna(pid) else str(pid)] = {
            "depth_order": int(r.get("DepthOrder")) if pd.notna(r.get("DepthOrder")) else None,
            "position": str(r.get("Position") or "").upper(),
            "position_category": str(r.get("PositionCategory") or "").upper(),
            "name": str(r.get("Name") or ""),
        }
    return out


def _injury_status(row: Mapping) -> str:
    if bool(row.get("DeclaredInactive", False)):
        return "OUT"
    status = str(row.get("Status") or row.get("InjuryStatus") or "").upper().strip()
    return status


def summarize_team_injuries(team: str, injuries: Sequence[dict], depth_lookup: Mapping[str, dict]) -> dict:
    t = canonical_team(team)
    selected = []
    for r in injuries:
        if not isinstance(r, dict):
            continue
        rt = canonical_team(str(r.get("Team") or r.get("TeamKey") or ""))
        if rt != t:
            continue
        x = dict(r)
        pid = x.get("PlayerID")
        pid_key = str(int(pid)) if isinstance(pid, (int, np.integer, float, np.floating)) and not pd.isna(pid) else str(pid)
        d = depth_lookup.get(pid_key, {})
        x["_starter"] = d.get("depth_order") == 1
        x["_position"] = str(x.get("Position") or d.get("position") or "").upper()
        x["_position_category"] = str(d.get("position_category") or "").upper()
        x["_status"] = _injury_status(x)
        x["_severity"] = STATUS_SEVERITY.get(x["_status"], 0.25 if x["_status"] else 0.0)
        selected.append(x)

    starters = [r for r in selected if r["_starter"]]
    starter_out = [r for r in starters if r["_status"] in {"OUT", "IR", "INJURED RESERVE"}]
    starter_doubt = [r for r in starters if r["_status"] == "DOUBTFUL"]
    starter_q = [r for r in starters if r["_status"] == "QUESTIONABLE"]
    qb1 = [r for r in starters if r["_position"] == "QB"]

    def count_out_group(group: set) -> int:
        return sum(1 for r in starter_out if r["_position"] in group)

    key_names = []
    for r in sorted(starters, key=lambda z: z["_severity"], reverse=True)[:8]:
        name = str(r.get("Name") or r.get("PlayerName") or r.get("FirstName") or "").strip()
        if r.get("LastName") and str(r.get("LastName")) not in name:
            name = (name + " " + str(r.get("LastName"))).strip()
        label = f"{name or 'player'} {r['_position']} {r['_status']}".strip()
        key_names.append(label)

    return {
        "injuries_available": True,
        "injury_listed_count": len(selected),
        "starter_injury_count": len(starters),
        "starter_out_count": len(starter_out),
        "starter_doubtful_count": len(starter_doubt),
        "starter_questionable_count": len(starter_q),
        "starter_availability_severity_sum": float(sum(r["_severity"] for r in starters)),
        "qb1_injury_status": qb1[0]["_status"] if qb1 else "",
        "qb1_injury_severity": float(qb1[0]["_severity"]) if qb1 else 0.0,
        "ol_starters_out": count_out_group(OFFENSIVE_LINE),
        "skill_starters_out": count_out_group(SKILL),
        "def_starters_out": count_out_group(DEFENSE),
        "key_injury_labels": " | ".join(key_names),
    }


def summarize_team_news(team: str, player_details: Sequence[dict], retrieval_utc: datetime) -> dict:
    events = []
    latest_published = None
    for p in player_details:
        if not isinstance(p, dict):
            continue
        pos = str(p.get("Position") or "").upper()
        player_name = (str(p.get("FirstName") or "") + " " + str(p.get("LastName") or "")).strip()
        for n in p.get("LatestNews") or []:
            if not isinstance(n, dict):
                continue
            text = " ".join(str(n.get(k) or "") for k in ("Title", "Content", "Description", "Summary"))
            pub_raw = n.get("Updated") or n.get("TimeAgo") or n.get("Created") or n.get("Date")
            pub = pd.to_datetime(pub_raw, utc=True, errors="coerce")
            # Strict leakage guard: if we can parse a future timestamp, ignore it.
            if pd.notna(pub) and pub.to_pydatetime() > retrieval_utc:
                continue
            flags = [name for name, rx in CONCRETE_NEWS_PATTERNS.items() if rx.search(text)]
            if flags:
                events.append({"player": player_name, "position": pos, "flags": flags, "text": text[:260]})
            if pd.notna(pub):
                latest_published = pub if latest_published is None or pub > latest_published else latest_published

    counts = {k: 0 for k in CONCRETE_NEWS_PATTERNS}
    for e in events:
        for f in e["flags"]:
            counts[f] += 1
    labels = [f"{e['player']} {e['position']}: {','.join(e['flags'])}" for e in events[:8]]
    return {
        "news_available": True,
        "concrete_news_event_count": len(events),
        **{f"news_{k}_count": int(v) for k, v in counts.items()},
        "latest_news_timestamp_utc": "" if latest_published is None else str(latest_published.isoformat()),
        "key_news_labels": " | ".join(labels),
    }


def default_injury_summary() -> dict:
    return {
        "injuries_available": False, "injury_listed_count": 0, "starter_injury_count": 0,
        "starter_out_count": 0, "starter_doubtful_count": 0, "starter_questionable_count": 0,
        "starter_availability_severity_sum": np.nan, "qb1_injury_status": "",
        "qb1_injury_severity": np.nan, "ol_starters_out": 0, "skill_starters_out": 0,
        "def_starters_out": 0, "key_injury_labels": "",
    }


def default_news_summary() -> dict:
    d = {"news_available": False, "concrete_news_event_count": 0, "latest_news_timestamp_utc": "", "key_news_labels": ""}
    d.update({f"news_{k}_count": 0 for k in CONCRETE_NEWS_PATTERNS})
    return d


def collect_pregame_context(
    games: Sequence[Mapping],
    *,
    sportsdata_api_key: Optional[str] = None,
    include_news: bool = True,
    retrieval_time: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Collect one timestamped context row per game. No prediction changes."""
    retrieval = (retrieval_time or utc_now()).astimezone(timezone.utc)
    key = (sportsdata_api_key or os.environ.get("SPORTSDATAIO_API_KEY", "")).strip()

    injuries: List[dict] = []
    depth: List[dict] = []
    depth_lookup: Dict[str, dict] = {}
    feed_meta = {"sportsdata_enabled": bool(key), "sportsdata_error": "", "news_errors": {}}
    if key:
        try:
            injuries, depth, meta = fetch_sportsdata_injuries_and_depth(key)
            depth_lookup = _build_depth_lookup(depth)
            feed_meta.update(meta)
        except Exception as exc:
            feed_meta["sportsdata_error"] = str(exc)

    unique_teams = sorted({canonical_team(str(g.get("home") or g.get("home_team") or "")) for g in games} |
                          {canonical_team(str(g.get("away") or g.get("away_team") or "")) for g in games})
    news_by_team: Dict[str, dict] = {}
    if key and include_news and not feed_meta.get("sportsdata_error"):
        for t in unique_teams:
            if not t:
                continue
            try:
                players, resource = fetch_sportsdata_player_details_by_team(key, t)
                s = summarize_team_news(t, players, retrieval)
                s["news_resource"] = resource
                news_by_team[t] = s
            except Exception as exc:
                feed_meta["news_errors"][t] = str(exc)

    rows = []
    for g in games:
        home = canonical_team(str(g.get("home") or g.get("home_team") or ""))
        away = canonical_team(str(g.get("away") or g.get("away_team") or ""))
        kickoff = str(g.get("schedule_kickoff_utc") or g.get("kickoff_utc") or g.get("api_commence_time") or "")
        game_id = str(g.get("game_id") or "")
        row = {
            "context_retrieved_at_utc": iso_utc(retrieval),
            "game_id": game_id,
            "season": g.get("season"),
            "week": g.get("week"),
            "away_team": away,
            "home_team": home,
            "kickoff_utc": kickoff,
            "context_is_pregame": bool(pd.to_datetime(kickoff, utc=True, errors="coerce") > pd.Timestamp(retrieval)) if kickoff else False,
        }
        try:
            row.update(fetch_open_meteo_weather(home, kickoff))
            row["weather_error"] = ""
        except Exception as exc:
            row.update({"weather_available": False, "weather_error": str(exc)})

        home_inj = default_injury_summary(); away_inj = default_injury_summary()
        if depth_lookup and injuries:
            home_inj = summarize_team_injuries(home, injuries, depth_lookup)
            away_inj = summarize_team_injuries(away, injuries, depth_lookup)
        row.update({f"home_{k}": v for k, v in home_inj.items()})
        row.update({f"away_{k}": v for k, v in away_inj.items()})

        home_news = news_by_team.get(home, default_news_summary())
        away_news = news_by_team.get(away, default_news_summary())
        row.update({f"home_{k}": v for k, v in home_news.items() if k != "news_resource"})
        row.update({f"away_{k}": v for k, v in away_news.items() if k != "news_resource"})

        # Differential features are deliberately descriptive, not a point adjustment.
        for suffix in (
            "starter_out_count", "starter_doubtful_count", "starter_questionable_count",
            "starter_availability_severity_sum", "qb1_injury_severity", "ol_starters_out",
            "skill_starters_out", "def_starters_out", "concrete_news_event_count",
        ):
            h = _safe_float(row.get(f"home_{suffix}"))
            a = _safe_float(row.get(f"away_{suffix}"))
            row[f"net_{suffix}_home_minus_away"] = h - a if np.isfinite(h) and np.isfinite(a) else np.nan

        rows.append(row)

    return pd.DataFrame(rows), feed_meta


def append_snapshot(frame: pd.DataFrame, *, ledger_dir: str, metadata: Optional[dict] = None) -> Tuple[Path, Path]:
    root = Path(ledger_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "pregame_context_snapshots.csv"
    jsonl_path = root / "pregame_context_metadata.jsonl"

    if not frame.empty:
        header = not csv_path.exists()
        frame.to_csv(csv_path, mode="a", header=header, index=False)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"saved_at_utc": iso_utc(), **(metadata or {})}, default=str) + "\n")
    return csv_path, jsonl_path


def compact_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    cols = [
        "week", "away_team", "home_team", "stadium_roof_type", "temperature_f",
        "wind_mph", "wind_gust_mph", "precip_probability_pct",
        "away_starter_out_count", "home_starter_out_count",
        "away_qb1_injury_status", "home_qb1_injury_status",
        "away_concrete_news_event_count", "home_concrete_news_event_count",
    ]
    return frame[[c for c in cols if c in frame.columns]].copy()
