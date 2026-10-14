#!/usr/bin/env python3
"""
run_production_live_v6_2026.py
===============================

ONE-COMMAND provisional 2026 production runner:

    frozen V3 football model
        -> frozen V6 ATS probability model
        -> predeclared V6_EV_POSITIVE spread filter
        -> optional/read-only weather + injury/news context
        -> immutable forward-test ledger

Official spread policy (FROZEN)
-------------------------------
1) V3 must first select a spread bet.
2) V6 scores the SAME selected side at the CURRENT spread price.
3) Keep the spread only when V6 expected value per unit risked is >= 0.00.
4) If V6 EV < 0, the V3 spread is vetoed and the official decision is PASS.
5) V6 does NOT promote new marginal spread bets.
6) Existing V3 moneyline fallback decisions are held fixed. A V6 veto of a
   V3 spread does NOT create a new moneyline fallback.

This exactly matches the predeclared V6_EV_POSITIVE policy that passed the
retrospective portfolio shadow test. Do not tune the EV threshold week to week.

Context policy
--------------
Weather is collected automatically and remains READ ONLY. SportsDataIO is
optional. If no injury API key is configured, manually review injuries/depth
charts before placing a qualifying bet. Context does not alter V3/V6 yet.

Required local files
--------------------
- run_production_live_market_2026.py
- run_live_production_2026.py
- run_production_ridge_v3_ats_regime_promotion_2026.py
- run_spread_probability_optimizer_v6.py
- pregame_context_live.py
- team_continuity_2018_2026.csv
- data/production_ridge_v3_replay_2022_2023/
    backtest_predictions_with_ml_fallback.parquet

Required environment variable
-----------------------------
ODDS_API_KEY

Optional
--------
SPORTSDATAIO_API_KEY
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import pregame_context_live as ctx
import run_live_production_2026 as live
import run_production_live_market_2026 as market
import run_spread_probability_optimizer_v6 as v6


MODEL_VERSION = "V3_RIDGE_REGIME_PROMOTION_2026__V6_EV_POSITIVE_FILTER_PROVISIONAL"
V6_POLICY = "V6_EV_POSITIVE"
V6_MIN_EV = 0.0
DEFAULT_LEDGER_DIR = "data/live_production_ledger_v6_2026"
DEFAULT_REPLAY = (
    "data/production_ridge_v3_replay_2022_2023/"
    "backtest_predictions_with_ml_fallback.parquet"
)

# Frozen V6 settings from the optimizer that passed the declared gates.
V6_PROD_FLOOR = 0.50
V6_PROD_CEILING = 0.60
V6_MARGINAL_FLOOR = 0.47
V6_MARGINAL_CEILING = 0.58
V6_MIN_TRAIN_ROWS = 30
V6_PRODUCTION_RANK_AMPLITUDE = 0.0125
V6_ONLINE_RANK_MIN_GAMES = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "2026 live NFL production: frozen V3 + provisional V6 EV-positive "
            "spread filter + read-only pregame context."
        )
    )
    p.add_argument("--production-script", default=market.DEFAULT_PRODUCTION_SCRIPT)
    p.add_argument("--continuity", default=market.DEFAULT_CONTINUITY)
    p.add_argument("--season", type=int, default=market.DEFAULT_SEASON)
    p.add_argument("--book", default=market.DEFAULT_BOOK)
    p.add_argument("--region", default=market.DEFAULT_REGION)
    p.add_argument("--lookahead-days", type=int, default=market.DEFAULT_LOOKAHEAD_DAYS)
    p.add_argument("--ledger-dir", default=DEFAULT_LEDGER_DIR)
    p.add_argument("--display-timezone", default=live.DEFAULT_DISPLAY_TIMEZONE)
    p.add_argument("--ridge-alpha", type=float, default=250.0)
    p.add_argument("--train-start", type=int, default=2018)
    p.add_argument("--champion-train-end", type=int, default=2023)
    p.add_argument("--api-key-env", default="ODDS_API_KEY")
    p.add_argument("--sportsdata-key-env", default="SPORTSDATAIO_API_KEY")
    p.add_argument("--replay", default=DEFAULT_REPLAY)
    p.add_argument("--no-news", action="store_true")
    p.add_argument("--no-context", action="store_true", help="Skip weather/injury context collection.")
    p.add_argument("--market-only", action="store_true")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def american_profit_per_unit(odds: float) -> float:
    o = float(odds)
    if o > 0:
        return o / 100.0
    if o < 0:
        return 100.0 / abs(o)
    return np.nan


def american_break_even(odds: float) -> float:
    o = float(odds)
    if o < 0:
        return (-o) / ((-o) + 100.0)
    if o > 0:
        return 100.0 / (o + 100.0)
    return np.nan


def build_batch_with_full_history(
    *,
    prod,
    args,
    matched_markets: Sequence[dict],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Same frozen V3 live build as the existing runner, but retain full OOS history.

    Full history is needed so V6's Week-N rank activation can use only completed
    Weeks < N in the CURRENT 2026 season instead of relying on a stale saved
    backtest artifact.
    """
    if not matched_markets:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    config = market.build_config(prod, args)
    c1, c2, c3 = prod.load_components()

    model_seasons = list(range(config.train_start, int(args.season) + 1))
    feature_seasons = list(range(config.train_start - 1, int(args.season) + 1))

    print("\n[2/8] Loading shared PBP + schedule data ONCE")
    pbp = c1.load_pbp(feature_seasons)
    raw_schedules = prod.load_nflverse_schedules(model_seasons)
    schedules = prod.prepare_schedule_table(raw_schedules, seasons=model_seasons)
    context_module = prod.load_context_feature_module()
    schedules = context_module.add_context_features(schedules)

    target_game_ids = {str(m["game_id"]) for m in matched_markets}
    target_rows: List[pd.DataFrame] = []

    for m in matched_markets:
        hit = schedules[schedules["game_id"].astype(str) == str(m["game_id"])].copy()
        if len(hit) != 1:
            raise RuntimeError(
                f"Expected one schedule row for game_id={m['game_id']}; found {len(hit)}"
            )
        hit["close_spread_home"] = float(m["spread_home"])
        hit["bet_spread_home"] = float(m["spread_home"])
        hit["close_total"] = market.safe_float(m.get("total"))
        hit["close_home_ml"] = market.safe_float(m.get("home_ml"))
        hit["close_away_ml"] = market.safe_float(m.get("away_ml"))
        hit["bet_american_odds_home"] = market.safe_float(m.get("home_spread_odds"), -110.0)
        hit["bet_american_odds_away"] = market.safe_float(m.get("away_spread_odds"), -110.0)
        hit["home_score"] = np.nan
        hit["away_score"] = np.nan
        hit["market_timestamp_utc"] = args.market_timestamp
        hit["live_market_source"] = str(m.get("bookmaker_key", ""))
        hit["live_market_event_id"] = str(m.get("api_event_id", ""))
        hit = prod.add_devigged_moneyline_probabilities(hit)
        target_rows.append(hit)

    target_frame = pd.concat(target_rows, ignore_index=True, sort=False)

    continuity = prod.load_continuity_table(args.continuity, seasons=model_seasons)
    prod.validate_continuity_coverage(
        schedules,
        continuity,
        target_seasons=model_seasons,
        allow_missing=config.allow_missing_continuity,
    )

    print("\n[3/8] Building Component #1 opponent-adjusted efficiency")
    _, current_snapshots, prior_baselines = prod.build_component1_outputs(
        c1,
        pbp=pbp,
        all_feature_seasons=feature_seasons,
        config=config,
    )

    print("\n[4/8] Building Component #2 continuity blends")
    blended_ratings = prod.build_component2_blends(
        c2,
        current_snapshots=current_snapshots,
        prior_baselines=prior_baselines,
        schedules=schedules,
        continuity=continuity,
        target_seasons=model_seasons,
        config=config,
    )

    print("\n[5/8] Synthesizing completed history + live market targets")
    completed_games = prod.build_game_table(
        schedules,
        model_seasons=model_seasons,
        external_market=None,
    )
    completed_games = completed_games[
        ~completed_games["game_id"].astype(str).isin(target_game_ids)
    ].copy()
    all_games = pd.concat([completed_games, target_frame], ignore_index=True, sort=False)

    model_dataset = c3.synthesize_dataset(
        games=all_games,
        blended_team_ratings=blended_ratings,
        market=None,
    )
    features = prod.choose_model_features(c3, model_dataset)
    context_cols = ["net_rest", "travel_distance_diff", "is_divisional", "tz_shifted"]
    present_context = [c for c in context_cols if c in features]
    print("  Model features:", len(features))
    print("  Option-2 context in Ridge:", ", ".join(present_context) or "NONE")

    print("\n[6/8] Running chronological frozen V3 prediction pass")
    predictions = prod.weekly_walk_forward_predict(
        c3,
        model_dataset,
        feature_columns=features,
        test_start=2024,
        test_end=int(args.season),
        config=config,
    )

    annotated = prod.add_moneyline_fallback_analysis(
        predictions,
        spread_threshold=config.spread_qualifying_threshold,
        strong_prob_edge=config.ml_strong_prob_edge,
        strong_ev=config.ml_strong_ev,
        very_strong_prob_edge=config.ml_very_strong_prob_edge,
        very_strong_ev=config.ml_very_strong_ev,
        low_confidence_ev_surcharge=config.ml_low_confidence_ev_surcharge,
        medium_confidence_ev_surcharge=config.ml_medium_confidence_ev_surcharge,
        heavy_favorite_threshold=config.ml_heavy_favorite_threshold,
        heavy_favorite_ev_surcharge=config.ml_heavy_favorite_ev_surcharge,
    )

    target_predictions = annotated[
        annotated["game_id"].astype(str).isin(target_game_ids)
    ].copy()

    return target_predictions, annotated, model_dataset, list(features)


def score_live_v6(
    *,
    full_v3_predictions: pd.DataFrame,
    target_game_ids: Sequence[str],
    replay_path: Path,
    season: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the already-frozen V6 procedure and score this live slate.

    History:
      replay artifact -> 2022-2023
      freshly rebuilt V3 OOS predictions -> 2024 through completed 2026 weeks

    Target:
      current live market rows in the requested 2026 slate
    """
    if not replay_path.exists():
        raise FileNotFoundError(
            f"Missing V6 replay history: {replay_path}\n"
            "Run/copy the 2022-2023 replay artifact before live V6 scoring."
        )

    replay = _read_table(replay_path).copy()
    # Preserve V6's context-recovery behavior for the older replay artifact.
    if hasattr(v6, "_merge_sibling_context"):
        replay = v6._merge_sibling_context(replay, replay_path)
    replay["_source"] = "REPLAY"

    current = full_v3_predictions.copy()
    current["_source"] = "LIVE_REBUILD"

    combined = pd.concat([replay, current], ignore_index=True, sort=False)
    combined["season"] = pd.to_numeric(combined["season"], errors="coerce")
    combined["week"] = pd.to_numeric(combined["week"], errors="coerce")
    combined = combined[combined["season"].notna() & combined["week"].notna()].copy()
    combined["season"] = combined["season"].astype(int)
    combined["week"] = combined["week"].astype(int)

    # Keep one row per historical game; the fresh live rebuild has priority.
    if "game_id" in combined.columns:
        combined["_priority"] = combined["_source"].map(
            {"REPLAY": 0, "LIVE_REBUILD": 1}
        ).fillna(0)
        combined = (
            combined.sort_values(["season", "week", "game_id", "_priority"])
            .drop_duplicates(["season", "week", "game_id"], keep="last")
            .drop(columns=["_priority"])
        )

    prepared = v6.prepare_ats_frame(combined)
    calibrated, _, _, _, champions, activation = v6.fit_and_predict_targets(
        prepared,
        [int(season)],
        prod_floor=V6_PROD_FLOOR,
        prod_ceiling=V6_PROD_CEILING,
        marginal_floor=V6_MARGINAL_FLOOR,
        marginal_ceiling=V6_MARGINAL_CEILING,
        min_train_rows=V6_MIN_TRAIN_ROWS,
        production_rank_amplitude=V6_PRODUCTION_RANK_AMPLITUDE,
        online_rank_min_games=V6_ONLINE_RANK_MIN_GAMES,
    )

    live_rows = calibrated[
        calibrated["game_id"].astype(str).isin([str(x) for x in target_game_ids])
    ].copy()

    if live_rows.empty:
        raise RuntimeError("V6 produced no rows for the live target slate.")

    side = live_rows["ats_v6_selected_side"].astype(str).str.upper()
    home_odds = pd.to_numeric(live_rows.get("bet_american_odds_home", -110.0), errors="coerce")
    away_odds = pd.to_numeric(live_rows.get("bet_american_odds_away", -110.0), errors="coerce")
    selected_odds = pd.Series(
        np.where(side.eq("HOME"), home_odds, away_odds),
        index=live_rows.index,
        dtype=float,
    ).fillna(-110.0)

    live_rows["v6_selected_odds"] = selected_odds
    live_rows["v6_break_even_probability"] = selected_odds.map(american_break_even)
    p = pd.to_numeric(live_rows["ats_v6_cover_probability"], errors="coerce")
    live_rows["v6_probability_edge_vs_price"] = p - live_rows["v6_break_even_probability"]
    win_profit = selected_odds.map(american_profit_per_unit)
    live_rows["v6_ev_per_unit"] = p * win_profit - (1.0 - p)
    live_rows["v6_keep_v3_spread"] = (
        live_rows["ats_v6_current_v3_bet"].astype(bool)
        & live_rows["v6_ev_per_unit"].notna()
        & (live_rows["v6_ev_per_unit"] >= V6_MIN_EV)
    )

    # Compact diagnostics for console output.
    diag = activation.copy()
    if not champions.empty:
        selected = champions[champions["population"].eq("PRODUCTION")].copy()
        if not selected.empty:
            print("\nV6 LIVE CONFIGURATION")
            print("-" * 132)
            cols = [
                c for c in [
                    "target_season", "calibrator", "prior_cv_brier",
                    "prior_market_brier", "prior_cv_ece", "prior_market_ece",
                    "history_seasons",
                ] if c in selected.columns
            ]
            print(selected[cols].to_string(index=False))

    return live_rows, diag


def v6_official_final(v3_final: Dict, v6_row: pd.Series) -> Tuple[Dict, str]:
    """Apply the frozen production policy to the V3 final recommendation."""
    if v3_final["final_bet_type"] == "SPREAD":
        ev = market.safe_float(v6_row.get("v6_ev_per_unit"))
        if math.isfinite(ev) and ev >= V6_MIN_EV:
            out = dict(v3_final)
            out["final_reason"] = "V6_EV_POSITIVE_KEEP__" + str(v3_final.get("final_reason", "V3_SPREAD"))
            return out, "V6_KEEP"
        if math.isfinite(ev):
            return {
                "final_bet_type": "PASS",
                "final_selected_side": "NONE",
                "final_selected_team": "NONE",
                "final_market_value": np.nan,
                "final_american_odds": np.nan,
                "final_confidence": "V6 VETO",
                "final_reason": "V6_NEGATIVE_EV_VETO",
            }, "V6_VETO"
        return {
            "final_bet_type": "PASS",
            "final_selected_side": "NONE",
            "final_selected_team": "NONE",
            "final_market_value": np.nan,
            "final_american_odds": np.nan,
            "final_confidence": "V6 UNAVAILABLE",
            "final_reason": "V6_SCORE_UNAVAILABLE_FAIL_CLOSED",
        }, "V6_UNAVAILABLE"

    if v3_final["final_bet_type"] == "MONEYLINE":
        out = dict(v3_final)
        out["final_reason"] = "V3_ML_FALLBACK_FIXED"
        return out, "FIXED_V3_ML"

    return dict(v3_final), "PASS"


def _final_text(final: Dict) -> str:
    if final["final_bet_type"] == "PASS":
        return "PASS"
    if final["final_bet_type"] == "SPREAD":
        return f"{final['final_selected_team']} {float(final['final_market_value']):+g}"
    return f"{final['final_selected_team']} ML {float(final['final_american_odds']):+.0f}"


def save_official_decisions(
    *,
    args,
    matched_markets: Sequence[dict],
    target_predictions: pd.DataFrame,
    v6_live: pd.DataFrame,
    features: Sequence[str],
    context_frame: pd.DataFrame,
) -> pd.DataFrame:
    root = Path(args.ledger_dir)
    root.mkdir(parents=True, exist_ok=True)
    official_path = live.ledger_paths(args.ledger_dir)["predictions"]
    comparison_path = root / "v3_vs_v6_decision_snapshots.csv"

    v6_by_game = {
        str(r["game_id"]): r for _, r in v6_live.iterrows()
    }
    context_by_game: Dict[str, pd.Series] = {}
    if not context_frame.empty and "game_id" in context_frame.columns:
        context_by_game = {
            str(r["game_id"]): r for _, r in context_frame.iterrows()
        }

    rows: List[dict] = []
    timestamp = args.market_timestamp

    for m in matched_markets:
        game_id = str(m["game_id"])
        hit = target_predictions[target_predictions["game_id"].astype(str) == game_id]
        if len(hit) != 1:
            print(f"WARNING: V3 target prediction count for {game_id} = {len(hit)}; skipping")
            continue
        if game_id not in v6_by_game:
            print(f"WARNING: V6 row missing for {game_id}; spread decisions will fail closed")
            vr = pd.Series(dtype=object)
        else:
            vr = v6_by_game[game_id]

        row = hit.iloc[0]
        game_args = market.make_game_args(
            args,
            season=int(m["season"]),
            week=int(m["week"]),
            away=str(m["away"]),
            home=str(m["home"]),
            market=m,
            timestamp=timestamp,
        )
        v3_final = live.final_recommendation(row, game_args)
        official, status = v6_official_final(v3_final, vr)
        pred_id = live.prediction_id_for(
            season=int(m["season"]),
            week=int(m["week"]),
            away=str(m["away"]),
            home=str(m["home"]),
            timestamp_utc=timestamp,
        )

        c = context_by_game.get(game_id, pd.Series(dtype=object))

        extra = {
            "model_version": MODEL_VERSION,
            "v6_policy": V6_POLICY,
            "v6_min_ev_per_unit": V6_MIN_EV,
            "v6_decision_status": status,
            "v6_cover_probability": vr.get("ats_v6_cover_probability", np.nan),
            "v6_market_cover_probability": vr.get("ats_v6_market_cover_probability", np.nan),
            "v6_selected_odds": vr.get("v6_selected_odds", np.nan),
            "v6_break_even_probability": vr.get("v6_break_even_probability", np.nan),
            "v6_probability_edge_vs_price": vr.get("v6_probability_edge_vs_price", np.nan),
            "v6_ev_per_unit": vr.get("v6_ev_per_unit", np.nan),
            "v6_rank_strength": vr.get("ats_v6_rank_strength", np.nan),
            "v6_rank_multiplier": vr.get("ats_v6_rank_multiplier", np.nan),
            "v6_rank_direction": vr.get("ats_v6_rank_direction", np.nan),
            "v6_ranker": vr.get("ats_v6_ranker", ""),
            "v6_calibrator": vr.get("ats_v6_calibrator", ""),
            "v6_population": vr.get("ats_v6_population", ""),
            "v3_counterfactual_bet_type": v3_final["final_bet_type"],
            "v3_counterfactual_selected_team": v3_final["final_selected_team"],
            "v3_counterfactual_market_value": v3_final["final_market_value"],
            "v3_counterfactual_american_odds": v3_final["final_american_odds"],
            "v3_counterfactual_confidence": v3_final["final_confidence"],
            "v3_counterfactual_reason": v3_final["final_reason"],
            "context_authority": "READ_ONLY",
            "temperature_f": c.get("temperature_f", np.nan),
            "wind_mph": c.get("wind_mph", np.nan),
            "wind_gust_mph": c.get("wind_gust_mph", np.nan),
            "precip_probability_pct": c.get("precip_probability_pct", np.nan),
            "away_starter_out_count": c.get("away_starter_out_count", np.nan),
            "home_starter_out_count": c.get("home_starter_out_count", np.nan),
            "away_qb1_injury_status": c.get("away_qb1_injury_status", ""),
            "home_qb1_injury_status": c.get("home_qb1_injury_status", ""),
        }

        if not args.no_save:
            ledger_row = live.make_prediction_ledger_row(
                row,
                game_args,
                prediction_id=pred_id,
                timestamp_utc=timestamp,
                final=official,
                features=features,
            )
            ledger_row.update(extra)
            ledger_row["production_script"] = str(args.production_script)
            ledger_row["market_source_mode"] = str(m.get("bookmaker_key", ""))
            ledger_row["market_source_title"] = str(m.get("bookmaker_title", ""))
            ledger_row["market_source_last_update"] = str(m.get("bookmaker_last_update", ""))
            ledger_row["market_consensus_book_count"] = int(m.get("consensus_book_count", 1))
            ledger_row["odds_api_event_id"] = str(m.get("api_event_id", ""))
            ledger_row["odds_api_commence_time"] = str(m.get("api_commence_time", ""))
            live.append_row_csv(official_path, ledger_row)

            compare = {
                "prediction_id": pred_id,
                "timestamp_utc": timestamp,
                "season": int(m["season"]),
                "week": int(m["week"]),
                "game_id": game_id,
                "game": f"{m['away']} @ {m['home']}",
                "market_source": str(m.get("bookmaker_key", "")),
                "home_spread": float(m["spread_home"]),
                "home_spread_odds": market.safe_float(m.get("home_spread_odds"), -110.0),
                "away_spread_odds": market.safe_float(m.get("away_spread_odds"), -110.0),
                "v3_final": _final_text(v3_final),
                "official_v6_final": _final_text(official),
                **extra,
            }
            live.append_row_csv(comparison_path, compare)

        rows.append(
            {
                "week": int(m["week"]),
                "game": f"{m['away']} @ {m['home']}",
                "source": str(m.get("bookmaker_key", "")),
                "home_spread": float(m["spread_home"]),
                "model_home_margin": market.safe_float(row.get("predicted_home_margin")),
                "decision_edge_home": market.safe_float(row.get("decision_edge_home")),
                "v3_final": _final_text(v3_final),
                "v6_p": market.safe_float(vr.get("ats_v6_cover_probability")),
                "break_even_p": market.safe_float(vr.get("v6_break_even_probability")),
                "v6_ev": market.safe_float(vr.get("v6_ev_per_unit")),
                "v6_status": status,
                "official_final": _final_text(official),
                "temperature_f": c.get("temperature_f", np.nan),
                "wind_mph": c.get("wind_mph", np.nan),
                "wind_gust_mph": c.get("wind_gust_mph", np.nan),
                "precip_probability_pct": c.get("precip_probability_pct", np.nan),
                "prediction_id": pred_id,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    odds_key = os.environ.get(args.api_key_env, "").strip()
    if not odds_key:
        raise SystemExit(
            f"Missing {args.api_key_env}. Set it locally in PowerShell:\n"
            f"  $env:{args.api_key_env}=\"YOUR_KEY\""
        )

    timestamp = market.iso_utc()
    args.market_timestamp = timestamp

    print("=" * 132)
    print("NFL 2026 PROVISIONAL PRODUCTION -- FROZEN V3 + V6 EV-POSITIVE FILTER")
    print("=" * 132)
    print(f"Official spread policy : {V6_POLICY} (V6 EV >= {V6_MIN_EV:+.2f})")
    print("Spread promotions      : OFF")
    print("V3 moneyline fallback  : FIXED / unchanged")
    print("Weather/injury context : READ ONLY")
    print(f"Ledger                 : {args.ledger_dir}")
    print(f"Snapshot timestamp     : {timestamp}")

    print("\n[0/8] Fetching current NFL market odds")
    api_events, quota = market.fetch_current_nfl_odds(
        api_key=odds_key,
        region=args.region,
        book=args.book,
    )
    print(f"  API events returned: {len(api_events)}")
    if quota.get("requests_remaining"):
        print(
            "  API quota: "
            f"remaining={quota['requests_remaining']} | "
            f"used={quota['requests_used']} | "
            f"this_call={quota['requests_last']}"
        )

    prod = live.load_production_module(args.production_script)
    raw_sched = prod.load_nflverse_schedules([int(args.season)])
    sched = prod.prepare_schedule_table(raw_sched, seasons=[int(args.season)])
    matched, warnings = market.match_market_events_to_schedule(
        prod=prod,
        schedules=sched,
        api_events=api_events,
        book=args.book,
        season=int(args.season),
        now_utc=market.utc_now(),
        lookahead_days=int(args.lookahead_days),
    )
    for w in warnings:
        print("  WARNING:", w)
    market.print_market_preview(matched)
    if not matched:
        print("\nNo upcoming matched NFL games.")
        return

    context_frame = pd.DataFrame()
    if not args.no_context:
        print("\n[1/8] Fetching stadium weather + optional injury/news context")
        sports_key = os.environ.get(args.sportsdata_key_env, "").strip()
        context_frame, context_meta = ctx.collect_pregame_context(
            matched,
            sportsdata_api_key=sports_key,
            include_news=not args.no_news,
        )
        if sports_key:
            if context_meta.get("sportsdata_error"):
                print("  Injury/news warning:", context_meta["sportsdata_error"])
            else:
                print("  Injury/depth feed: connected")
        else:
            print("  Injury API not configured -> weather works; injury review remains MANUAL.")

        if not context_frame.empty:
            print("\nPREGAME CONTEXT SNAPSHOT")
            print("-" * 132)
            print(ctx.compact_display(context_frame).to_string(index=False))

        if not args.no_save:
            ctx.append_snapshot(
                context_frame,
                ledger_dir=args.ledger_dir,
                metadata={
                    "market_timestamp_utc": timestamp,
                    "season": int(args.season),
                    "book": args.book,
                    "official_model": MODEL_VERSION,
                    "context_authority": "READ_ONLY",
                },
            )
    else:
        print("\n[1/8] Context collection skipped by --no-context")

    if args.market_only:
        print("\nMARKET-ONLY mode: odds/context collected; model not run.")
        return

    target_predictions, full_predictions, _, features = build_batch_with_full_history(
        prod=prod,
        args=args,
        matched_markets=matched,
    )

    print("\n[7/8] Scoring live slate with frozen V6 probability procedure")
    target_ids = [str(m["game_id"]) for m in matched]
    v6_live, activation = score_live_v6(
        full_v3_predictions=full_predictions,
        target_game_ids=target_ids,
        replay_path=Path(args.replay),
        season=int(args.season),
    )
    if not activation.empty:
        current_weeks = sorted({int(m["week"]) for m in matched})
        show = activation[activation["week"].isin(current_weeks)].copy()
        if not show.empty:
            print("\nV6 CURRENT-WEEK RANK ACTIVATION")
            print("-" * 132)
            print(show.to_string(index=False))

    print("\n[8/8] Applying OFFICIAL V6_EV_POSITIVE spread filter + saving forward ledger")
    summary = save_official_decisions(
        args=args,
        matched_markets=matched,
        target_predictions=target_predictions,
        v6_live=v6_live,
        features=features,
        context_frame=context_frame,
    )

    print("\n" + "=" * 132)
    print("LIVE V3 + V6 PRODUCTION SLATE")
    print("=" * 132)
    if summary.empty:
        print("No decisions were generated.")
        return

    display = summary.drop(columns=["prediction_id"], errors="ignore").copy()
    for c in ["v6_p", "break_even_p", "v6_ev"]:
        if c in display.columns:
            vals = pd.to_numeric(display[c], errors="coerce")
            display[c] = vals.map(lambda x: "N/A" if pd.isna(x) else f"{100.0*x:.2f}%")
    print(display.to_string(index=False))

    bets = summary[summary["official_final"].ne("PASS")].copy()
    print("\nOFFICIAL QUALIFYING BETS")
    print("-" * 132)
    if bets.empty:
        print("No official qualifying bets at this market snapshot.")
    else:
        bet_show = bets[
            [
                "week", "game", "official_final", "v3_final",
                "v6_p", "break_even_p", "v6_ev", "v6_status",
                "temperature_f", "wind_mph", "wind_gust_mph",
                "precip_probability_pct",
            ]
        ].copy()
        for c in ["v6_p", "break_even_p", "v6_ev"]:
            vals = pd.to_numeric(bet_show[c], errors="coerce")
            bet_show[c] = vals.map(lambda x: "N/A" if pd.isna(x) else f"{100.0*x:.2f}%")
        print(bet_show.to_string(index=False))

    vetoes = summary[summary["v6_status"].eq("V6_VETO")]
    print("\nV6 VETOED V3 SPREADS")
    print("-" * 132)
    if vetoes.empty:
        print("None.")
    else:
        veto_show = vetoes[["week", "game", "v3_final", "v6_p", "break_even_p", "v6_ev"]].copy()
        for c in ["v6_p", "break_even_p", "v6_ev"]:
            vals = pd.to_numeric(veto_show[c], errors="coerce")
            veto_show[c] = vals.map(lambda x: "N/A" if pd.isna(x) else f"{100.0*x:.2f}%")
        print(veto_show.to_string(index=False))

    print("\nMANUAL PREGAME CHECK BEFORE PLACING ANY OFFICIAL BET")
    print("-" * 132)
    print("  [ ] Starting QB status / unexpected QB change")
    print("  [ ] Starting offensive-line absences")
    print("  [ ] Top WR/RB/TE absences")
    print("  [ ] Major defensive starter clusters")
    print("  [ ] Late inactive / suspension / breaking-news check")
    print("  Weather is displayed automatically but remains read-only for now.")

    if args.no_save:
        print("\nNO-SAVE mode: official and counterfactual decisions were NOT written.")
    else:
        paths = live.ledger_paths(args.ledger_dir)
        print("\nOfficial V6-filtered snapshots appended to:")
        print(" ", paths["predictions"])
        print("V3-vs-V6 counterfactual snapshots appended to:")
        print(" ", Path(args.ledger_dir) / "v3_vs_v6_decision_snapshots.csv")
        print("\nIMPORTANT: keep V6_EV_POSITIVE frozen during the forward test. Do not tune it after wins/losses.")


if __name__ == "__main__":
    main()
