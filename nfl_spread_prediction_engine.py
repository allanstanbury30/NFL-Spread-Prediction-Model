"""
Component #3: NFL Spread Prediction Engine
==========================================

Purpose
-------
Finalize the NFL Point Spreads Analyzer by:
1) stitching blended opponent-adjusted team features to historical games,
2) training a time-aware regression model to predict HOME score margin,
3) comparing that predicted margin to the sportsbook's HOME spread,
4) walk-forward backtesting ATS bets using configurable edge thresholds,
5) reporting win rate, ROI, and CLV diagnostics.

Sign convention
---------------
- Target:
    home_margin = home_score - away_score

- Market spread:
    close_spread_home
    Example: Home -3.5  -> -3.5
             Home +3.5  -> +3.5

- Model ATS edge on HOME:
    home_edge = predicted_home_margin + close_spread_home

  Example:
    predicted_home_margin = +7.0
    close_spread_home     = -3.5
    home_edge             = +3.5 points

  Positive edge -> HOME side
  Negative edge -> AWAY side

Time-awareness
--------------
Walk-forward training is strictly chronological:
a prediction for kickoff T is trained only on games with kickoff < T.

Component #1 / #2 features supplied to this engine must already be time-safe:
Week N ratings may use only information available before Week N kickoff.

CLV note
--------
For a true kickoff-time model, if the wager is simulated at the official
closing spread, CLV versus the same closing line is 0 by definition.

This engine therefore supports BOTH:
- close_spread_home: the official closing spread used by the kickoff model
- bet_spread_home: the spread actually available when the simulated wager
  was placed

If bet_spread_home is absent, it defaults to close_spread_home and CLV = 0.
To evaluate early-week CLV, provide the earlier bet_spread_home separately.

Dependencies
------------
Required:
    pandas
    numpy
    scikit-learn
    pyarrow   (only if reading/writing parquet)

Optional:
    xgboost   (only if model_type="xgboost")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DEFAULT_BLENDED_FEATURES: Tuple[str, ...] = (
    "blended_adj_off_epa",
    "blended_adj_def_epa_allowed",
    "blended_adj_off_success_rate",
    "blended_adj_def_success_rate_allowed",
)


@dataclass(frozen=True)
class ModelConfig:
    model_type: Literal["ridge", "xgboost"] = "ridge"

    # Ridge regularization. Tune ONLY with walk-forward CV.
    ridge_alpha: float = 10.0

    # Minimum historical games before a prediction is allowed.
    min_train_games: int = 500

    # Optional rolling training window in seasons.
    # None = expanding history.
    max_train_seasons: Optional[int] = None

    # Betting simulation.
    edge_threshold: float = 1.5
    default_american_odds: float = -110.0

    # Optional XGBoost parameters.
    xgb_n_estimators: int = 400
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.03
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85
    xgb_reg_lambda: float = 10.0
    random_state: int = 42


# ---------------------------------------------------------------------
# BASIC UTILITIES
# ---------------------------------------------------------------------

def american_profit_per_unit(american_odds: float) -> float:
    """
    Profit returned on a 1-unit risk if the bet wins.

    -110 -> 0.90909 profit
    +120 -> 1.20 profit
    """
    if pd.isna(american_odds):
        raise ValueError("American odds cannot be NaN.")

    odds = float(american_odds)

    if odds == 0:
        raise ValueError("American odds cannot be 0.")

    if odds < 0:
        return 100.0 / abs(odds)

    return odds / 100.0


def compute_ats_result(home_margin: float, spread_home: float) -> float:
    """
    ATS result from the HOME perspective:
        1.0 = home covers
        0.0 = away covers
        NaN = push
    """
    ats_margin = float(home_margin) + float(spread_home)

    if np.isclose(ats_margin, 0.0):
        return np.nan

    return 1.0 if ats_margin > 0 else 0.0


# ---------------------------------------------------------------------
# DATASET SYNTHESIS
# ---------------------------------------------------------------------

def synthesize_dataset(
    games: pd.DataFrame,
    blended_team_ratings: pd.DataFrame,
    market: Optional[pd.DataFrame] = None,
    *,
    blended_features: Sequence[str] = DEFAULT_BLENDED_FEATURES,
) -> pd.DataFrame:
    """
    Build one row per historical game.

    Required `games` columns:
        game_id
        season
        week
        kickoff_utc
        home_team
        away_team
        home_score
        away_score

    Required `blended_team_ratings` columns:
        season
        week
        team
        blended_adj_off_epa
        blended_adj_def_epa_allowed
        blended_adj_off_success_rate
        blended_adj_def_success_rate_allowed

    Market can be:
    A) already merged into `games`, or
    B) supplied separately with at least:
        game_id
        close_spread_home

    Recommended additional market columns:
        close_total
        close_home_ml
        close_away_ml
        bet_spread_home
        bet_american_odds_home
        bet_american_odds_away

    IMPORTANT:
    close_spread_home must represent the HOME team's line.
    """
    games = games.copy()
    ratings = blended_team_ratings.copy()

    required_games = {
        "game_id", "season", "week", "kickoff_utc",
        "home_team", "away_team", "home_score", "away_score",
    }
    missing = required_games - set(games.columns)
    if missing:
        raise ValueError(
            "games missing required columns: " + ", ".join(sorted(missing))
        )

    required_ratings = {"season", "week", "team", *blended_features}
    missing = required_ratings - set(ratings.columns)
    if missing:
        raise ValueError(
            "blended_team_ratings missing required columns: "
            + ", ".join(sorted(missing))
        )

    games["kickoff_utc"] = pd.to_datetime(games["kickoff_utc"], utc=True)

    if market is not None:
        market = market.copy()

        if "game_id" not in market.columns:
            raise ValueError("market table must contain game_id.")

        # Avoid duplicate market columns if already in games.
        duplicate_market_cols = [
            c for c in market.columns
            if c != "game_id" and c in games.columns
        ]
        if duplicate_market_cols:
            market = market.drop(columns=duplicate_market_cols)

        games = games.merge(
            market,
            on="game_id",
            how="inner",
            validate="one_to_one",
        )

    if "close_spread_home" not in games.columns:
        raise ValueError(
            "Dataset needs close_spread_home. "
            "Use negative numbers for home favorites."
        )

    # If a separate wager-time line is not supplied, kickoff-model wagers
    # happen at the closing line.
    if "bet_spread_home" not in games.columns:
        games["bet_spread_home"] = games["close_spread_home"]

    # Prepare HOME ratings.
    home = ratings[
        ["season", "week", "team", *blended_features]
    ].rename(
        columns={
            "team": "home_team",
            **{f: f"home_{f}" for f in blended_features},
        }
    )

    # Prepare AWAY ratings.
    away = ratings[
        ["season", "week", "team", *blended_features]
    ].rename(
        columns={
            "team": "away_team",
            **{f: f"away_{f}" for f in blended_features},
        }
    )

    df = games.merge(
        home,
        on=["season", "week", "home_team"],
        how="left",
        validate="many_to_one",
    )

    df = df.merge(
        away,
        on=["season", "week", "away_team"],
        how="left",
        validate="many_to_one",
    )

    # Target.
    df["home_margin"] = df["home_score"] - df["away_score"]

    # The market-implied margin is easier for regression to interpret than
    # the bookmaker sign convention.
    df["market_implied_home_margin"] = -df["close_spread_home"]

    # ATS result at closing line.
    df["home_cover_close"] = [
        compute_ats_result(m, s)
        for m, s in zip(df["home_margin"], df["close_spread_home"])
    ]

    # -----------------------------------------------------------------
    # Matchup-difference features
    # Positive values are generally HOME-favorable.
    # -----------------------------------------------------------------

    h_off_epa = "home_blended_adj_off_epa"
    a_off_epa = "away_blended_adj_off_epa"
    h_def_epa = "home_blended_adj_def_epa_allowed"
    a_def_epa = "away_blended_adj_def_epa_allowed"

    h_off_sr = "home_blended_adj_off_success_rate"
    a_off_sr = "away_blended_adj_off_success_rate"
    h_def_sr = "home_blended_adj_def_success_rate_allowed"
    a_def_sr = "away_blended_adj_def_success_rate_allowed"

    df["off_epa_diff"] = df[h_off_epa] - df[a_off_epa]
    df["def_epa_diff"] = df[a_def_epa] - df[h_def_epa]
    df["net_epa_strength_diff"] = df["off_epa_diff"] + df["def_epa_diff"]

    df["off_success_diff"] = df[h_off_sr] - df[a_off_sr]
    df["def_success_diff"] = df[a_def_sr] - df[h_def_sr]
    df["net_success_strength_diff"] = (
        df["off_success_diff"] + df["def_success_diff"]
    )

    # Matchup-specific interaction features.
    # Larger = more favorable for the named offense.
    df["home_off_vs_away_def_epa"] = df[h_off_epa] + df[a_def_epa]
    df["away_off_vs_home_def_epa"] = df[a_off_epa] + df[h_def_epa]

    df["home_off_vs_away_def_success"] = df[h_off_sr] + df[a_def_sr]
    df["away_off_vs_home_def_success"] = df[a_off_sr] + df[h_def_sr]

    # Context features if rest columns are available.
    if {"home_rest_days", "away_rest_days"}.issubset(df.columns):
        df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]

    return df.sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------------------------

def default_model_features(df: pd.DataFrame) -> List[str]:
    """
    Build a safe default feature list from columns available in the dataset.

    The closing spread is included via market_implied_home_margin because this
    is explicitly a KICKOFF-TIME model.
    """
    candidates = [
        # Market.
        "market_implied_home_margin",
        "close_total",

        # Team level.
        "home_blended_adj_off_epa",
        "home_blended_adj_def_epa_allowed",
        "home_blended_adj_off_success_rate",
        "home_blended_adj_def_success_rate_allowed",
        "away_blended_adj_off_epa",
        "away_blended_adj_def_epa_allowed",
        "away_blended_adj_off_success_rate",
        "away_blended_adj_def_success_rate_allowed",

        # Differences.
        "off_epa_diff",
        "def_epa_diff",
        "net_epa_strength_diff",
        "off_success_diff",
        "def_success_diff",
        "net_success_strength_diff",

        # Matchup interactions.
        "home_off_vs_away_def_epa",
        "away_off_vs_home_def_epa",
        "home_off_vs_away_def_success",
        "away_off_vs_home_def_success",

        # Context.
        "rest_diff",
        "division_game",
        "week",
    ]

    return [c for c in candidates if c in df.columns]


# ---------------------------------------------------------------------
# MODEL FACTORY
# ---------------------------------------------------------------------

def make_model(
    feature_columns: Sequence[str],
    config: ModelConfig,
):
    """
    Create either:
    - Ridge regression pipeline, or
    - XGBoost regressor (if xgboost is installed)
    """
    numeric_features = list(feature_columns)

    if config.model_type == "ridge":
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
            ],
            remainder="drop",
        )

        return Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", Ridge(alpha=config.ridge_alpha)),
            ]
        )

    if config.model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "model_type='xgboost' requires xgboost. "
                "Install it with: pip install xgboost"
            ) from exc

        # XGBoost handles NaN natively.
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            subsample=config.xgb_subsample,
            colsample_bytree=config.xgb_colsample_bytree,
            reg_lambda=config.xgb_reg_lambda,
            random_state=config.random_state,
            n_jobs=-1,
        )

    raise ValueError(
        f"Unsupported model_type={config.model_type!r}. "
        "Use 'ridge' or 'xgboost'."
    )


# ---------------------------------------------------------------------
# WALK-FORWARD PREDICTION
# ---------------------------------------------------------------------

def walk_forward_predict(
    dataset: pd.DataFrame,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    config: ModelConfig = ModelConfig(),
) -> pd.DataFrame:
    """
    Produce strictly chronological out-of-sample predictions.

    For each unique kickoff timestamp T:
      train = all rows with kickoff_utc < T
      test  = all rows with kickoff_utc == T

    This prevents a game from training on itself or future results.

    If max_train_seasons is set, training is restricted to the most recent
    N seasons available before the target game.
    """
    df = dataset.copy().sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)

    if feature_columns is None:
        feature_columns = default_model_features(df)

    feature_columns = list(feature_columns)

    if not feature_columns:
        raise ValueError("No model features were found.")

    required = {
        "game_id", "season", "kickoff_utc", "home_margin",
        "close_spread_home",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "dataset missing required columns: " + ", ".join(sorted(missing))
        )

    predictions = []

    for kickoff, test_idx in df.groupby("kickoff_utc", sort=True).groups.items():
        test = df.loc[list(test_idx)].copy()

        train = df[df["kickoff_utc"] < kickoff].copy()

        if config.max_train_seasons is not None and not train.empty:
            target_season = int(test["season"].min())
            min_season = target_season - config.max_train_seasons
            train = train[train["season"] >= min_season].copy()

        train = train.dropna(subset=["home_margin"])

        if len(train) < config.min_train_games:
            # Not enough history yet; skip rather than produce a weak fake OOS
            # result.
            continue

        model = make_model(feature_columns, config)

        X_train = train[feature_columns]
        y_train = train["home_margin"].astype(float)

        X_test = test[feature_columns]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        block = test[
            [
                c for c in [
                    "game_id", "season", "week", "kickoff_utc",
                    "home_team", "away_team",
                    "home_score", "away_score", "home_margin",
                    "close_spread_home", "bet_spread_home",
                    "close_total",
                    "bet_american_odds_home",
                    "bet_american_odds_away",
                ]
                if c in test.columns
            ]
        ].copy()

        block["predicted_home_margin"] = pred

        # HOME edge relative to the official close.
        block["model_edge_home_close"] = (
            block["predicted_home_margin"]
            + block["close_spread_home"]
        )

        # Edge versus the actual wager-time line.
        block["model_edge_home_bet"] = (
            block["predicted_home_margin"]
            + block["bet_spread_home"]
        )

        block["train_games"] = len(train)
        block["model_type"] = config.model_type

        predictions.append(block)

    if not predictions:
        return pd.DataFrame()

    return pd.concat(predictions, ignore_index=True)


# ---------------------------------------------------------------------
# BET SIMULATION
# ---------------------------------------------------------------------

def simulate_ats_bets(
    predictions: pd.DataFrame,
    *,
    edge_threshold: float = 1.5,
    default_american_odds: float = -110.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Simulate 1-unit ATS wagers when absolute model edge meets the threshold.

    Bet rule:
        edge >= threshold  -> HOME
        edge <= -threshold -> AWAY
        otherwise          -> PASS

    The threshold is evaluated against bet_spread_home, which defaults to the
    close for a kickoff-time model.

    Returns:
        bet_ledger, summary_dict
    """
    if predictions.empty:
        return pd.DataFrame(), {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate_ex_pushes": np.nan,
            "roi": np.nan,
            "avg_clv_points": np.nan,
        }

    df = predictions.copy()

    if "bet_spread_home" not in df.columns:
        df["bet_spread_home"] = df["close_spread_home"]

    df["edge_at_bet"] = (
        df["predicted_home_margin"]
        + df["bet_spread_home"]
    )

    df["bet_side"] = np.select(
        [
            df["edge_at_bet"] >= edge_threshold,
            df["edge_at_bet"] <= -edge_threshold,
        ],
        ["HOME", "AWAY"],
        default="PASS",
    )

    bets = df[df["bet_side"] != "PASS"].copy()

    if bets.empty:
        return bets, {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate_ex_pushes": np.nan,
            "roi": np.nan,
            "avg_clv_points": np.nan,
        }

    # Actual ATS margin from HOME perspective at the bet line.
    bets["actual_home_ats_margin"] = (
        bets["home_margin"] + bets["bet_spread_home"]
    )

    # From the selected side's perspective.
    side_sign = np.where(bets["bet_side"] == "HOME", 1.0, -1.0)
    bets["selected_side_ats_margin"] = (
        bets["actual_home_ats_margin"] * side_sign
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

    # Price.
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
        pd.to_numeric(bets["american_odds"], errors="coerce")
        .fillna(default_american_odds)
    )

    bets["profit_units"] = 0.0

    win_mask = bets["bet_result"] == "WIN"
    loss_mask = bets["bet_result"] == "LOSS"

    bets.loc[win_mask, "profit_units"] = bets.loc[
        win_mask, "american_odds"
    ].map(american_profit_per_unit)

    bets.loc[loss_mask, "profit_units"] = -1.0

    # CLV in points:
    # Home example: bet -3.5, close -6.0 => +2.5 CLV.
    # Away receives the opposite sign.
    bets["clv_points"] = (
        (bets["bet_spread_home"] - bets["close_spread_home"])
        * side_sign
    )

    wins = int((bets["bet_result"] == "WIN").sum())
    losses = int((bets["bet_result"] == "LOSS").sum())
    pushes = int((bets["bet_result"] == "PUSH").sum())
    graded = wins + losses

    summary = {
        "edge_threshold": float(edge_threshold),
        "bets": int(len(bets)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate_ex_pushes": (
            float(wins / graded) if graded else np.nan
        ),
        "profit_units": float(bets["profit_units"].sum()),
        "roi": float(bets["profit_units"].sum() / len(bets)),
        "avg_clv_points": float(bets["clv_points"].mean()),
    }

    return bets, summary


# ---------------------------------------------------------------------
# MODEL ERROR METRICS
# ---------------------------------------------------------------------

def regression_metrics(predictions: pd.DataFrame) -> Dict[str, float]:
    """Out-of-sample score-margin accuracy."""
    if predictions.empty:
        return {
            "n_predictions": 0,
            "mae": np.nan,
            "rmse": np.nan,
        }

    y_true = predictions["home_margin"].astype(float)
    y_pred = predictions["predicted_home_margin"].astype(float)

    return {
        "n_predictions": int(len(predictions)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# ---------------------------------------------------------------------
# THRESHOLD SWEEP
# ---------------------------------------------------------------------

def evaluate_thresholds(
    predictions: pd.DataFrame,
    thresholds: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    *,
    default_american_odds: float = -110.0,
) -> pd.DataFrame:
    """
    Compare ATS performance across edge thresholds.

    IMPORTANT:
    Do not choose the final threshold by maximizing performance on your final
    holdout/test period. Tune it only on a validation period, then freeze it.
    """
    rows = []

    for threshold in thresholds:
        _, summary = simulate_ats_bets(
            predictions,
            edge_threshold=float(threshold),
            default_american_odds=default_american_odds,
        )
        rows.append(summary)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# OPTIONAL: SIMPLE TRAIN / VALIDATION / TEST HELPER
# ---------------------------------------------------------------------

def chronological_split_by_season(
    df: pd.DataFrame,
    *,
    train_end_season: int,
    validation_season: int,
    test_season: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience helper for architecture validation.

    Example:
        train:      <= 2022
        validation: 2023
        test:       2024

    This is separate from walk_forward_predict(), which is even stricter and
    retrains through time.
    """
    train = df[df["season"] <= train_end_season].copy()
    validation = df[df["season"] == validation_season].copy()
    test = df[df["season"] == test_season].copy()

    return train, validation, test


# ---------------------------------------------------------------------
# CLI / FILE-BASED RUNNER
# ---------------------------------------------------------------------

def read_table(path: str) -> pd.DataFrame:
    path_obj = Path(path)

    if path_obj.suffix.lower() == ".parquet":
        return pd.read_parquet(path_obj)

    if path_obj.suffix.lower() == ".csv":
        return pd.read_csv(path_obj)

    raise ValueError(
        f"Unsupported file type for {path}. Use .csv or .parquet."
    )


def main():
    """
    Example command:

    python nfl_spread_prediction_engine.py \
        --games data/processed/games.csv \
        --ratings data/processed/blended_team_ratings.parquet \
        --market data/market/nfl_market_history.csv \
        --output data/processed/spread_model_predictions.parquet \
        --model ridge \
        --edge-threshold 1.5
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="NFL Point Spreads Analyzer - Component #3"
    )

    parser.add_argument("--games", required=True)
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--market", default=None)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--model",
        choices=["ridge", "xgboost"],
        default="ridge",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=10.0,
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

    args = parser.parse_args()

    games = read_table(args.games)
    ratings = read_table(args.ratings)
    market = read_table(args.market) if args.market else None

    dataset = synthesize_dataset(
        games=games,
        blended_team_ratings=ratings,
        market=market,
    )

    features = default_model_features(dataset)

    config = ModelConfig(
        model_type=args.model,
        ridge_alpha=args.ridge_alpha,
        min_train_games=args.min_train_games,
        max_train_seasons=args.max_train_seasons,
        edge_threshold=args.edge_threshold,
    )

    predictions = walk_forward_predict(
        dataset,
        feature_columns=features,
        config=config,
    )

    metrics = regression_metrics(predictions)

    bets, bet_summary = simulate_ats_bets(
        predictions,
        edge_threshold=args.edge_threshold,
        default_american_odds=config.default_american_odds,
    )

    threshold_table = evaluate_thresholds(
        predictions,
        default_american_odds=config.default_american_odds,
    )

    print("\nMODEL FEATURES")
    print("-" * 72)
    for feature in features:
        print(feature)

    print("\nOUT-OF-SAMPLE MARGIN METRICS")
    print("-" * 72)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nATS BACKTEST")
    print("-" * 72)
    for key, value in bet_summary.items():
        print(f"{key}: {value}")

    print("\nEDGE THRESHOLD SWEEP")
    print("-" * 72)
    print(
        threshold_table.to_string(
            index=False,
            float_format=lambda x: f"{x:0.4f}",
        )
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() == ".parquet":
        predictions.to_parquet(output, index=False)
    elif output.suffix.lower() == ".csv":
        predictions.to_csv(output, index=False)
    else:
        raise ValueError("Output must end in .csv or .parquet")

    bet_output = output.with_name(
        output.stem + "_bets" + output.suffix
    )

    if bet_output.suffix.lower() == ".parquet":
        bets.to_parquet(bet_output, index=False)
    else:
        bets.to_csv(bet_output, index=False)

    threshold_output = output.with_name(
        output.stem + "_thresholds.csv"
    )
    threshold_table.to_csv(threshold_output, index=False)

    print(f"\nSaved predictions: {output}")
    print(f"Saved bet ledger:   {bet_output}")
    print(f"Saved thresholds:   {threshold_output}")


if __name__ == "__main__":
    main()
