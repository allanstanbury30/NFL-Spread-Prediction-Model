# NFL Spread Prediction Model

An end-to-end NFL point-spread prediction and betting analytics project built in Python.

The system combines historical play-by-play efficiency, opponent adjustment, prior-season continuity, schedule context, sportsbook market information, and regularized machine learning to estimate expected scoring margin and evaluate potential value against the point spread.

## What the Model Predicts

The primary target is:

```text
Home Margin = Home Team Score - Away Team Score
```

The model compares its predicted home margin with the sportsbook's home point spread:

```text
Home Edge = Predicted Home Margin + Closing Home Spread
```

A positive edge favors the home side; a negative edge favors the away side.

The project is designed to evaluate pricing relative to the market rather than simply predict which team wins.

## Methodology

The modeling pipeline is built around chronological, leakage-safe forecasting.

```text
NFL Play-by-Play Data
        |
        v
Opponent-Adjusted EPA / Success Rate
        |
        v
Recent-Form Weighting
        |
        v
Prior-Season / Current-Season Blending
        |
        v
HC / OC / QB Continuity Adjustment
        |
        v
Schedule + Travel Context
        |
        v
Sportsbook Market Information
        |
        v
Regularized Ridge Regression
        |
        v
Predicted Home Margin
        |
        v
ATS Edge vs. Market Spread
```

### Time-aware design

For a target week, the model only uses information that would have been available before that week's kickoff.

Key protections include:

- Walk-forward chronological training
- No same-week or future-game results in historical predictions
- Team efficiency snapshots built only from previously completed games
- Prior-season blending during small early-season samples
- Pre-kickoff schedule and market variables only

### Opponent-adjusted efficiency

Team quality is estimated from play-by-play data using opponent-adjusted:

- Offensive EPA per play
- Defensive EPA allowed per play
- Offensive success rate
- Defensive success rate allowed

The opponent-adjustment engine uses Ridge regression to separate offense and defense effects while shrinking noisy estimates toward league average.

### Early-season continuity blending

Early-season estimates blend prior-year strength with current-year performance.

The scheduled prior-season weight begins at approximately:

```text
Week 1: 80% prior season / 20% current-or-fallback
```

The prior-season weight declines until it reaches zero by Week 8.

Head coach, offensive coordinator, and quarterback changes can further reduce the effective prior-season weight.

### Production Ridge refinements

The production Ridge workflow adds additional safeguards and adjustments, including:

- Garbage-time filtering
- Turnover-luck regression
- Iterative strength-of-schedule weighting
- Modern home-field baseline calibration
- Recent market/environment tracking
- Key-number awareness around common NFL margins
- Heavy-spread variance control
- Aggressive Ridge regularization
- Light market anchoring

## Features and Data

### Football efficiency

- Opponent-adjusted offensive EPA/play
- Opponent-adjusted defensive EPA/play allowed
- Opponent-adjusted offensive success rate
- Opponent-adjusted defensive success rate allowed
- Rolling recent efficiency
- Prior-season baseline strength

### Team continuity

- Head coach retained
- Offensive coordinator retained
- Quarterback retained

### Schedule context

- Net rest advantage
- Travel distance differential
- Divisional-game indicator
- Time-zone shift
- Home/away context

### Market information

- Closing point spread
- Game total when available
- Moneyline-derived market win probabilities when available
- Market-implied margin
- Bet-vs-close information when supplied

### Data sources

The core historical pipeline uses nflverse data through `nflreadpy`.

The project also contains support for live/pre-game context such as:

- The Odds API for market information
- Open-Meteo for weather
- SportsDataIO for optional injury, depth-chart, and player-news context

API credentials are expected through environment variables and should never be committed to the repository.

## Backtesting

The project uses chronological walk-forward backtesting instead of randomly shuffling games between training and test sets.

Typical development architecture:

```text
Initial training span: 2018-2023
Out-of-sample testing: 2024 onward
```

Each target season expands the training history using only information that would have existed at the time of prediction.

### Development results

Results recorded during model development included:

| Evaluation | Result |
| --- | ---: |
| 2025 out-of-sample ATS win rate | 53.26% |
| 2025 out-of-sample ROI | +1.83% |
| Early 2026 Ridge evaluation | 58.33% ATS |
| Later 2026 evaluation | ~54.55% ATS |

These are historical development/backtest results, not guarantees of future performance. Exact results depend on the model version, qualification threshold, market snapshot, and evaluation window.

## Repository Structure

```text
NFL-Spread-Prediction-Model/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- data/
|   `-- team_continuity_2018_2026.csv
|
|-- coaching_qb_continuity_table.py
|   `-- Builds / validates HC, OC, and QB continuity flags
|
|-- nfl_opponent_adjusted_engine.py
|   `-- Builds leakage-safe opponent-adjusted EPA and success-rate ratings
|
|-- nfl_spread_prediction_engine.py
|   `-- Core spread prediction framework and ATS evaluation logic
|
|-- prior_season_continuity_blender.py
|   `-- Blends prior- and current-season team strength
|
|-- schedule_context_features.py
|   `-- Creates rest, travel, divisional, and time-zone features
|
|-- pregame_context_live.py
|   `-- Collects read-only weather and optional injury/news context
|
|-- run_backtest_option2.py
|   `-- Historical walk-forward backtest orchestrator
|
|-- run_production_ridge_2026.py
|   `-- Main production Ridge forecasting / evaluation pipeline
|
`-- run_production_live_v6_2026.py
    `-- Development snapshot of the live V6 orchestration workflow
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/allanstanbury30/NFL-Spread-Prediction-Model.git
cd NFL-Spread-Prediction-Model
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## How to Run

### Historical backtest

```bash
python run_backtest_option2.py --continuity data/team_continuity_2018_2026.csv
```

For more control:

```bash
python run_backtest_option2.py \
  --continuity data/team_continuity_2018_2026.csv \
  --train-start 2018 \
  --train-end 2023 \
  --test-start 2024 \
  --test-end 2025 \
  --model ridge \
  --thresholds 1.0 1.5 2.0 2.5 3.0
```

### Production Ridge workflow

```bash
python run_production_ridge_2026.py \
  --continuity data/team_continuity_2018_2026.csv \
  --train-start 2018 \
  --train-end 2023 \
  --test-start 2024 \
  --test-end 2026
```

## API Configuration

The historical Ridge/backtest workflows do not require The Odds API key when using available nflverse schedule market fields.

For workflows that use The Odds API, configure the key outside the repository.

Windows PowerShell:

```powershell
setx ODDS_API_KEY "YOUR_KEY"
```

Optional SportsDataIO context:

```powershell
setx SPORTSDATAIO_API_KEY "YOUR_KEY"
```

Restart the terminal after using `setx`.

Never commit API keys to source code.

## Current Live-Runner Note

`run_production_live_v6_2026.py` is currently included as a development snapshot of the live production architecture.

That script references additional local production modules and replay artifacts that are not currently included in the public repository. As a result, the reproducible public entry points are currently:

- `run_backtest_option2.py`
- `run_production_ridge_2026.py`

The live V6 workflow can be made fully reproducible after its remaining production dependencies are added to the repository.

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- nflreadpy / nflverse
- PyArrow
- Git / GitHub

## Interview Summary

> I built an NFL point-spread prediction system in Python that uses play-by-play data to calculate opponent-adjusted EPA and success-rate metrics, combines those with organizational continuity, rest, travel, divisional context, and sportsbook market data, and feeds them into a regularized Ridge regression model. I designed the backtesting chronologically to prevent data leakage and built production workflows for evaluating model edge against the betting market.

## Disclaimer

This project is for educational, analytical, and portfolio purposes. Historical model performance does not guarantee future results.
