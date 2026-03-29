"""
Constants and configuration for the Sports Betting ML Robot.
This file is NOT modified by the autoresearch agent.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
EXPERIMENTS_TSV = RESULTS_DIR / "experiments.tsv"

# ── API Keys ───────────────────────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# ── Sports Configuration ───────────────────────────────────────────────────
# The Odds API sport keys
SPORTS = {
    "soccer_epl": {
        "name": "English Premier League",
        "category": "soccer",
        "markets": ["h2h", "spreads", "totals"],
        "outcomes": ["home", "draw", "away"],
    },
    "soccer_spain_la_liga": {
        "name": "La Liga",
        "category": "soccer",
        "markets": ["h2h", "spreads", "totals"],
        "outcomes": ["home", "draw", "away"],
    },
    "soccer_italy_serie_a": {
        "name": "Serie A",
        "category": "soccer",
        "markets": ["h2h", "spreads", "totals"],
        "outcomes": ["home", "draw", "away"],
    },
    "soccer_germany_bundesliga": {
        "name": "Bundesliga",
        "category": "soccer",
        "markets": ["h2h", "spreads", "totals"],
        "outcomes": ["home", "draw", "away"],
    },
    "basketball_nba": {
        "name": "NBA",
        "category": "basketball",
        "markets": ["h2h", "spreads", "totals"],
        "outcomes": ["home", "away"],
    },
    "americanfootball_nfl": {
        "name": "NFL",
        "category": "football",
        "markets": ["h2h", "spreads", "totals"],
        "outcomes": ["home", "away"],
    },
    "baseball_mlb": {
        "name": "MLB",
        "category": "baseball",
        "markets": ["h2h", "spreads", "totals"],
        "outcomes": ["home", "away"],
    },
    "icehockey_nhl": {
        "name": "NHL",
        "category": "hockey",
        "markets": ["h2h", "spreads", "totals"],
        "outcomes": ["home", "away"],
    },
}

# ── Backtesting Configuration ──────────────────────────────────────────────
# Temporal split: test set is the last N days of data
TEST_PERIOD_DAYS = 180  # 6 months out-of-sample

# Minimum number of bets for a valid experiment
MIN_BETS = 100

# Flat bet size in dollars
BET_SIZE = 100.0

# ── Autoresearch Ratchet ───────────────────────────────────────────────────
# Primary metric: ROI (higher is better)
PRIMARY_METRIC = "ROI"
