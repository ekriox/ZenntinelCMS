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

# API-Football (api-sports.io) — free: 100 req/day
# Sign up at: https://www.api-football.com/ (uses api-sports.io account)
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

# ── Sports Configuration ───────────────────────────────────────────────────
# The Odds API sport keys
SPORTS = {
    # ── Mexican Leagues (PRIORITY — less efficient markets) ────────────────
    "soccer_mexico_ligamx": {
        "name": "Liga MX",
        "category": "soccer",
        "markets": ["h2h", "spreads", "totals"],
        "outcomes": ["home", "draw", "away"],
        "api_football_league_id": 262,  # Liga MX in api-football
        "priority": "high",  # Less efficient market = more edge potential
    },
    "soccer_mexico_ligamx_femenil": {
        "name": "Liga MX Femenil",
        "category": "soccer",
        "markets": ["h2h"],
        "outcomes": ["home", "draw", "away"],
        "api_football_league_id": 1029,  # Liga MX Femenil
        "priority": "highest",  # Very inefficient market
    },
    # ── European Leagues ───────────────────────────────────────────────────
    "soccer_epl": {
        "name": "English Premier League",
        "category": "soccer",
        "markets": ["h2h", "spreads", "totals"],
        "outcomes": ["home", "draw", "away"],
        "api_football_league_id": 39,
        "priority": "low",  # Very efficient market
    },
    "soccer_spain_la_liga": {
        "name": "La Liga",
        "category": "soccer",
        "markets": ["h2h", "spreads", "totals"],
        "outcomes": ["home", "draw", "away"],
        "api_football_league_id": 140,
        "priority": "low",
    },
    "soccer_italy_serie_a": {
        "name": "Serie A",
        "category": "soccer",
        "markets": ["h2h", "spreads", "totals"],
        "outcomes": ["home", "draw", "away"],
        "api_football_league_id": 135,
        "priority": "low",
    },
    "soccer_germany_bundesliga": {
        "name": "Bundesliga",
        "category": "soccer",
        "markets": ["h2h", "spreads", "totals"],
        "outcomes": ["home", "draw", "away"],
        "api_football_league_id": 78,
        "priority": "low",
    },
    # ── US Sports ──────────────────────────────────────────────────────────
    "basketball_nba": {
        "name": "NBA",
        "category": "basketball",
        "markets": ["h2h", "spreads", "totals"],
        "outcomes": ["home", "away"],
        "priority": "medium",
    },
    "americanfootball_nfl": {
        "name": "NFL",
        "category": "football",
        "markets": ["h2h", "spreads", "totals"],
        "outcomes": ["home", "away"],
        "priority": "medium",
    },
    "baseball_mlb": {
        "name": "MLB",
        "category": "baseball",
        "markets": ["h2h", "spreads", "totals"],
        "outcomes": ["home", "away"],
        "priority": "medium",
    },
    "icehockey_nhl": {
        "name": "NHL",
        "category": "hockey",
        "markets": ["h2h", "spreads", "totals"],
        "outcomes": ["home", "away"],
        "priority": "medium",
    },
}

# ── Enriched Features (from API-Football) ──────────────────────────────────
# These features give us edge over basic odds-only models
ENRICHED_FEATURES = {
    "injuries": True,       # Number of injured/suspended players per team
    "venue": True,          # Stadium info (altitude, capacity, surface type)
    "h2h_detailed": True,   # Last 10 head-to-head results with scores
    "team_stats": True,     # Season stats: goals, shots, possession, cards
    "standings": True,      # Current league position, points, form string
    "weather": True,        # Temperature, wind, rain (from SoccerData API)
}

# ── Backtesting Configuration ──────────────────────────────────────────────
TEST_PERIOD_DAYS = 180  # 6 months out-of-sample
MIN_BETS = 100
BET_SIZE = 100.0

# ── Autoresearch Ratchet ───────────────────────────────────────────────────
PRIMARY_METRIC = "ROI"
