"""
Data pipeline for the Sports Betting ML Robot.
This file is NOT modified by the autoresearch agent.

Usage:
    python prepare.py --source sample    # Generate synthetic sample data (no API key needed)
    python prepare.py --source api       # Fetch from The Odds API (requires ODDS_API_KEY in .env)
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from constants import (
    BET_SIZE,
    ODDS_API_BASE,
    ODDS_API_KEY,
    PROCESSED_DIR,
    RAW_DIR,
    SPORTS,
    TEST_PERIOD_DAYS,
)


# ═══════════════════════════════════════════════════════════════════════════
# SAMPLE DATA GENERATOR (works without API key)
# ═══════════════════════════════════════════════════════════════════════════

# Realistic team pools per sport
TEAMS = {
    "soccer_epl": [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich Town",
        "Leicester", "Liverpool", "Man City", "Man United", "Newcastle",
        "Nottingham Forest", "Southampton", "Tottenham", "West Ham", "Wolves",
    ],
    "soccer_spain_la_liga": [
        "Real Madrid", "Barcelona", "Atletico Madrid", "Athletic Bilbao",
        "Real Sociedad", "Villarreal", "Real Betis", "Sevilla", "Valencia",
        "Girona", "Celta Vigo", "Osasuna", "Mallorca", "Rayo Vallecano",
        "Getafe", "Las Palmas", "Alaves", "Espanyol", "Leganes", "Valladolid",
    ],
    "soccer_italy_serie_a": [
        "Inter Milan", "AC Milan", "Juventus", "Napoli", "Atalanta",
        "Roma", "Lazio", "Fiorentina", "Bologna", "Torino",
        "Monza", "Udinese", "Sassuolo", "Empoli", "Cagliari",
        "Genoa", "Lecce", "Verona", "Frosinone", "Salernitana",
    ],
    "soccer_germany_bundesliga": [
        "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen",
        "Union Berlin", "Freiburg", "Eintracht Frankfurt", "Wolfsburg",
        "Hoffenheim", "Werder Bremen", "Mainz", "Augsburg", "Monchengladbach",
        "Koln", "Stuttgart", "Heidenheim", "Darmstadt", "Bochum",
    ],
    "basketball_nba": [
        "Lakers", "Celtics", "Warriors", "Bucks", "76ers", "Nuggets",
        "Heat", "Suns", "Knicks", "Mavericks", "Clippers", "Kings",
        "Nets", "Grizzlies", "Cavaliers", "Hawks", "Bulls", "Raptors",
        "Timberwolves", "Thunder", "Pelicans", "Trail Blazers", "Pacers",
        "Jazz", "Wizards", "Hornets", "Pistons", "Magic", "Rockets", "Spurs",
    ],
    "americanfootball_nfl": [
        "Chiefs", "49ers", "Eagles", "Bills", "Cowboys", "Ravens",
        "Lions", "Dolphins", "Bengals", "Packers", "Seahawks", "Texans",
        "Jets", "Browns", "Steelers", "Jaguars", "Vikings", "Chargers",
        "Rams", "Saints", "Bears", "Colts", "Broncos", "Raiders",
        "Commanders", "Giants", "Patriots", "Titans", "Cardinals", "Falcons",
        "Panthers", "Buccaneers",
    ],
    "baseball_mlb": [
        "Yankees", "Dodgers", "Astros", "Braves", "Phillies", "Rangers",
        "Orioles", "Twins", "Rays", "Mariners", "Blue Jays", "Padres",
        "Cardinals", "Brewers", "Mets", "Diamondbacks", "Cubs", "Red Sox",
        "Giants", "Guardians", "Reds", "Pirates", "Tigers", "Royals",
        "Marlins", "White Sox", "Angels", "Rockies", "Nationals", "Athletics",
    ],
    "icehockey_nhl": [
        "Bruins", "Panthers", "Avalanche", "Stars", "Rangers", "Oilers",
        "Hurricanes", "Devils", "Golden Knights", "Kings", "Maple Leafs",
        "Wild", "Jets", "Kraken", "Lightning", "Flames", "Capitals",
        "Islanders", "Canucks", "Penguins", "Sabres", "Red Wings",
        "Senators", "Predators", "Blues", "Flyers", "Coyotes", "Ducks",
        "Blackhawks", "Sharks", "Blue Jackets", "Canadiens",
    ],
}


def generate_sample_data(n_matches_per_sport: int = 500) -> None:
    """Generate realistic synthetic sports betting data for all sports."""
    rng = np.random.RandomState(42)
    print(f"Generating {n_matches_per_sport} sample matches per sport...")

    for sport_key, sport_cfg in SPORTS.items():
        teams = TEAMS.get(sport_key, [f"Team_{i}" for i in range(20)])
        is_soccer = sport_cfg["category"] == "soccer"
        records = []

        # Generate matches over ~2 years
        base_date = datetime(2024, 1, 1)

        for i in range(n_matches_per_sport):
            # Pick two different teams
            home, away = rng.choice(teams, size=2, replace=False)

            # Game date (spread over ~2 years)
            days_offset = int(i * (730 / n_matches_per_sport))
            match_date = base_date + timedelta(days=days_offset)

            # Generate "true" probabilities (hidden, what the model tries to learn)
            # Home advantage + some team strength noise
            home_strength = rng.normal(0.1, 0.15)  # slight home advantage

            if is_soccer:
                # Soccer: home/draw/away
                raw = np.array([
                    1.0 + home_strength,
                    0.6 + rng.normal(0, 0.1),
                    0.8 - home_strength,
                ])
                true_probs = raw / raw.sum()

                # Generate result
                result = rng.choice(["home", "draw", "away"], p=true_probs)

                # Generate scores
                if result == "home":
                    home_score = rng.choice([1, 2, 3, 4], p=[0.25, 0.40, 0.25, 0.10])
                    away_score = rng.choice([0, 1, 2], p=[0.50, 0.35, 0.15])
                elif result == "draw":
                    s = rng.choice([0, 1, 2, 3], p=[0.20, 0.45, 0.25, 0.10])
                    home_score = away_score = s
                else:
                    away_score = rng.choice([1, 2, 3, 4], p=[0.25, 0.40, 0.25, 0.10])
                    home_score = rng.choice([0, 1, 2], p=[0.50, 0.35, 0.15])
            else:
                # Non-soccer: home/away only
                home_prob = 0.5 + home_strength * 0.5
                home_prob = np.clip(home_prob, 0.25, 0.75)
                true_probs = np.array([home_prob, 1 - home_prob])

                result = rng.choice(["home", "away"], p=true_probs)

                # Generate scores based on sport
                if sport_cfg["category"] == "basketball":
                    base = rng.normal(110, 10)
                    margin = rng.exponential(8) * (1 if result == "home" else -1)
                    home_score = int(max(80, base + margin / 2))
                    away_score = int(max(80, base - margin / 2))
                elif sport_cfg["category"] == "football":
                    scores = [3, 7, 10, 13, 14, 17, 20, 21, 24, 27, 28, 31, 34]
                    home_score = rng.choice(scores)
                    away_score = rng.choice(scores)
                    if result == "home" and home_score <= away_score:
                        home_score = away_score + rng.choice([3, 7])
                    elif result == "away" and away_score <= home_score:
                        away_score = home_score + rng.choice([3, 7])
                elif sport_cfg["category"] == "baseball":
                    home_score = rng.poisson(4.5)
                    away_score = rng.poisson(4.2)
                    if result == "home" and home_score <= away_score:
                        home_score = away_score + rng.randint(1, 4)
                    elif result == "away" and away_score <= home_score:
                        away_score = home_score + rng.randint(1, 4)
                else:  # hockey
                    home_score = rng.poisson(3.0)
                    away_score = rng.poisson(2.8)
                    if result == "home" and home_score <= away_score:
                        home_score = away_score + rng.randint(1, 3)
                    elif result == "away" and away_score <= home_score:
                        away_score = home_score + rng.randint(1, 3)

            # Generate bookmaker odds (add vig/juice ~5-8%)
            vig = 1.0 + rng.uniform(0.05, 0.08)
            noise = rng.normal(0, 0.03, size=len(true_probs))
            noisy_probs = np.clip(true_probs + noise, 0.05, 0.95)
            noisy_probs = noisy_probs / noisy_probs.sum()
            # Odds = 1/prob * (1/vig) — but display as decimal odds
            odds = 1.0 / (noisy_probs * vig)

            record = {
                "sport": sport_key,
                "date": match_date.strftime("%Y-%m-%d"),
                "home_team": home,
                "away_team": away,
                "home_score": int(home_score),
                "away_score": int(away_score),
                "result": result,
                "odds_home": round(float(odds[0]), 2),
                "odds_away": round(float(odds[-1]), 2),
            }

            if is_soccer:
                record["odds_draw"] = round(float(odds[1]), 2)
            else:
                record["odds_draw"] = None

            # Add some basic features the model can use
            record["home_form_5"] = round(rng.uniform(0.2, 0.8), 3)  # win rate last 5
            record["away_form_5"] = round(rng.uniform(0.2, 0.8), 3)
            record["home_form_10"] = round(rng.uniform(0.25, 0.75), 3)
            record["away_form_10"] = round(rng.uniform(0.25, 0.75), 3)
            record["h2h_home_wins"] = rng.randint(0, 8)
            record["h2h_away_wins"] = rng.randint(0, 8)
            record["h2h_draws"] = rng.randint(0, 5)
            record["home_avg_scored"] = round(rng.uniform(0.8, 3.0), 2)
            record["home_avg_conceded"] = round(rng.uniform(0.5, 2.5), 2)
            record["away_avg_scored"] = round(rng.uniform(0.8, 3.0), 2)
            record["away_avg_conceded"] = round(rng.uniform(0.5, 2.5), 2)
            record["home_rest_days"] = rng.randint(2, 10)
            record["away_rest_days"] = rng.randint(2, 10)

            records.append(record)

        df = pd.DataFrame(records)
        raw_path = RAW_DIR / f"{sport_key}_sample.csv"
        df.to_csv(raw_path, index=False)
        print(f"  {sport_cfg['name']}: {len(df)} matches → {raw_path}")

    print("Sample data generation complete.")


# ═══════════════════════════════════════════════════════════════════════════
# THE ODDS API FETCHER
# ═══════════════════════════════════════════════════════════════════════════

def fetch_odds_api_data() -> None:
    """Fetch historical and upcoming odds from The Odds API."""
    if not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY not set. Add it to .env file.")
        print("Get your free key at: https://the-odds-api.com/")
        sys.exit(1)

    print(f"Fetching data from The Odds API...")

    for sport_key, sport_cfg in SPORTS.items():
        print(f"\n  Fetching {sport_cfg['name']}...")

        # Fetch upcoming/recent odds
        try:
            resp = requests.get(
                f"{ODDS_API_BASE}/sports/{sport_key}/odds",
                params={
                    "apiKey": ODDS_API_KEY,
                    "regions": "us,eu",
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                },
                timeout=30,
            )

            if resp.status_code == 200:
                data = resp.json()
                raw_path = RAW_DIR / f"{sport_key}_api.json"
                with open(raw_path, "w") as f:
                    json.dump(data, f, indent=2)

                remaining = resp.headers.get("x-requests-remaining", "?")
                print(f"    Got {len(data)} events. API requests remaining: {remaining}")
            elif resp.status_code == 401:
                print(f"    Invalid API key. Check your .env file.")
            elif resp.status_code == 422:
                print(f"    Sport {sport_key} not currently in season. Skipping.")
            else:
                print(f"    Error {resp.status_code}: {resp.text[:200]}")

        except requests.RequestException as e:
            print(f"    Network error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING & PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def process_data() -> None:
    """Process raw data into train/test parquet files with features."""
    print("\nProcessing data into train/test splits...")

    for sport_key, sport_cfg in SPORTS.items():
        # Try sample data first, then API data
        sample_path = RAW_DIR / f"{sport_key}_sample.csv"
        api_path = RAW_DIR / f"{sport_key}_api.json"

        if sample_path.exists():
            df = pd.read_csv(sample_path)
        elif api_path.exists():
            df = _parse_api_data(api_path, sport_key)
            if df is None or df.empty:
                continue
        else:
            print(f"  No data for {sport_cfg['name']}, skipping.")
            continue

        # Ensure date column is datetime
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # ── Feature engineering ─────────────────────────────────────────
        is_soccer = sport_cfg["category"] == "soccer"

        # Implied probabilities from odds (remove vig)
        if is_soccer:
            total_implied = (1 / df["odds_home"]) + (1 / df["odds_draw"]) + (1 / df["odds_away"])
            df["implied_prob_home"] = (1 / df["odds_home"]) / total_implied
            df["implied_prob_draw"] = (1 / df["odds_draw"]) / total_implied
            df["implied_prob_away"] = (1 / df["odds_away"]) / total_implied
        else:
            total_implied = (1 / df["odds_home"]) + (1 / df["odds_away"])
            df["implied_prob_home"] = (1 / df["odds_home"]) / total_implied
            df["implied_prob_away"] = (1 / df["odds_away"]) / total_implied
            df["implied_prob_draw"] = 0.0

        # Form differential
        df["form_diff_5"] = df["home_form_5"] - df["away_form_5"]
        df["form_diff_10"] = df["home_form_10"] - df["away_form_10"]

        # H2H ratio
        total_h2h = df["h2h_home_wins"] + df["h2h_away_wins"] + df["h2h_draws"] + 1  # +1 smoothing
        df["h2h_home_ratio"] = df["h2h_home_wins"] / total_h2h
        df["h2h_away_ratio"] = df["h2h_away_wins"] / total_h2h

        # Scoring differential
        df["home_goal_diff"] = df["home_avg_scored"] - df["home_avg_conceded"]
        df["away_goal_diff"] = df["away_avg_scored"] - df["away_avg_conceded"]
        df["scoring_diff"] = df["home_goal_diff"] - df["away_goal_diff"]

        # Rest advantage
        df["rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]

        # Odds movement proxy (ratio of home/away odds)
        df["odds_ratio"] = df["odds_home"] / df["odds_away"]

        # ── Encode result as numeric target ─────────────────────────────
        # For soccer: home=0, draw=1, away=2
        # For others: home=0, away=1
        if is_soccer:
            df["target"] = df["result"].map({"home": 0, "draw": 1, "away": 2})
        else:
            df["target"] = df["result"].map({"home": 0, "away": 1})

        # ── Temporal split ──────────────────────────────────────────────
        cutoff = df["date"].max() - timedelta(days=TEST_PERIOD_DAYS)
        train = df[df["date"] < cutoff].copy()
        test = df[df["date"] >= cutoff].copy()

        # Save
        train_path = PROCESSED_DIR / f"{sport_key}_train.parquet"
        test_path = PROCESSED_DIR / f"{sport_key}_test.parquet"
        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)

        print(f"  {sport_cfg['name']}: {len(train)} train / {len(test)} test → {PROCESSED_DIR}")

    print("\nData processing complete.")


def _parse_api_data(api_path: Path, sport_key: str) -> pd.DataFrame | None:
    """Parse The Odds API JSON into a DataFrame matching our schema."""
    with open(api_path) as f:
        events = json.load(f)

    if not events:
        return None

    records = []
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        date = event.get("commence_time", "")[:10]

        # Extract best odds from bookmakers
        best_odds = {}
        for bm in event.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market["key"] == "h2h":
                    for outcome in market["outcomes"]:
                        name = outcome["name"]
                        price = outcome["price"]
                        if name == home:
                            best_odds["home"] = max(best_odds.get("home", 0), price)
                        elif name == away:
                            best_odds["away"] = max(best_odds.get("away", 0), price)
                        elif name == "Draw":
                            best_odds["draw"] = max(best_odds.get("draw", 0), price)

        if "home" not in best_odds or "away" not in best_odds:
            continue

        records.append({
            "sport": sport_key,
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_score": None,  # not available from odds endpoint
            "away_score": None,
            "result": None,  # unknown for upcoming games
            "odds_home": best_odds.get("home", 0),
            "odds_draw": best_odds.get("draw"),
            "odds_away": best_odds.get("away", 0),
            # Placeholder features (need historical data to compute properly)
            "home_form_5": 0.5,
            "away_form_5": 0.5,
            "home_form_10": 0.5,
            "away_form_10": 0.5,
            "h2h_home_wins": 0,
            "h2h_away_wins": 0,
            "h2h_draws": 0,
            "home_avg_scored": 1.5,
            "home_avg_conceded": 1.2,
            "away_avg_scored": 1.3,
            "away_avg_conceded": 1.4,
            "home_rest_days": 4,
            "away_rest_days": 4,
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sports Betting Data Pipeline")
    parser.add_argument(
        "--source",
        choices=["sample", "api", "both"],
        default="sample",
        help="Data source: 'sample' for synthetic data, 'api' for The Odds API, 'both' for both",
    )
    parser.add_argument(
        "--matches",
        type=int,
        default=500,
        help="Number of sample matches per sport (default: 500)",
    )
    args = parser.parse_args()

    # Ensure directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.source in ("sample", "both"):
        generate_sample_data(args.matches)

    if args.source in ("api", "both"):
        fetch_odds_api_data()

    process_data()
