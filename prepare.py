"""
Data pipeline for the Sports Betting ML Robot.
This file is NOT modified by the autoresearch agent.

Usage:
    python prepare.py --source sample       # Generate synthetic sample data (no API key needed)
    python prepare.py --source api          # Fetch live odds + scores (costs ~2 credits/sport)
    python prepare.py --source historical   # Fetch historical odds (costs 10 credits/sport/date)
    python prepare.py --source both         # Sample + API live data
    python prepare.py --source full         # Sample + API live + historical (uses most credits)
    python prepare.py --credits             # Show remaining API credits without fetching

All API data is cached locally. Re-running only fetches NEW data, never re-downloads.
"""

import argparse
import json
import sys
import time
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
# API HELPERS
# ═══════════════════════════════════════════════════════════════════════════

_credits_remaining = None


def api_get(endpoint: str, params: dict, label: str = "") -> dict | list | None:
    """Make a cached, credit-aware API request."""
    global _credits_remaining

    params["apiKey"] = ODDS_API_KEY

    try:
        resp = requests.get(
            f"{ODDS_API_BASE}/{endpoint}",
            params=params,
            timeout=30,
        )
    except requests.RequestException as e:
        print(f"    Network error{f' ({label})' if label else ''}: {e}")
        return None

    _credits_remaining = resp.headers.get("x-requests-remaining", _credits_remaining)
    used = resp.headers.get("x-requests-used", "?")

    if resp.status_code == 200:
        data = resp.json()
        if label:
            print(f"    {label}: {len(data) if isinstance(data, list) else 'ok'} items. "
                  f"Credits used: {used}, remaining: {_credits_remaining}")
        return data
    elif resp.status_code == 401:
        print(f"    ERROR: Invalid API key. Check your .env file.")
        return None
    elif resp.status_code == 422:
        if label:
            print(f"    {label}: not in season or unavailable. Skipping.")
        return None
    elif resp.status_code == 429:
        print(f"    Rate limited. Waiting 2s...")
        time.sleep(2)
        return api_get(endpoint, params, label)
    else:
        print(f"    Error {resp.status_code}: {resp.text[:200]}")
        return None


def check_credits() -> str | None:
    """Check remaining API credits without spending any (sports endpoint is free)."""
    if not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY not set. Add it to .env file.")
        return None

    data = api_get("sports", {"all": "true"}, label="Credit check")
    return _credits_remaining


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

        base_date = datetime(2024, 1, 1)

        for i in range(n_matches_per_sport):
            home, away = rng.choice(teams, size=2, replace=False)
            days_offset = int(i * (730 / n_matches_per_sport))
            match_date = base_date + timedelta(days=days_offset)
            home_strength = rng.normal(0.1, 0.15)

            if is_soccer:
                raw = np.array([
                    1.0 + home_strength,
                    0.6 + rng.normal(0, 0.1),
                    0.8 - home_strength,
                ])
                true_probs = raw / raw.sum()
                result = rng.choice(["home", "draw", "away"], p=true_probs)
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
                home_prob = np.clip(0.5 + home_strength * 0.5, 0.25, 0.75)
                true_probs = np.array([home_prob, 1 - home_prob])
                result = rng.choice(["home", "away"], p=true_probs)
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
                else:
                    home_score = rng.poisson(3.0)
                    away_score = rng.poisson(2.8)
                    if result == "home" and home_score <= away_score:
                        home_score = away_score + rng.randint(1, 3)
                    elif result == "away" and away_score <= home_score:
                        away_score = home_score + rng.randint(1, 3)

            vig = 1.0 + rng.uniform(0.05, 0.08)
            noise = rng.normal(0, 0.03, size=len(true_probs))
            noisy_probs = np.clip(true_probs + noise, 0.05, 0.95)
            noisy_probs = noisy_probs / noisy_probs.sum()
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
                "odds_draw": round(float(odds[1]), 2) if is_soccer else None,
                "home_form_5": round(rng.uniform(0.2, 0.8), 3),
                "away_form_5": round(rng.uniform(0.2, 0.8), 3),
                "home_form_10": round(rng.uniform(0.25, 0.75), 3),
                "away_form_10": round(rng.uniform(0.25, 0.75), 3),
                "h2h_home_wins": rng.randint(0, 8),
                "h2h_away_wins": rng.randint(0, 8),
                "h2h_draws": rng.randint(0, 5),
                "home_avg_scored": round(rng.uniform(0.8, 3.0), 2),
                "home_avg_conceded": round(rng.uniform(0.5, 2.5), 2),
                "away_avg_scored": round(rng.uniform(0.8, 3.0), 2),
                "away_avg_conceded": round(rng.uniform(0.5, 2.5), 2),
                "home_rest_days": rng.randint(2, 10),
                "away_rest_days": rng.randint(2, 10),
            }
            records.append(record)

        df = pd.DataFrame(records)
        raw_path = RAW_DIR / f"{sport_key}_sample.csv"
        df.to_csv(raw_path, index=False)
        print(f"  {sport_cfg['name']}: {len(df)} matches -> {raw_path}")

    print("Sample data generation complete.")


# ═══════════════════════════════════════════════════════════════════════════
# THE ODDS API — LIVE ODDS + SCORES
# ═══════════════════════════════════════════════════════════════════════════

def fetch_live_odds() -> None:
    """Fetch current/upcoming odds. Costs 1 credit per sport (h2h, us region)."""
    if not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY not set. Add it to .env file.")
        sys.exit(1)

    print("\n=== Fetching LIVE odds (1 credit/sport) ===")

    for sport_key, sport_cfg in SPORTS.items():
        cache_path = RAW_DIR / f"{sport_key}_live.json"

        data = api_get(
            f"sports/{sport_key}/odds",
            {"regions": "us", "markets": "h2h", "oddsFormat": "decimal"},
            label=sport_cfg["name"],
        )

        if data is not None:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)

    print(f"\nCredits remaining: {_credits_remaining}")


def fetch_scores() -> None:
    """
    Fetch completed game scores + results. Costs 1 credit per sport.
    This gives us REAL results to build train/test data with.
    """
    if not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY not set. Add it to .env file.")
        sys.exit(1)

    print("\n=== Fetching SCORES & RESULTS (1 credit/sport) ===")

    for sport_key, sport_cfg in SPORTS.items():
        cache_path = RAW_DIR / f"{sport_key}_scores.json"

        # daysFrom=3 gets completed games from the last 3 days (max allowed)
        data = api_get(
            f"sports/{sport_key}/scores",
            {"daysFrom": "3"},
            label=sport_cfg["name"],
        )

        if data is not None:
            # Merge with existing cached scores (append, deduplicate)
            existing = []
            if cache_path.exists():
                with open(cache_path) as f:
                    existing = json.load(f)

            # Merge by event ID
            seen_ids = {e["id"] for e in existing}
            for event in data:
                if event["id"] not in seen_ids:
                    existing.append(event)

            with open(cache_path, "w") as f:
                json.dump(existing, f, indent=2)

            print(f"    Total cached: {len(existing)} events")

    print(f"\nCredits remaining: {_credits_remaining}")


# ═══════════════════════════════════════════════════════════════════════════
# THE ODDS API — HISTORICAL ODDS (heavy credit usage)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_historical_odds(days_back: int = 30, sports_limit: list[str] | None = None) -> None:
    """
    Fetch historical odds snapshots.
    Costs 10 credits per sport per date per region per market.

    With 500 free credits:
      - 1 sport, 1 region, 1 market, 50 dates = 500 credits (entire budget)
      - 2 sports, 1 region, 1 market, 25 dates = 500 credits

    Strategy: fetch 1 snapshot per day (noon UTC) to maximize date coverage.
    """
    if not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY not set. Add it to .env file.")
        sys.exit(1)

    target_sports = sports_limit or list(SPORTS.keys())
    cost_estimate = len(target_sports) * days_back * 10  # 10 credits per sport per date
    print(f"\n=== Fetching HISTORICAL odds ===")
    print(f"  Sports: {len(target_sports)}, Days: {days_back}")
    print(f"  Estimated credit cost: ~{cost_estimate}")
    print(f"  Current credits: {_credits_remaining or 'unknown (run --credits first)'}")

    if _credits_remaining and int(_credits_remaining) < cost_estimate:
        print(f"\n  WARNING: Not enough credits ({_credits_remaining} < {cost_estimate}).")
        print(f"  Reducing to fit within budget...")
        days_back = max(1, int(_credits_remaining) // (len(target_sports) * 10))
        print(f"  Adjusted: {days_back} days back")

    hist_dir = RAW_DIR / "historical"
    hist_dir.mkdir(parents=True, exist_ok=True)

    for sport_key in target_sports:
        if sport_key not in SPORTS:
            continue
        sport_cfg = SPORTS[sport_key]
        print(f"\n  --- {sport_cfg['name']} ---")

        for days_ago in range(1, days_back + 1):
            target_date = datetime.utcnow() - timedelta(days=days_ago)
            date_str = target_date.strftime("%Y-%m-%dT12:00:00Z")  # noon UTC snapshot
            date_label = target_date.strftime("%Y-%m-%d")

            # Check cache — skip if already downloaded
            cache_path = hist_dir / f"{sport_key}_{date_label}.json"
            if cache_path.exists():
                print(f"    {date_label}: cached, skipping")
                continue

            data = api_get(
                f"sports/{sport_key}/odds-history",
                {
                    "regions": "us",
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                    "date": date_str,
                },
                label=f"{date_label}",
            )

            if data is not None:
                with open(cache_path, "w") as f:
                    json.dump(data, f, indent=2)

            # Small delay to avoid rate limiting
            time.sleep(0.1)

    print(f"\nHistorical fetch complete. Credits remaining: {_credits_remaining}")


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING & PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def process_data() -> None:
    """Process all raw data (sample + API) into train/test parquet files."""
    print("\n=== Processing data into train/test splits ===")

    for sport_key, sport_cfg in SPORTS.items():
        frames = []

        # 1. Load sample data (synthetic)
        sample_path = RAW_DIR / f"{sport_key}_sample.csv"
        if sample_path.exists():
            frames.append(pd.read_csv(sample_path))

        # 2. Load live API data
        live_path = RAW_DIR / f"{sport_key}_live.json"
        if live_path.exists():
            live_df = _parse_odds_json(live_path, sport_key)
            if live_df is not None and not live_df.empty:
                frames.append(live_df)

        # 3. Load scores (real results)
        scores_path = RAW_DIR / f"{sport_key}_scores.json"
        if scores_path.exists():
            scores_df = _parse_scores_json(scores_path, sport_key)
            if scores_df is not None and not scores_df.empty:
                frames.append(scores_df)

        # 4. Load historical odds
        hist_dir = RAW_DIR / "historical"
        if hist_dir.exists():
            for hist_file in sorted(hist_dir.glob(f"{sport_key}_*.json")):
                hist_df = _parse_historical_json(hist_file, sport_key)
                if hist_df is not None and not hist_df.empty:
                    frames.append(hist_df)

        if not frames:
            print(f"  No data for {sport_cfg['name']}, skipping.")
            continue

        # Combine all sources, deduplicate by date+teams
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Deduplicate: prefer rows with real results over placeholders
        df["has_result"] = df["result"].notna().astype(int)
        df = df.sort_values(["date", "home_team", "away_team", "has_result"], ascending=[True, True, True, False])
        df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
        df = df.drop(columns=["has_result"])

        # ── Feature engineering ─────────────────────────────────────────
        df = _engineer_features(df, sport_cfg)

        # ── Temporal split ──────────────────────────────────────────────
        cutoff = df["date"].max() - timedelta(days=TEST_PERIOD_DAYS)
        train = df[df["date"] < cutoff].copy()
        test = df[df["date"] >= cutoff].copy()

        train_path = PROCESSED_DIR / f"{sport_key}_train.parquet"
        test_path = PROCESSED_DIR / f"{sport_key}_test.parquet"
        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)

        print(f"  {sport_cfg['name']}: {len(train)} train / {len(test)} test "
              f"({len(df)} total, {df['result'].notna().sum()} with results)")

    print("\nData processing complete.")


def _engineer_features(df: pd.DataFrame, sport_cfg: dict) -> pd.DataFrame:
    """Apply feature engineering to a combined DataFrame."""
    is_soccer = sport_cfg["category"] == "soccer"

    # Fill missing odds with reasonable defaults
    df["odds_home"] = pd.to_numeric(df["odds_home"], errors="coerce").fillna(2.0)
    df["odds_away"] = pd.to_numeric(df["odds_away"], errors="coerce").fillna(2.0)
    if is_soccer:
        df["odds_draw"] = pd.to_numeric(df["odds_draw"], errors="coerce").fillna(3.5)

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

    # Fill form/stats features with defaults if missing
    for col, default in [
        ("home_form_5", 0.5), ("away_form_5", 0.5),
        ("home_form_10", 0.5), ("away_form_10", 0.5),
        ("h2h_home_wins", 0), ("h2h_away_wins", 0), ("h2h_draws", 0),
        ("home_avg_scored", 1.5), ("home_avg_conceded", 1.2),
        ("away_avg_scored", 1.3), ("away_avg_conceded", 1.4),
        ("home_rest_days", 4), ("away_rest_days", 4),
    ]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    # Computed features
    df["form_diff_5"] = df["home_form_5"] - df["away_form_5"]
    df["form_diff_10"] = df["home_form_10"] - df["away_form_10"]

    total_h2h = df["h2h_home_wins"] + df["h2h_away_wins"] + df["h2h_draws"] + 1
    df["h2h_home_ratio"] = df["h2h_home_wins"] / total_h2h
    df["h2h_away_ratio"] = df["h2h_away_wins"] / total_h2h

    df["home_goal_diff"] = df["home_avg_scored"] - df["home_avg_conceded"]
    df["away_goal_diff"] = df["away_avg_scored"] - df["away_avg_conceded"]
    df["scoring_diff"] = df["home_goal_diff"] - df["away_goal_diff"]

    df["rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]
    df["odds_ratio"] = df["odds_home"] / df["odds_away"]

    # Encode result
    if is_soccer:
        df["target"] = df["result"].map({"home": 0, "draw": 1, "away": 2})
    else:
        df["target"] = df["result"].map({"home": 0, "away": 1})

    return df


# ═══════════════════════════════════════════════════════════════════════════
# PARSERS FOR DIFFERENT API RESPONSE FORMATS
# ═══════════════════════════════════════════════════════════════════════════

def _parse_odds_json(path: Path, sport_key: str) -> pd.DataFrame | None:
    """Parse live odds JSON (v4/sports/{sport}/odds)."""
    with open(path) as f:
        events = json.load(f)

    if not events:
        return None

    records = []
    for event in events:
        record = _extract_odds_from_event(event, sport_key)
        if record:
            records.append(record)

    return pd.DataFrame(records) if records else None


def _parse_scores_json(path: Path, sport_key: str) -> pd.DataFrame | None:
    """Parse scores JSON (v4/sports/{sport}/scores) — gives us REAL results."""
    with open(path) as f:
        events = json.load(f)

    if not events:
        return None

    records = []
    for event in events:
        if not event.get("completed"):
            continue

        home = event.get("home_team", "")
        away = event.get("away_team", "")
        date = event.get("commence_time", "")[:10]

        scores = event.get("scores", [])
        home_score = away_score = None
        for s in scores:
            if s.get("name") == home:
                home_score = int(s.get("score", 0))
            elif s.get("name") == away:
                away_score = int(s.get("score", 0))

        if home_score is None or away_score is None:
            continue

        if home_score > away_score:
            result = "home"
        elif away_score > home_score:
            result = "away"
        else:
            result = "draw"

        records.append({
            "sport": sport_key,
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "result": result,
            "odds_home": None,  # scores endpoint doesn't include odds
            "odds_draw": None,
            "odds_away": None,
        })

    return pd.DataFrame(records) if records else None


def _parse_historical_json(path: Path, sport_key: str) -> pd.DataFrame | None:
    """Parse historical odds JSON (v4/sports/{sport}/odds-history)."""
    with open(path) as f:
        payload = json.load(f)

    # Historical endpoint wraps data differently
    if isinstance(payload, dict):
        events = payload.get("data", [])
    elif isinstance(payload, list):
        events = payload
    else:
        return None

    if not events:
        return None

    records = []
    for event in events:
        record = _extract_odds_from_event(event, sport_key)
        if record:
            records.append(record)

    return pd.DataFrame(records) if records else None


def _extract_odds_from_event(event: dict, sport_key: str) -> dict | None:
    """Extract a standardized record from an odds API event."""
    home = event.get("home_team", "")
    away = event.get("away_team", "")
    date = event.get("commence_time", "")[:10]

    if not home or not away or not date:
        return None

    # Find best odds across all bookmakers
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
        return None

    return {
        "sport": sport_key,
        "date": date,
        "home_team": home,
        "away_team": away,
        "home_score": None,
        "away_score": None,
        "result": None,
        "odds_home": best_odds.get("home"),
        "odds_draw": best_odds.get("draw"),
        "odds_away": best_odds.get("away"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sports Betting Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare.py --source sample          # Free, no API key needed
  python prepare.py --credits                # Check remaining API credits
  python prepare.py --source api             # Fetch live odds + scores (~16 credits)
  python prepare.py --source historical --days 7 --sports soccer_epl basketball_nba
  python prepare.py --source full --days 30  # Everything (expensive on free tier)
        """,
    )
    parser.add_argument(
        "--source",
        choices=["sample", "api", "historical", "both", "full"],
        default="sample",
        help="Data source (default: sample)",
    )
    parser.add_argument("--matches", type=int, default=500, help="Sample matches per sport")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data to fetch")
    parser.add_argument("--sports", nargs="*", help="Limit to specific sport keys (e.g. soccer_epl basketball_nba)")
    parser.add_argument("--credits", action="store_true", help="Just check remaining API credits")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.credits:
        remaining = check_credits()
        if remaining:
            print(f"\n  API credits remaining: {remaining}")
        sys.exit(0)

    if args.source in ("sample", "both", "full"):
        generate_sample_data(args.matches)

    if args.source in ("api", "both", "full"):
        # Check credits first
        check_credits()
        fetch_live_odds()
        fetch_scores()

    if args.source in ("historical", "full"):
        if _credits_remaining is None:
            check_credits()
        fetch_historical_odds(days_back=args.days, sports_limit=args.sports)

    process_data()
