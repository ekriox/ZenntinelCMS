"""
Data enrichment module — fetches additional features from API-Football.
This file is NOT modified by the autoresearch agent.

Provides: injuries, venue/stadium info, detailed H2H, team season stats,
league standings, and weather data.

These are the features that give us edge over basic odds-only models,
especially in less efficient markets like Liga MX and Liga MX Femenil.

Usage:
    python enrich.py --league 262          # Enrich Liga MX data
    python enrich.py --league 262 --h2h    # Fetch head-to-head history
    python enrich.py --league 262 --all    # Fetch everything
    python enrich.py --credits             # Check API-Football credits

API-Football free tier: 100 requests/day. Cache aggressively.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from constants import API_FOOTBALL_BASE, API_FOOTBALL_KEY, RAW_DIR, SPORTS


ENRICH_DIR = RAW_DIR / "enriched"


def api_football_get(endpoint: str, params: dict, label: str = "") -> dict | None:
    """Make a request to API-Football with caching."""
    if not API_FOOTBALL_KEY:
        print("ERROR: API_FOOTBALL_KEY not set. Add it to .env")
        print("Sign up free at: https://www.api-football.com/")
        return None

    headers = {"x-apisports-key": API_FOOTBALL_KEY}

    try:
        resp = requests.get(
            f"{API_FOOTBALL_BASE}/{endpoint}",
            params=params,
            headers=headers,
            timeout=30,
        )
    except requests.RequestException as e:
        print(f"    Network error: {e}")
        return None

    if resp.status_code == 200:
        data = resp.json()
        remaining = data.get("errors", {})
        if remaining:
            print(f"    API error: {remaining}")
            return None

        results = data.get("response", [])
        requests_info = data.get("paging", {})
        if label:
            print(f"    {label}: {len(results)} results")
        return data
    else:
        print(f"    HTTP {resp.status_code}: {resp.text[:200]}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# INJURIES — Key players missing = massive impact on result
# ═══════════════════════════════════════════════════════════════════════════

def fetch_injuries(league_id: int, season: int = 2025) -> None:
    """Fetch current injuries for all teams in a league."""
    ENRICH_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = ENRICH_DIR / f"injuries_{league_id}_{season}.json"

    print(f"\n=== Fetching injuries for league {league_id} ===")

    data = api_football_get(
        "injuries",
        {"league": league_id, "season": season},
        label="Injuries",
    )

    if data:
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
        injuries = data.get("response", [])
        # Group by team
        teams = {}
        for inj in injuries:
            team = inj.get("team", {}).get("name", "Unknown")
            teams.setdefault(team, []).append(inj)
        for team, injs in sorted(teams.items()):
            print(f"    {team}: {len(injs)} injured/doubtful")


def parse_injuries(league_id: int, season: int = 2025) -> dict:
    """Parse cached injuries into a team -> count mapping."""
    cache_path = ENRICH_DIR / f"injuries_{league_id}_{season}.json"
    if not cache_path.exists():
        return {}

    with open(cache_path) as f:
        data = json.load(f)

    team_injuries = {}
    for inj in data.get("response", []):
        team_name = inj.get("team", {}).get("name", "")
        player_type = inj.get("player", {}).get("type", "")
        # Weight by type: Missing > Doubtful > Questionable
        weight = {"Missing": 1.0, "Doubtful": 0.7, "Questionable": 0.3}.get(player_type, 0.5)
        team_injuries[team_name] = team_injuries.get(team_name, 0) + weight

    return team_injuries


# ═══════════════════════════════════════════════════════════════════════════
# VENUES — Altitude, surface, capacity matter
# ═══════════════════════════════════════════════════════════════════════════

def fetch_venues(league_id: int, season: int = 2025) -> None:
    """Fetch venue info for all teams in a league."""
    ENRICH_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = ENRICH_DIR / f"venues_{league_id}.json"

    if cache_path.exists():
        print(f"  Venues for league {league_id}: cached")
        return

    print(f"\n=== Fetching venues for league {league_id} ===")

    # First get teams
    teams_data = api_football_get(
        "teams",
        {"league": league_id, "season": season},
        label="Teams",
    )

    if not teams_data:
        return

    venues = []
    for item in teams_data.get("response", []):
        team = item.get("team", {})
        venue = item.get("venue", {})
        venues.append({
            "team_id": team.get("id"),
            "team_name": team.get("name"),
            "venue_name": venue.get("name"),
            "city": venue.get("city"),
            "capacity": venue.get("capacity"),
            "surface": venue.get("surface"),  # grass, artificial turf, etc.
        })
        print(f"    {team.get('name')}: {venue.get('name')} ({venue.get('surface')}, cap: {venue.get('capacity')})")

    with open(cache_path, "w") as f:
        json.dump(venues, f, indent=2)


def parse_venues(league_id: int) -> dict:
    """Parse cached venues into team -> venue info mapping."""
    cache_path = ENRICH_DIR / f"venues_{league_id}.json"
    if not cache_path.exists():
        return {}

    with open(cache_path) as f:
        venues = json.load(f)

    return {
        v["team_name"]: {
            "capacity": v.get("capacity", 0) or 0,
            "surface": v.get("surface", "grass"),
        }
        for v in venues
    }


# ═══════════════════════════════════════════════════════════════════════════
# HEAD-TO-HEAD — Historical matchups between specific teams
# ═══════════════════════════════════════════════════════════════════════════

def fetch_h2h(team1_id: int, team2_id: int) -> list:
    """Fetch last 10 H2H matches between two teams."""
    cache_path = ENRICH_DIR / f"h2h_{min(team1_id, team2_id)}_{max(team1_id, team2_id)}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    data = api_football_get(
        "fixtures/headtohead",
        {"h2h": f"{team1_id}-{team2_id}", "last": 10},
        label=f"H2H {team1_id} vs {team2_id}",
    )

    if data:
        results = data.get("response", [])
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)
        return results

    return []


def parse_h2h(h2h_matches: list, home_team: str, away_team: str) -> dict:
    """Parse H2H matches into features."""
    if not h2h_matches:
        return {
            "h2h_total_matches": 0,
            "h2h_home_wins_detailed": 0,
            "h2h_away_wins_detailed": 0,
            "h2h_draws_detailed": 0,
            "h2h_home_avg_goals": 0,
            "h2h_away_avg_goals": 0,
        }

    home_wins = away_wins = draws = 0
    home_goals_total = away_goals_total = 0

    for match in h2h_matches:
        teams = match.get("teams", {})
        goals = match.get("goals", {})

        h_goals = goals.get("home", 0) or 0
        a_goals = goals.get("away", 0) or 0

        # Figure out which team is which in this match
        match_home = teams.get("home", {}).get("name", "")
        if match_home == home_team:
            home_goals_total += h_goals
            away_goals_total += a_goals
        else:
            home_goals_total += a_goals
            away_goals_total += h_goals

        if h_goals > a_goals:
            if match_home == home_team:
                home_wins += 1
            else:
                away_wins += 1
        elif a_goals > h_goals:
            if match_home == home_team:
                away_wins += 1
            else:
                home_wins += 1
        else:
            draws += 1

    n = len(h2h_matches)
    return {
        "h2h_total_matches": n,
        "h2h_home_wins_detailed": home_wins,
        "h2h_away_wins_detailed": away_wins,
        "h2h_draws_detailed": draws,
        "h2h_home_avg_goals": round(home_goals_total / max(n, 1), 2),
        "h2h_away_avg_goals": round(away_goals_total / max(n, 1), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# TEAM SEASON STATS — Goals, shots, possession, cards, form
# ═══════════════════════════════════════════════════════════════════════════

def fetch_team_stats(league_id: int, season: int = 2025) -> None:
    """Fetch season statistics for all teams in a league."""
    ENRICH_DIR.mkdir(parents=True, exist_ok=True)

    # First get team list
    teams_data = api_football_get(
        "teams",
        {"league": league_id, "season": season},
        label=f"Teams (league {league_id})",
    )

    if not teams_data:
        return

    print(f"\n=== Fetching team stats for league {league_id} ===")

    all_stats = {}
    for item in teams_data.get("response", []):
        team = item.get("team", {})
        team_id = team.get("id")
        team_name = team.get("name")

        cache_path = ENRICH_DIR / f"teamstats_{league_id}_{team_id}.json"
        if cache_path.exists():
            print(f"    {team_name}: cached")
            with open(cache_path) as f:
                all_stats[team_name] = json.load(f)
            continue

        stats_data = api_football_get(
            "teams/statistics",
            {"league": league_id, "season": season, "team": team_id},
            label=team_name,
        )

        if stats_data:
            stats = stats_data.get("response", {})
            with open(cache_path, "w") as f:
                json.dump(stats, f, indent=2)
            all_stats[team_name] = stats

        time.sleep(0.5)  # Rate limit respect

    return all_stats


def parse_team_stats(league_id: int, team_name: str) -> dict:
    """Parse cached team stats into features."""
    # Try to find the team's stats file
    for f in ENRICH_DIR.glob(f"teamstats_{league_id}_*.json"):
        with open(f) as fh:
            stats = json.load(fh)
        if stats.get("team", {}).get("name") == team_name:
            fixtures = stats.get("fixtures", {})
            goals = stats.get("goals", {})

            played = (fixtures.get("played", {}).get("total", 0)) or 0
            wins = (fixtures.get("wins", {}).get("total", 0)) or 0
            draws = (fixtures.get("draws", {}).get("total", 0)) or 0
            losses = (fixtures.get("losses", {}).get("total", 0)) or 0
            goals_for = (goals.get("for", {}).get("total", {}).get("total", 0)) or 0
            goals_against = (goals.get("against", {}).get("total", {}).get("total", 0)) or 0
            clean_sheets = stats.get("clean_sheet", {}).get("total", 0) or 0

            return {
                "season_played": played,
                "season_win_pct": round(wins / max(played, 1), 3),
                "season_draw_pct": round(draws / max(played, 1), 3),
                "season_loss_pct": round(losses / max(played, 1), 3),
                "season_goals_per_game": round(goals_for / max(played, 1), 2),
                "season_conceded_per_game": round(goals_against / max(played, 1), 2),
                "season_goal_diff": goals_for - goals_against,
                "season_clean_sheet_pct": round(clean_sheets / max(played, 1), 3),
            }

    # Default if not found
    return {
        "season_played": 0,
        "season_win_pct": 0.33,
        "season_draw_pct": 0.33,
        "season_loss_pct": 0.33,
        "season_goals_per_game": 1.2,
        "season_conceded_per_game": 1.2,
        "season_goal_diff": 0,
        "season_clean_sheet_pct": 0.2,
    }


# ═══════════════════════════════════════════════════════════════════════════
# STANDINGS — League position, points, recent form
# ═══════════════════════════════════════════════════════════════════════════

def fetch_standings(league_id: int, season: int = 2025) -> None:
    """Fetch current league standings."""
    ENRICH_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = ENRICH_DIR / f"standings_{league_id}_{season}.json"

    print(f"\n=== Fetching standings for league {league_id} ===")

    data = api_football_get(
        "standings",
        {"league": league_id, "season": season},
        label="Standings",
    )

    if data:
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)

        standings = data.get("response", [])
        if standings:
            league_standings = standings[0].get("league", {}).get("standings", [[]])[0]
            for team in league_standings[:5]:
                print(f"    {team.get('rank')}. {team.get('team', {}).get('name')} "
                      f"- {team.get('points')} pts, Form: {team.get('form', '?')}")


def parse_standings(league_id: int, team_name: str, season: int = 2025) -> dict:
    """Parse standings into features for a specific team."""
    cache_path = ENRICH_DIR / f"standings_{league_id}_{season}.json"
    if not cache_path.exists():
        return {"league_rank": 10, "league_points": 0, "league_form_score": 0.5}

    with open(cache_path) as f:
        data = json.load(f)

    standings = data.get("response", [])
    if not standings:
        return {"league_rank": 10, "league_points": 0, "league_form_score": 0.5}

    league_standings = standings[0].get("league", {}).get("standings", [[]])[0]
    total_teams = len(league_standings)

    for team in league_standings:
        if team.get("team", {}).get("name") == team_name:
            rank = team.get("rank", total_teams)
            form = team.get("form", "")

            # Convert form string (e.g. "WWDLW") to a score 0-1
            form_map = {"W": 1.0, "D": 0.5, "L": 0.0}
            form_values = [form_map.get(c, 0.5) for c in form[-5:]]
            form_score = sum(form_values) / max(len(form_values), 1)

            return {
                "league_rank": rank,
                "league_rank_normalized": round(1 - (rank - 1) / max(total_teams - 1, 1), 3),
                "league_points": team.get("points", 0),
                "league_form_score": round(form_score, 3),
            }

    return {"league_rank": total_teams, "league_rank_normalized": 0, "league_points": 0, "league_form_score": 0.5}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Fetch all enrichment data for a league
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch enriched features from API-Football")
    parser.add_argument("--league", type=int, required=True, help="API-Football league ID")
    parser.add_argument("--season", type=int, default=2025, help="Season year")
    parser.add_argument("--all", action="store_true", help="Fetch everything")
    parser.add_argument("--injuries", action="store_true")
    parser.add_argument("--venues", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--standings", action="store_true")
    parser.add_argument("--h2h", action="store_true")
    args = parser.parse_args()

    ENRICH_DIR.mkdir(parents=True, exist_ok=True)

    if not API_FOOTBALL_KEY:
        print("ERROR: API_FOOTBALL_KEY not set.")
        print("1. Sign up at https://www.api-football.com/ (free)")
        print("2. Add to .env: API_FOOTBALL_KEY=your_key_here")
        sys.exit(1)

    if args.all or args.injuries:
        fetch_injuries(args.league, args.season)

    if args.all or args.venues:
        fetch_venues(args.league, args.season)

    if args.all or args.stats:
        fetch_team_stats(args.league, args.season)

    if args.all or args.standings:
        fetch_standings(args.league, args.season)

    if args.h2h:
        print("\nH2H requires team IDs. Use --stats first to get team IDs,")
        print("then call fetch_h2h(team1_id, team2_id) programmatically.")

    print(f"\nDone. Enriched data saved to {ENRICH_DIR}")
