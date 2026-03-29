"""
Backtesting engine for the Sports Betting ML Robot.
This file is NOT modified by the autoresearch agent.

Runs predict.py against historical test data and reports metrics.
The autoresearch ratchet uses ROI as the primary fitness signal.

Usage:
    python evaluate.py              # Run backtesting on all sports
    python evaluate.py --sport nba  # Run on a specific sport category
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from constants import (
    BET_SIZE,
    EXPERIMENTS_TSV,
    MIN_BETS,
    PROCESSED_DIR,
    RESULTS_DIR,
    SPORTS,
)

# Import the prediction module (this is what the agent modifies)
import predict


def run_backtest(sport_key: str, sport_cfg: dict) -> dict | None:
    """Run backtest for a single sport. Returns metrics dict or None."""
    test_path = PROCESSED_DIR / f"{sport_key}_test.parquet"
    train_path = PROCESSED_DIR / f"{sport_key}_train.parquet"

    if not test_path.exists() or not train_path.exists():
        return None

    test_df = pd.read_parquet(test_path)
    train_df = pd.read_parquet(train_path)

    # Filter to rows with known results
    test_df = test_df.dropna(subset=["result", "target"])
    if len(test_df) < 10:
        return None

    # ── Train the model ─────────────────────────────────────────────────
    is_soccer = sport_cfg["category"] == "soccer"
    feature_cols = predict.get_feature_columns()

    # Verify all feature columns exist
    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        print(f"  WARNING: Missing features {missing} for {sport_key}, skipping.")
        return None

    try:
        model = predict.train_model(
            train_df[feature_cols].values,
            train_df["target"].values.astype(int),
            is_soccer=is_soccer,
        )
    except Exception as e:
        print(f"  ERROR training {sport_key}: {e}")
        return None

    # ── Run predictions on test set ─────────────────────────────────────
    bets = []
    for _, row in test_df.iterrows():
        features = row[feature_cols].values.reshape(1, -1)

        try:
            probs = predict.predict_probabilities(model, features, is_soccer=is_soccer)
        except Exception as e:
            continue

        # Get odds from the row
        if is_soccer:
            odds = {
                "home": row["odds_home"],
                "draw": row["odds_draw"],
                "away": row["odds_away"],
            }
            outcomes = ["home", "draw", "away"]
        else:
            odds = {
                "home": row["odds_home"],
                "away": row["odds_away"],
            }
            outcomes = ["home", "away"]

        # Ask predict.py for bet recommendations
        recommendations = predict.get_bet_recommendations(probs, odds, outcomes)

        for rec in recommendations:
            outcome = rec["outcome"]
            actual_result = row["result"]
            won = outcome == actual_result
            payout = BET_SIZE * odds[outcome] if won else 0
            profit = payout - BET_SIZE

            bets.append({
                "date": row["date"],
                "sport": sport_key,
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "bet_on": outcome,
                "odds": odds[outcome],
                "model_prob": rec["model_prob"],
                "implied_prob": rec["implied_prob"],
                "edge": rec["edge"],
                "won": won,
                "profit": profit,
            })

    if not bets:
        return None

    bets_df = pd.DataFrame(bets)
    return _calculate_metrics(bets_df, sport_key)


def _calculate_metrics(bets_df: pd.DataFrame, sport_key: str) -> dict:
    """Calculate performance metrics from a bets DataFrame."""
    total_bets = len(bets_df)
    wins = bets_df["won"].sum()
    total_wagered = total_bets * BET_SIZE
    total_profit = bets_df["profit"].sum()
    roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0

    # Win rate
    win_rate = (wins / total_bets) * 100 if total_bets > 0 else 0

    # Max drawdown
    cumulative = bets_df["profit"].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()

    # Sharpe ratio (annualized, assuming ~1 bet/day average)
    daily_returns = bets_df["profit"] / BET_SIZE
    sharpe = 0.0
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)

    # Average edge on bets placed
    avg_edge = bets_df["edge"].mean() * 100

    return {
        "sport": sport_key,
        "total_bets": total_bets,
        "wins": int(wins),
        "win_rate": round(win_rate, 2),
        "total_wagered": round(total_wagered, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(roi, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe": round(sharpe, 2),
        "avg_edge": round(avg_edge, 2),
    }


def print_results(all_metrics: list[dict]) -> None:
    """Print results in a parseable format for the autoresearch ratchet."""
    # Per-sport results
    for m in all_metrics:
        print(f"\n--- {SPORTS.get(m['sport'], {}).get('name', m['sport'])} ---")
        print(f"  Bets: {m['total_bets']} | Wins: {m['wins']} | Win Rate: {m['win_rate']}%")
        print(f"  Profit: ${m['total_profit']} | ROI: {m['roi']}%")
        print(f"  Max Drawdown: ${m['max_drawdown']} | Sharpe: {m['sharpe']}")
        print(f"  Avg Edge: {m['avg_edge']}%")

    # Aggregate results (THIS IS WHAT THE RATCHET USES)
    total_bets = sum(m["total_bets"] for m in all_metrics)
    total_wagered = sum(m["total_wagered"] for m in all_metrics)
    total_profit = sum(m["total_profit"] for m in all_metrics)
    total_wins = sum(m["wins"] for m in all_metrics)
    agg_roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
    agg_win_rate = (total_wins / total_bets) * 100 if total_bets > 0 else 0

    print("\n" + "=" * 50)
    print("=== RESULTS ===")
    print(f"ROI: {agg_roi:.2f}%")
    print(f"WIN_RATE: {agg_win_rate:.2f}%")
    print(f"TOTAL_BETS: {total_bets}")
    print(f"PROFIT: ${total_profit:.2f}")
    print(f"MAX_DRAWDOWN: ${min(m['max_drawdown'] for m in all_metrics):.2f}")
    print(f"SHARPE: {np.mean([m['sharpe'] for m in all_metrics]):.2f}")
    print(f"SPORTS_COUNT: {len(all_metrics)}")
    print("=== END RESULTS ===")
    print("=" * 50)


def log_experiment(all_metrics: list[dict], description: str = "") -> None:
    """Append experiment results to the TSV log."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total_bets = sum(m["total_bets"] for m in all_metrics)
    total_wagered = sum(m["total_wagered"] for m in all_metrics)
    total_profit = sum(m["total_profit"] for m in all_metrics)
    agg_roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0

    row = {
        "timestamp": datetime.now().isoformat(),
        "roi": round(agg_roi, 2),
        "total_bets": total_bets,
        "profit": round(total_profit, 2),
        "sports": len(all_metrics),
        "description": description,
    }

    write_header = not EXPERIMENTS_TSV.exists()
    with open(EXPERIMENTS_TSV, "a") as f:
        if write_header:
            f.write("\t".join(row.keys()) + "\n")
        f.write("\t".join(str(v) for v in row.values()) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Sports Betting Backtester")
    parser.add_argument("--sport", type=str, default=None, help="Filter by sport category")
    parser.add_argument("--description", type=str, default="", help="Experiment description for log")
    args = parser.parse_args()

    start = time.time()
    all_metrics = []

    for sport_key, sport_cfg in SPORTS.items():
        if args.sport and sport_cfg["category"] != args.sport:
            continue

        metrics = run_backtest(sport_key, sport_cfg)
        if metrics:
            all_metrics.append(metrics)

    elapsed = time.time() - start

    if not all_metrics:
        print("=== RESULTS ===")
        print("ROI: 0.00%")
        print("TOTAL_BETS: 0")
        print("PROFIT: $0.00")
        print("ERROR: No data available. Run 'python prepare.py --source sample' first.")
        print("=== END RESULTS ===")
        sys.exit(1)

    print_results(all_metrics)
    print(f"\nBacktest completed in {elapsed:.1f}s")

    # Log to experiments TSV
    log_experiment(all_metrics, args.description)


if __name__ == "__main__":
    main()
