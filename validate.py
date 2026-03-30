"""
Statistical validation for the Sports Betting ML Robot.
This file is NOT modified by the autoresearch agent.

Runs thousands of simulated scenarios to validate that the model's ROI
is real and not just an anomaly/overfitting artifact.

Methods:
1. Walk-Forward Validation — train on window, test on next window, slide forward
2. Bootstrap Sampling — resample bets 10,000 times, calculate ROI distribution
3. Random Shuffle Test — shuffle results randomly to see if ROI survives
4. Multi-Period Test — test on different time periods independently

Usage:
    python validate.py                    # Run all validations
    python validate.py --method bootstrap # Run only bootstrap (fast)
    python validate.py --method walkforward
    python validate.py --method shuffle
    python validate.py --method multiperiod
"""

import argparse
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

from constants import BET_SIZE, PROCESSED_DIR, SPORTS
import predict


def load_all_data() -> dict[str, pd.DataFrame]:
    """Load all sport data (train + test combined)."""
    all_data = {}
    for sport_key, sport_cfg in SPORTS.items():
        train_path = PROCESSED_DIR / f"{sport_key}_train.parquet"
        test_path = PROCESSED_DIR / f"{sport_key}_test.parquet"
        frames = []
        if train_path.exists():
            frames.append(pd.read_parquet(train_path))
        if test_path.exists():
            frames.append(pd.read_parquet(test_path))
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["result", "target"])
            df = df.sort_values("date").reset_index(drop=True)
            if len(df) >= 50:
                all_data[sport_key] = df
    return all_data


def simulate_bets(test_df: pd.DataFrame, model, sport_cfg: dict) -> list[dict]:
    """Run model predictions on test data, return list of bets."""
    is_soccer = sport_cfg["category"] == "soccer"
    feature_cols = predict.get_feature_columns()
    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        return []

    bets = []
    for _, row in test_df.iterrows():
        features = row[feature_cols].values.reshape(1, -1)
        try:
            probs = predict.predict_probabilities(model, features, is_soccer=is_soccer)
        except Exception:
            continue

        if is_soccer:
            odds = {"home": row["odds_home"], "draw": row["odds_draw"], "away": row["odds_away"]}
            outcomes = ["home", "draw", "away"]
        else:
            odds = {"home": row["odds_home"], "away": row["odds_away"]}
            outcomes = ["home", "away"]

        recs = predict.get_bet_recommendations(probs, odds, outcomes)
        for rec in recs:
            won = rec["outcome"] == row["result"]
            profit = BET_SIZE * odds[rec["outcome"]] - BET_SIZE if won else -BET_SIZE
            bets.append({"profit": profit, "won": won, "odds": odds[rec["outcome"]]})

    return bets


def calc_roi(bets: list[dict]) -> float:
    if not bets:
        return 0.0
    total_profit = sum(b["profit"] for b in bets)
    total_wagered = len(bets) * BET_SIZE
    return (total_profit / total_wagered) * 100 if total_wagered > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 1. WALK-FORWARD VALIDATION
#    Train on 3 months, test on next month. Slide forward. Repeat.
#    If the model only works on one period, this exposes it.
# ═══════════════════════════════════════════════════════════════════════════

def walk_forward_validation(all_data: dict, train_days: int = 90, test_days: int = 30) -> None:
    """Slide a training window forward through time. Test on the next window."""
    print("=" * 60)
    print("WALK-FORWARD VALIDATION")
    print(f"Train window: {train_days} days, Test window: {test_days} days")
    print("=" * 60)

    all_window_rois = []

    for sport_key, df in all_data.items():
        sport_cfg = SPORTS[sport_key]
        is_soccer = sport_cfg["category"] == "soccer"
        feature_cols = predict.get_feature_columns()

        if any(c not in df.columns for c in feature_cols):
            continue

        min_date = df["date"].min()
        max_date = df["date"].max()
        total_days = (max_date - min_date).days

        window_rois = []
        current_start = min_date

        while current_start + timedelta(days=train_days + test_days) <= max_date:
            train_end = current_start + timedelta(days=train_days)
            test_end = train_end + timedelta(days=test_days)

            train = df[(df["date"] >= current_start) & (df["date"] < train_end)]
            test = df[(df["date"] >= train_end) & (df["date"] < test_end)]

            if len(train) < 20 or len(test) < 5:
                current_start += timedelta(days=test_days)
                continue

            try:
                model = predict.train_model(
                    train[feature_cols].values,
                    train["target"].values.astype(int),
                    is_soccer=is_soccer,
                )
                bets = simulate_bets(test, model, sport_cfg)
                if bets:
                    roi = calc_roi(bets)
                    window_rois.append(roi)
                    all_window_rois.append(roi)
            except Exception:
                pass

            current_start += timedelta(days=test_days)

        if window_rois:
            avg_roi = np.mean(window_rois)
            win_pct = sum(1 for r in window_rois if r > 0) / len(window_rois) * 100
            print(f"\n  {sport_cfg['name']}:")
            print(f"    Windows tested: {len(window_rois)}")
            print(f"    ROI per window: {[f'{r:.1f}%' for r in window_rois]}")
            print(f"    Average ROI: {avg_roi:.2f}%")
            print(f"    Profitable windows: {win_pct:.0f}%")

    if all_window_rois:
        print(f"\n  === OVERALL ===")
        print(f"  Total windows: {len(all_window_rois)}")
        print(f"  Average ROI across all windows: {np.mean(all_window_rois):.2f}%")
        print(f"  Std dev: {np.std(all_window_rois):.2f}%")
        print(f"  Profitable windows: {sum(1 for r in all_window_rois if r > 0)}/{len(all_window_rois)} "
              f"({sum(1 for r in all_window_rois if r > 0) / len(all_window_rois) * 100:.0f}%)")
        print(f"  Worst window: {min(all_window_rois):.2f}%")
        print(f"  Best window: {max(all_window_rois):.2f}%")
    else:
        print("\n  No windows had enough data to test.")


# ═══════════════════════════════════════════════════════════════════════════
# 2. BOOTSTRAP — Resample bets 10,000 times. See distribution of ROI.
#    If 95% of samples are profitable, the edge is likely real.
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_validation(all_data: dict, n_simulations: int = 10000) -> None:
    """Resample bet outcomes to build a confidence interval for ROI."""
    print("=" * 60)
    print(f"BOOTSTRAP VALIDATION ({n_simulations:,} simulations)")
    print("=" * 60)

    # First, collect all real bets using the current model
    all_bets = []
    for sport_key, df in all_data.items():
        sport_cfg = SPORTS[sport_key]
        is_soccer = sport_cfg["category"] == "soccer"
        feature_cols = predict.get_feature_columns()

        if any(c not in df.columns for c in feature_cols):
            continue

        # Use last 30% as test
        split_idx = int(len(df) * 0.7)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]

        if len(train) < 20 or len(test) < 5:
            continue

        try:
            model = predict.train_model(
                train[feature_cols].values,
                train["target"].values.astype(int),
                is_soccer=is_soccer,
            )
            bets = simulate_bets(test, model, sport_cfg)
            all_bets.extend(bets)
        except Exception:
            continue

    if len(all_bets) < 10:
        print("  Not enough bets to bootstrap.")
        return

    profits = np.array([b["profit"] for b in all_bets])
    actual_roi = profits.sum() / (len(profits) * BET_SIZE) * 100

    print(f"\n  Actual bets: {len(all_bets)}")
    print(f"  Actual ROI: {actual_roi:.2f}%")
    print(f"\n  Running {n_simulations:,} bootstrap resamples...")

    rng = np.random.RandomState(42)
    bootstrap_rois = []
    for _ in range(n_simulations):
        sample = rng.choice(profits, size=len(profits), replace=True)
        roi = sample.sum() / (len(sample) * BET_SIZE) * 100
        bootstrap_rois.append(roi)

    bootstrap_rois = np.array(bootstrap_rois)

    ci_lower = np.percentile(bootstrap_rois, 2.5)
    ci_upper = np.percentile(bootstrap_rois, 97.5)
    pct_profitable = (bootstrap_rois > 0).sum() / len(bootstrap_rois) * 100

    print(f"\n  === BOOTSTRAP RESULTS ===")
    print(f"  Mean ROI: {bootstrap_rois.mean():.2f}%")
    print(f"  Median ROI: {np.median(bootstrap_rois):.2f}%")
    print(f"  95% Confidence Interval: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
    print(f"  Simulations with positive ROI: {pct_profitable:.1f}%")
    print(f"  Probability of profit: {pct_profitable:.1f}%")
    print()

    if ci_lower > 0:
        print(f"  CONCLUSION: STRONG SIGNAL. Even worst case (2.5th percentile) is profitable.")
    elif pct_profitable > 70:
        print(f"  CONCLUSION: MODERATE SIGNAL. Profitable in most scenarios but not guaranteed.")
    elif pct_profitable > 50:
        print(f"  CONCLUSION: WEAK SIGNAL. Slightly better than random, could be noise.")
    else:
        print(f"  CONCLUSION: NO SIGNAL. The ROI is likely random/noise.")


# ═══════════════════════════════════════════════════════════════════════════
# 3. RANDOM SHUFFLE TEST — Shuffle who wins. Does ROI survive?
#    If yes, the model is just lucky. If ROI drops to ~0, the edge is real.
# ═══════════════════════════════════════════════════════════════════════════

def shuffle_test(all_data: dict, n_shuffles: int = 1000) -> None:
    """Randomly shuffle match results. The model's ROI should collapse to ~0."""
    print("=" * 60)
    print(f"RANDOM SHUFFLE TEST ({n_shuffles:,} shuffles)")
    print("If the model is real, shuffled ROI should be near 0% or negative.")
    print("=" * 60)

    # Get real bets first
    all_bets_real = []
    all_bets_template = []  # bets without result (to re-simulate with shuffled results)

    for sport_key, df in all_data.items():
        sport_cfg = SPORTS[sport_key]
        is_soccer = sport_cfg["category"] == "soccer"
        feature_cols = predict.get_feature_columns()

        if any(c not in df.columns for c in feature_cols):
            continue

        split_idx = int(len(df) * 0.7)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]

        if len(train) < 20 or len(test) < 5:
            continue

        try:
            model = predict.train_model(
                train[feature_cols].values,
                train["target"].values.astype(int),
                is_soccer=is_soccer,
            )
        except Exception:
            continue

        for _, row in test.iterrows():
            features = row[feature_cols].values.reshape(1, -1)
            try:
                probs = predict.predict_probabilities(model, features, is_soccer=is_soccer)
            except Exception:
                continue

            if is_soccer:
                odds = {"home": row["odds_home"], "draw": row["odds_draw"], "away": row["odds_away"]}
                outcomes = ["home", "draw", "away"]
            else:
                odds = {"home": row["odds_home"], "away": row["odds_away"]}
                outcomes = ["home", "away"]

            recs = predict.get_bet_recommendations(probs, odds, outcomes)
            for rec in recs:
                actual_won = rec["outcome"] == row["result"]
                profit = BET_SIZE * odds[rec["outcome"]] - BET_SIZE if actual_won else -BET_SIZE
                all_bets_real.append(profit)
                all_bets_template.append({
                    "outcome": rec["outcome"],
                    "odds": odds[rec["outcome"]],
                    "possible_results": outcomes,
                    "implied_probs": [1 / odds.get(o, 99) for o in outcomes],
                })

    if len(all_bets_real) < 10:
        print("  Not enough bets.")
        return

    real_roi = sum(all_bets_real) / (len(all_bets_real) * BET_SIZE) * 100
    print(f"\n  Real ROI: {real_roi:.2f}% ({len(all_bets_real)} bets)")

    rng = np.random.RandomState(42)
    shuffled_rois = []

    for _ in range(n_shuffles):
        total_profit = 0
        for bet in all_bets_template:
            # Random result weighted by implied probabilities
            probs = np.array(bet["implied_probs"])
            probs = probs / probs.sum()
            random_result = rng.choice(bet["possible_results"], p=probs)
            won = bet["outcome"] == random_result
            total_profit += BET_SIZE * bet["odds"] - BET_SIZE if won else -BET_SIZE
        shuffled_roi = total_profit / (len(all_bets_template) * BET_SIZE) * 100
        shuffled_rois.append(shuffled_roi)

    shuffled_rois = np.array(shuffled_rois)
    pct_real_beats_shuffled = (real_roi > shuffled_rois).sum() / len(shuffled_rois) * 100

    print(f"  Shuffled ROI mean: {shuffled_rois.mean():.2f}%")
    print(f"  Shuffled ROI std: {shuffled_rois.std():.2f}%")
    print(f"  Real ROI beats {pct_real_beats_shuffled:.1f}% of shuffled scenarios")
    print()

    if pct_real_beats_shuffled > 95:
        print(f"  CONCLUSION: STRONG. Model beats 95%+ of random scenarios. Edge is real (p < 0.05).")
    elif pct_real_beats_shuffled > 80:
        print(f"  CONCLUSION: MODERATE. Model beats most random scenarios but not statistically conclusive.")
    else:
        print(f"  CONCLUSION: WEAK. Model doesn't clearly beat random. ROI may be luck.")


# ═══════════════════════════════════════════════════════════════════════════
# 4. MULTI-PERIOD — Test on completely separate time periods
# ═══════════════════════════════════════════════════════════════════════════

def multi_period_test(all_data: dict, n_periods: int = 5) -> None:
    """Split data into N non-overlapping periods. Test each independently."""
    print("=" * 60)
    print(f"MULTI-PERIOD TEST ({n_periods} independent periods)")
    print("Model must be profitable in MULTIPLE periods, not just one.")
    print("=" * 60)

    for sport_key, df in all_data.items():
        sport_cfg = SPORTS[sport_key]
        is_soccer = sport_cfg["category"] == "soccer"
        feature_cols = predict.get_feature_columns()

        if any(c not in df.columns for c in feature_cols):
            continue

        # Split into N equal periods
        period_size = len(df) // n_periods
        if period_size < 30:
            continue

        period_rois = []
        for i in range(n_periods):
            start = i * period_size
            end = start + period_size if i < n_periods - 1 else len(df)
            period_df = df.iloc[start:end]

            # Use first 70% for training, last 30% for testing within this period
            split = int(len(period_df) * 0.7)
            train = period_df.iloc[:split]
            test = period_df.iloc[split:]

            if len(train) < 15 or len(test) < 5:
                continue

            try:
                model = predict.train_model(
                    train[feature_cols].values,
                    train["target"].values.astype(int),
                    is_soccer=is_soccer,
                )
                bets = simulate_bets(test, model, sport_cfg)
                if bets:
                    roi = calc_roi(bets)
                    period_rois.append(roi)
            except Exception:
                continue

        if period_rois:
            profitable = sum(1 for r in period_rois if r > 0)
            print(f"\n  {sport_cfg['name']}:")
            print(f"    Periods tested: {len(period_rois)}")
            print(f"    ROI by period: {[f'{r:.1f}%' for r in period_rois]}")
            print(f"    Profitable: {profitable}/{len(period_rois)}")
            print(f"    Average ROI: {np.mean(period_rois):.2f}%")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical Validation")
    parser.add_argument("--method", choices=["all", "walkforward", "bootstrap", "shuffle", "multiperiod"],
                        default="all")
    parser.add_argument("--simulations", type=int, default=10000, help="Number of bootstrap/shuffle iterations")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  SPORTS BETTING MODEL — STATISTICAL VALIDATION")
    print("  Testing if the ROI is real or just luck")
    print("=" * 60)

    all_data = load_all_data()
    print(f"\nLoaded {len(all_data)} sports with sufficient data.\n")

    if args.method in ("all", "bootstrap"):
        bootstrap_validation(all_data, n_simulations=args.simulations)

    if args.method in ("all", "shuffle"):
        shuffle_test(all_data, n_shuffles=min(args.simulations, 1000))

    if args.method in ("all", "walkforward"):
        walk_forward_validation(all_data)

    if args.method in ("all", "multiperiod"):
        multi_period_test(all_data)

    print("\n" + "=" * 60)
    print("  VALIDATION COMPLETE")
    print("=" * 60)
