"""
Sports Betting Prediction Model.
═══════════════════════════════════════════════════════════════════════════
THIS IS THE FILE THE AUTORESEARCH AGENT MODIFIES.
═══════════════════════════════════════════════════════════════════════════

The agent can change anything in this file:
- Model architecture (XGBoost, LightGBM, CatBoost, ensemble, neural net...)
- Feature columns used
- Hyperparameters
- Betting strategy (thresholds, Kelly criterion, etc.)
- Parlay logic
- Anything else that might improve ROI

The evaluate.py backtester calls three functions from this file:
1. get_feature_columns() → list of feature column names to use
2. train_model(X, y, is_soccer) → trained model object
3. predict_probabilities(model, X, is_soccer) → dict of outcome probabilities
4. get_bet_recommendations(probs, odds, outcomes) → list of recommended bets
"""

import numpy as np
from xgboost import XGBClassifier


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════

def get_feature_columns() -> list[str]:
    """Return the list of feature columns to use for prediction."""
    return [
        # Implied probabilities from bookmaker odds
        "implied_prob_home",
        "implied_prob_draw",
        "implied_prob_away",
        # Recent form
        "home_form_5",
        "away_form_5",
        "home_form_10",
        "away_form_10",
        "form_diff_5",
        "form_diff_10",
        # Head-to-head
        "h2h_home_ratio",
        "h2h_away_ratio",
        # Scoring
        "home_avg_scored",
        "home_avg_conceded",
        "away_avg_scored",
        "away_avg_conceded",
        "home_goal_diff",
        "away_goal_diff",
        "scoring_diff",
        # Other
        "rest_advantage",
        "odds_ratio",
    ]


# ═══════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_model(X: np.ndarray, y: np.ndarray, is_soccer: bool = False) -> XGBClassifier:
    """Train a prediction model on historical data."""
    n_classes = 3 if is_soccer else 2
    objective = "multi:softprob" if is_soccer else "binary:logistic"

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective=objective,
        num_class=n_classes if is_soccer else None,
        eval_metric="mlogloss" if is_soccer else "logloss",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )

    model.fit(X, y)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def predict_probabilities(
    model: XGBClassifier, X: np.ndarray, is_soccer: bool = False
) -> dict[str, float]:
    """Predict outcome probabilities for a match."""
    probs = model.predict_proba(X)[0]

    if is_soccer:
        return {
            "home": float(probs[0]),
            "draw": float(probs[1]),
            "away": float(probs[2]),
        }
    else:
        return {
            "home": float(probs[0]),
            "away": float(probs[1]),
        }


# ═══════════════════════════════════════════════════════════════════════════
# BETTING STRATEGY
# ═══════════════════════════════════════════════════════════════════════════

# Minimum edge required to place a bet (model_prob - implied_prob)
MIN_EDGE = 0.05  # 5% edge minimum

def get_bet_recommendations(
    probs: dict[str, float],
    odds: dict[str, float],
    outcomes: list[str],
) -> list[dict]:
    """
    Decide which bets to place based on model probabilities vs bookmaker odds.

    A value bet exists when:
        model_probability > implied_probability + MIN_EDGE

    Returns a list of bet recommendations (can be empty if no value found).
    """
    recommendations = []

    for outcome in outcomes:
        if outcome not in probs or outcome not in odds:
            continue

        model_prob = probs[outcome]
        decimal_odds = odds[outcome]

        if decimal_odds <= 1.0:
            continue

        implied_prob = 1.0 / decimal_odds
        edge = model_prob - implied_prob

        if edge >= MIN_EDGE:
            recommendations.append({
                "outcome": outcome,
                "model_prob": round(model_prob, 4),
                "implied_prob": round(implied_prob, 4),
                "edge": round(edge, 4),
                "odds": decimal_odds,
            })

    return recommendations
