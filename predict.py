"""
Sports Betting Prediction Model.
═══════════════════════════════════════════════════════════════════════════
THIS IS THE FILE THE AUTORESEARCH AGENT MODIFIES.
═══════════════════════════════════════════════════════════════════════════

Current strategy: XGBoost with reduced feature set (drop noisy scoring stats).
Focus on odds-derived + form features. MIN_EDGE=0.10 for high selectivity.
"""

import numpy as np
from xgboost import XGBClassifier


def get_feature_columns() -> list[str]:
    return [
        "implied_prob_home",
        "implied_prob_draw",
        "implied_prob_away",
        "home_form_5",
        "away_form_5",
        "form_diff_5",
        "form_diff_10",
        "h2h_home_ratio",
        "h2h_away_ratio",
        "odds_ratio",
        "rest_advantage",
        "scoring_diff",
    ]


def train_model(X: np.ndarray, y: np.ndarray, is_soccer: bool = False) -> XGBClassifier:
    n_classes = 3 if is_soccer else 2
    objective = "multi:softprob" if is_soccer else "binary:logistic"

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        min_child_weight=5,
        objective=objective,
        num_class=n_classes if is_soccer else None,
        eval_metric="mlogloss" if is_soccer else "logloss",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )

    model.fit(X, y)
    return model


def predict_probabilities(
    model: XGBClassifier, X: np.ndarray, is_soccer: bool = False
) -> dict[str, float]:
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


# Only bet when we have a strong edge
MIN_EDGE = 0.15
MAX_EDGE = 0.35  # Cap unrealistic edges (likely model error)

def get_bet_recommendations(
    probs: dict[str, float],
    odds: dict[str, float],
    outcomes: list[str],
) -> list[dict]:
    """Only bet on the single best value outcome. Cap unrealistic edges."""
    best = None
    best_edge = 0.0

    for outcome in outcomes:
        if outcome not in probs or outcome not in odds:
            continue

        model_prob = probs[outcome]
        decimal_odds = odds[outcome]

        if decimal_odds <= 1.0:
            continue

        implied_prob = 1.0 / decimal_odds
        edge = model_prob - implied_prob

        # Skip unrealistically large edges (likely model overconfidence)
        if edge >= MIN_EDGE and edge < MAX_EDGE and edge > best_edge:
            best_edge = edge
            best = {
                "outcome": outcome,
                "model_prob": round(model_prob, 4),
                "implied_prob": round(implied_prob, 4),
                "edge": round(edge, 4),
                "odds": decimal_odds,
            }

    return [best] if best else []
