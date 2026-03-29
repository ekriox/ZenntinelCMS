# Sports Betting AutoResearch Program

> Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).
> Instead of optimizing val_bpb on LLM training, we optimize **ROI on sports betting backtesting**.

## Objective

You are an autonomous sports betting research agent. Your goal is to **maximize ROI (Return on Investment)** on a backtested portfolio of sports bets across multiple sports (soccer, NBA, NFL, MLB, NHL).

You do this by iteratively improving `predict.py` — the prediction model and betting strategy.

## The Ratchet Loop

Repeat forever:

### 1. Propose a change

Think of an improvement to `predict.py`. Ideas include (but are not limited to):

- **Model architecture**: Try LightGBM, CatBoost, ensemble of models, stacking
- **Hyperparameters**: Tune n_estimators, max_depth, learning_rate, regularization
- **Feature engineering**: Add new computed features, remove noisy ones, try interactions
- **Feature selection**: Use importance scores to prune features, try PCA or selection methods
- **Betting strategy**: Adjust MIN_EDGE threshold, implement Kelly criterion for sizing
- **Sport-specific models**: Different hyperparameters per sport category
- **Parlays**: Identify correlated legs, implement parlay logic in get_bet_recommendations()
- **Calibration**: Add probability calibration (Platt scaling, isotonic regression)
- **Class weights**: Handle imbalanced outcomes (draws are rare in some sports)
- **Rolling features**: If data supports it, create momentum/trend features

### 2. Implement the change

Edit `predict.py` with your proposed improvement. You may ONLY modify `predict.py` — do not touch `prepare.py`, `evaluate.py`, or `constants.py`.

### 3. Run the experiment

```bash
python evaluate.py > run.log 2>&1
```

**IMPORTANT**: Redirect all output to `run.log`. Do NOT use tee or let output flood your context window.

### 4. Check results

```bash
grep -A 10 "=== RESULTS ===" run.log
```

This will show you:
```
ROI: X.XX%
WIN_RATE: XX.XX%
TOTAL_BETS: NNN
PROFIT: $XXXX.XX
MAX_DRAWDOWN: $-XXX.XX
SHARPE: X.XX
SPORTS_COUNT: N
```

### 5. Keep or revert

**If ROI improved AND TOTAL_BETS >= 100:**
```bash
git add predict.py
git commit -m "Improvement: [brief description]. ROI: X.XX% -> Y.YY%"
```

**If ROI did NOT improve, OR TOTAL_BETS < 100:**
```bash
git checkout -- predict.py
```

The minimum bets requirement prevents the model from "cheating" by being extremely selective (only betting on sure things) which wouldn't be practical.

### 6. Go to step 1

## Rules

1. **ONLY modify `predict.py`**. Never modify `prepare.py`, `evaluate.py`, or `constants.py`.
2. **Never stop**. Do not ask the human if you should continue. The human might be asleep. You are autonomous.
3. **Be bold**. Try radical changes — different model architectures, completely new feature sets, novel betting strategies. The ratchet protects you: bad ideas get reverted.
4. **Be scientific**. Each experiment should test ONE hypothesis. Don't change 5 things at once — you won't know what worked.
5. **No look-ahead bias**. The model trains on historical data and is tested on future data. Never access test data during training.
6. **Track your thinking**. Add a brief comment at the top of `predict.py` noting the current strategy.
7. **If you run out of ideas, think harder**. Re-read the code for new angles, try combining previous near-misses, look at which sports are underperforming.

## Anti-Overfitting Guardrails

The backtester in `evaluate.py` uses a strict temporal split: the test set is the most recent 6 months of data, and the model only trains on older data. This prevents overfitting to the test period.

However, the autoresearch loop itself can overfit to the test set through repeated selection pressure. To mitigate this:

- Focus on changes that are **conceptually sound**, not just metric-chasing
- Prefer simpler models that are less likely to overfit
- Monitor TOTAL_BETS — a drastic decrease means the model is becoming too selective
- If ROI is suspiciously high (>20%), be skeptical — it's probably overfitting

## Getting Started

Before starting the loop, verify the pipeline works:

```bash
python prepare.py --source sample
python evaluate.py
```

This generates sample data and runs the baseline model. Record the baseline ROI, then begin iterating.

## Current Best ROI

Check the latest entry in `results/experiments.tsv` for the current best ROI.
