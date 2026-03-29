# ZenntinelCMS — Sports Betting ML Robot

An autonomous self-improving sports betting prediction system, inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

An AI agent iteratively improves a prediction model (`predict.py`) against a backtesting engine (`evaluate.py`), keeping only changes that improve ROI. The "ratchet loop" runs autonomously — you sleep, the robot improves.

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  predict.py  │────▶│  evaluate.py  │────▶│  ROI better? │
│  (agent edits)│    │  (backtester) │    │             │
└─────────────┘     └──────────────┘     └──────┬──────┘
       ▲                                    │         │
       │                                   YES        NO
       │                                    │         │
       │                              git commit   git revert
       │                                    │         │
       └────────────────────────────────────┴─────────┘
```

## Sports Covered

- Soccer: Premier League, La Liga, Serie A, Bundesliga
- Basketball: NBA
- Football: NFL
- Baseball: MLB
- Hockey: NHL

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample data (no API key needed)
python prepare.py --source sample

# 3. Run backtesting to see baseline performance
python evaluate.py

# 4. Launch the autoresearch loop with Claude Code
claude --print program.md
# Then tell Claude: "Follow the instructions in program.md. Start the ratchet loop."
```

## Using Real Data (The Odds API)

```bash
# 1. Get a free API key at https://the-odds-api.com/ (500 req/month)
cp .env.example .env
# Edit .env and add your ODDS_API_KEY

# 2. Fetch real odds data
python prepare.py --source api

# 3. Run backtesting
python evaluate.py
```

## Project Structure

| File | Modified by Agent? | Purpose |
|---|---|---|
| `predict.py` | **YES** | Prediction model & betting strategy |
| `evaluate.py` | No | Backtesting engine, calculates ROI |
| `prepare.py` | No | Data fetching & feature engineering |
| `constants.py` | No | Configuration & constants |
| `program.md` | No | Instructions for the autoresearch agent |
| `results/experiments.tsv` | Auto-appended | Log of all experiments |

## The Ratchet

The core innovation from Karpathy's autoresearch: a simple hill-climbing loop.

1. Agent edits `predict.py` with an improvement idea
2. Runs `python evaluate.py > run.log 2>&1`
3. Checks `grep "ROI:" run.log`
4. If ROI improved → `git commit` (keep)
5. If ROI didn't improve → `git checkout -- predict.py` (revert)
6. Repeat forever

## Metrics

The backtester reports:
- **ROI%** — primary metric (profit / total wagered)
- **Win Rate%** — percentage of bets won
- **Total Bets** — minimum 100 required for valid experiment
- **Profit** — absolute dollar profit
- **Max Drawdown** — worst peak-to-trough loss
- **Sharpe Ratio** — risk-adjusted return

## License

MIT
