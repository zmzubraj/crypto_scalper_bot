# crypto_scalper_bot

Experimental Python scaffold for a Binance Spot crypto scalping bot.

The repository already includes:
- Async Binance execution plumbing in `execution/binance_connector.py`
- Config templates in `config/`
- Pinned Python dependencies in `requirements.txt`
- Project folders for strategies, models, risk controls, utilities, and backtesting

Several modules are still placeholders, so this repo is better understood as an early foundation than a finished trading system.

## Current layout

```text
config/
  config.yaml
  secrets_example.yaml
execution/
  binance_connector.py
strategies/
risk_management/
models/
utils/
backtesting/
main.py
```

## What is implemented today

- `execution/binance_connector.py` provides an async wrapper around `python-binance`
- Credentials are loaded from `config/secrets.yaml`
- Retry handling and helper methods exist for market data, balances, market orders, OCO orders, cancellations, and account-stream events
- `config/config.yaml` already defines trading pairs, risk settings, exit rules, model settings, sentiment thresholds, logging, and storage paths

## What is still scaffolded

At the moment, `main.py` and several modules under `strategies/`, `models/`, `risk_management/`, `utils/`, and `backtesting/` are placeholders. If you plan to extend this repo, the next logical step is wiring the connector into a small orchestrator and implementing one strategy end to end.

## Setup

1. Create a local virtual environment.
2. Install the pinned dependencies.
3. Copy `config/secrets_example.yaml` to `config/secrets.yaml`.
4. Add Binance Spot API credentials with trading enabled but withdrawals disabled.

Example:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy config\secrets_example.yaml config\secrets.yaml
```

## Configuration

- `config/config.yaml`: bot mode, pairs, risk rules, exit logic, model paths, logging, and storage paths
- `config/secrets.yaml`: private credentials and notifier secrets

Do not commit real API keys, exchange secrets, Telegram tokens, or email passwords.

## Minimal connector smoke test

Since `main.py` is currently empty, the easiest way to validate the implemented connector is a short script like this:

```python
import asyncio

from execution.binance_connector import BinanceConnector


async def main():
    async with BinanceConnector(testnet=False) as bx:
        price = await bx.get_symbol_price("BTCUSDT")
        print(f"BTCUSDT: {price}")


asyncio.run(main())
```

This requires a valid `config/secrets.yaml`.

## Dependency note

The repo pins `ta-lib`. On Windows, that package can require a prebuilt wheel or local native setup. If installation fails, the comment in `requirements.txt` points to the pure-Python fallback option.

## Safety notes

- Use Binance Spot keys with no withdrawal permission
- Start in paper or test-like conditions before touching live funds
- Keep secrets out of version control
- Treat this as experimental trading infrastructure, not production-ready execution software

## Suggested next steps

- Implement one complete strategy module
- Add a runnable `main.py` orchestration flow
- Add tests around order formatting and retry behavior
- Remove checked-in environment artifacts from version control in a separate cleanup PR
