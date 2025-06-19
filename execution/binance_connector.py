"""
execution/binance_connector.py
──────────────────────────────
Asynchronous wrapper around Binance Spot API (python-binance) with:

1. Secure YAML-based credential loading (config/secrets.yaml).
2. Context-managed AsyncClient startup / graceful shutdown.
3. Resilient order & account helpers (market, OCO, cancel, balance).
4. Robust exponential-back-off retry decorator.
5. Optional user-data (account) WebSocket listener for fill events.

Compatible with the high-level bot orchestrator in main.py
───────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import datetime as dt
import functools
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import yaml
from binance import AsyncClient
from binance import BinanceSocketManager
from binance.enums import ORDER_TYPE_MARKET, TIME_IN_FORCE_GTC, SIDE_BUY, SIDE_SELL
from binance.exceptions import BinanceAPIException, BinanceOrderException

# -----------------------------------------------------------------------------
# Global logger
# -----------------------------------------------------------------------------
logger = logging.getLogger("binance_connector")
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# YAML credential loader
# -----------------------------------------------------------------------------
_SECRETS_PATH = Path(__file__).parent.parent / "config" / "secrets.yaml"

if not _SECRETS_PATH.exists():
    logger.critical("Secrets file not found: %s", _SECRETS_PATH.resolve())
    sys.exit(1)

with _SECRETS_PATH.open() as f:
    _SECRETS = yaml.safe_load(f)

API_KEY = _SECRETS["binance"]["api_key"]
API_SECRET = _SECRETS["binance"]["api_secret"]

# -----------------------------------------------------------------------------
# Retry decorator with exponential back-off
# -----------------------------------------------------------------------------
def retry_async(
    attempts: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    allowed_exceptions: tuple[type[Exception], ...] = (
        aiohttp.ClientError,
        BinanceAPIException,
        BinanceOrderException,
    ),
):
    """
    Decorator to retry async functions with exponential back-off.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except allowed_exceptions as e:
                    logger.warning(
                        "%s failed (attempt %s/%s): %s",
                        func.__name__,
                        attempt,
                        attempts,
                        e,
                    )
                    if attempt == attempts:
                        logger.error("%s: giving up after %s attempts", func.__name__, attempts)
                        raise
                    await asyncio.sleep(delay)
                    delay *= backoff_factor

        return wrapper

    return decorator


# -----------------------------------------------------------------------------
# BinanceConnector class
# -----------------------------------------------------------------------------
class BinanceConnector:
    """
    High-level async wrapper around python-binance AsyncClient.

    Usage:
        async with BinanceConnector() as bx:
            price = await bx.get_symbol_price("BTCUSDT")
            order_id = await bx.place_market_order("BTCUSDT", SIDE_BUY, quantity=0.001)
    """

    def __init__(
        self,
        api_key: str = API_KEY,
        api_secret: str = API_SECRET,
        testnet: bool = False,
        recv_window: int = 5000,
    ):
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._recv_window = recv_window
        self._client: Optional[AsyncClient] = None
        self._bsm: Optional[BinanceSocketManager] = None
        self._listen_key: Optional[str] = None
        self._user_stream_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ #
    # Context manager helpers
    # ------------------------------------------------------------------ #
    async def __aenter__(self):
        self._client = await AsyncClient.create(
            self._api_key, self._api_secret, testnet=self._testnet
        )
        if self._client is None:
            raise RuntimeError("Failed to initialize Binance AsyncClient")
        self._bsm = BinanceSocketManager(self._client)
        logger.info("Connected to Binance (testnet=%s)", self._testnet)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._user_stream_task:
            self._user_stream_task.cancel()
        # BinanceSocketManager does not require explicit close
        if self._client:
            await self._client.close_connection()
        logger.info("Binance connection closed.")

    # ------------------------------------------------------------------ #
    # Market data helpers
    # ------------------------------------------------------------------ #
    @retry_async()
    async def get_symbol_price(self, symbol: str) -> float:
        """Return the latest mark price for a symbol."""
        if not self._client:
            raise RuntimeError("Client not initialized")
        ticker = await self._client.get_symbol_ticker(symbol=symbol)
        return float(ticker["price"])

    @retry_async()
    async def get_available_balance(self, asset: str) -> float:
        """Return free balance of an asset."""
        if not self._client:
            raise RuntimeError("Client not initialized")
        account = await self._client.get_account()
        for bal in account["balances"]:
            if bal["asset"] == asset:
                return float(bal["free"])
        return 0.0

    # ------------------------------------------------------------------ #
    # Order helpers
    # ------------------------------------------------------------------ #
    @retry_async()
    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float | None = None,
        quote_quantity: float | None = None,
    ) -> Dict[str, Any]:
        """
        Place a market order.

        Pass either `quantity` (base asset) or `quote_quantity` (USDT value).
        """
        if not self._client:
            raise RuntimeError("Client not initialized")

        params = {
            "symbol": symbol,
            "side": side,
            "type": ORDER_TYPE_MARKET,
        }

        if quantity:
            params["quantity"] = self._format_quantity(symbol, quantity)
        elif quote_quantity:
            params["quoteOrderQty"] = self._format_quote_qty(symbol, quote_quantity)
        else:
            raise ValueError("Either quantity or quote_quantity must be provided")

        order = await self._client.create_order(**params)
        logger.info("Market order filled: %s", order["orderId"])
        return order

    @retry_async()
    async def place_oco_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        stop_price: float,
        stop_limit_price: float,
        stop_limit_time_in_force: str = TIME_IN_FORCE_GTC,
    ) -> Dict[str, Any]:
        """Place an OCO (One-Cancels-Other) order."""
        if not self._client:
            raise RuntimeError("Client not initialized")

        order = await self._client.create_oco_order(
            symbol=symbol,
            side=side,
            quantity=self._format_quantity(symbol, quantity),
            price=self._format_price(symbol, price),
            stopPrice=self._format_price(symbol, stop_price),
            stopLimitPrice=self._format_price(symbol, stop_limit_price),
            stopLimitTimeInForce=stop_limit_time_in_force,
        )
        logger.info("OCO order placed (%s): %s", side, order["orderListId"])
        return order

    @retry_async()
    async def cancel_open_orders(self, symbol: str) -> None:
        if not self._client:
            raise RuntimeError("Client not initialized")
        open_orders = await self._client.get_open_orders(symbol=symbol)
        for order in open_orders:
            await self._client.cancel_order(symbol=symbol, orderId=order["orderId"])
        logger.info("Cancelled all open orders for %s", symbol)

    # ------------------------------------------------------------------ #
    # User-data stream for fills / account updates
    # ------------------------------------------------------------------ #
    async def start_user_stream(self, callback) -> None:
        """
        Start a background task that forwards user-data events (fills, balance updates)
        to the given callback coroutine.
        """
        if not self._bsm:
            raise RuntimeError("Socket manager not initialized")
        if self._user_stream_task:
            logger.warning("User stream already running")
            return

        if not self._client:
            raise RuntimeError("Client not initialized")
        self._listen_key = await self._client.stream_get_listen_key()
        socket = self._bsm.user_socket()

        async def _run():
            async with socket as s:
                while True:
                    msg = await s.recv()
                    await callback(msg)

        self._user_stream_task = asyncio.create_task(_run())
        logger.info("User-data WebSocket started.")

    # ------------------------------------------------------------------ #
    # Utility – formatters using exchange info precision
    # ------------------------------------------------------------------ #
    def _format_quantity(self, symbol: str, qty: float) -> str:
        """
        Format qty to required stepSize for the symbol.
        """
        step = self._get_step_size(symbol)
        formatted = f"{math.floor(qty / step) * step:.{self._precision(step)}f}"
        return formatted

    def _format_price(self, symbol: str, price: float) -> str:
        tick = self._get_tick_size(symbol)
        formatted = f"{round(price / tick) * tick:.{self._precision(tick)}f}"
        return formatted

    def _format_quote_qty(self, symbol: str, quote_qty: float) -> str:
        step = self._get_step_size(symbol)
        price = asyncio.run(self.get_symbol_price(symbol))  # quick synchronous call
        base_qty = quote_qty / price
        return self._format_quantity(symbol, base_qty)

    # --- helpers ------------------------------------------------------- #
    _exchange_info_cache: Dict[str, Dict[str, Any]] = {}

    def _get_exchange_info(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self._exchange_info_cache:
            if not self._client:
                raise RuntimeError("Client not initialized")
            loop = asyncio.get_event_loop()
            info = loop.run_until_complete(self._client.get_symbol_info(symbol))
            if info is None:
                raise ValueError(f"Exchange info for symbol '{symbol}' not found")
            self._exchange_info_cache[symbol] = info
        return self._exchange_info_cache[symbol]

    def _get_step_size(self, symbol: str) -> float:
        filt = next(
            x for x in self._get_exchange_info(symbol)["filters"] if x["filterType"] == "LOT_SIZE"
        )
        return float(filt["stepSize"])

    def _get_tick_size(self, symbol: str) -> float:
        filt = next(
            x
            for x in self._get_exchange_info(symbol)["filters"]
            if x["filterType"] == "PRICE_FILTER"
        )
        return float(filt["tickSize"])

    @staticmethod
    def _precision(number: float) -> int:
        """Return decimal places for a float like 0.001."""
        return max(0, abs(int(round(math.log10(number)))))

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    async def ping(self) -> float:
        """Return API latency (ms)."""
        if not self._client:
            raise RuntimeError("Client not initialized")
        start = dt.datetime.utcnow()
        await self._client.ping()
        latency = (dt.datetime.utcnow() - start).total_seconds() * 1000
        logger.debug("Ping latency %.2f ms", latency)
        return latency


# -----------------------------------------------------------------------------
# Quick CLI test
# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    async def _quick_test():
        async with BinanceConnector(testnet=False) as bx:
            price = await bx.get_symbol_price("BTCUSDT")
            bal = await bx.get_available_balance("USDT")
            print("BTC price:", price, "USDT bal:", bal)

    asyncio.run(_quick_test())