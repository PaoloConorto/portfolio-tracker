"""
Price service for fetching market data from external APIs.
Uses yfinance for free stock market data.
"""

import yfinance as yf
import pandas as pd
from typing import Optional, List
from datetime import datetime, timedelta


class PriceService:
    """
    Service for fetching stock price data.

    Uses Yahoo Finance API (via yfinance) for market data.
    """

    def __init__(self):
        """Initialize price service."""
        self.cache = {}

    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Fetch the current market price for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Current price or None if not found
        """
        try:
            ticker = ticker.upper()
            stock = yf.Ticker(ticker)

            # Try multiple price fields as fallback
            info = stock.info
            price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            )

            return float(price) if price else None

        except Exception as e:
            print(f"Error fetching price for {ticker}: {e}")
            return None

    def get_current_prices(self, tickers: List[str]) -> dict:
        """
        Fetch current prices for multiple tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker to price
        """
        prices = {}
        for ticker in tickers:
            price = self.get_current_price(ticker)
            if price:
                prices[ticker.upper()] = price
        return prices

    def get_historical_prices(
        self,
        ticker: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date (defaults to today)
            interval: Data interval ("1d", "1wk", "1mo")

        Returns:
            DataFrame with OHLCV data
        """
        try:
            if end_date is None:
                end_date = datetime.now()

            ticker = ticker.upper()
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                print(f"No data found for {ticker}")
                return pd.DataFrame()

            return df

        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {e}")
            return pd.DataFrame()

    def get_ticker_info(self, ticker: str) -> dict:
        """
        Get detailed information about a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with ticker information
        """
        try:
            ticker = ticker.upper()
            stock = yf.Ticker(ticker)
            info = stock.info

            return {
                "name": info.get("longName", ticker),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "current_price": self.get_current_price(ticker),
                "market_cap": info.get("marketCap"),
                "currency": info.get("currency", "USD"),
            }

        except Exception as e:
            print(f"Error fetching info for {ticker}: {e}")
            return {"name": ticker, "sector": "Unknown", "industry": "Unknown"}

    def validate_ticker(self, ticker: str) -> bool:
        """
        Check if a ticker is valid.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if ticker exists, False otherwise
        """
        price = self.get_current_price(ticker)
        return price is not None

    def get_price_change(self, ticker: str, days: int = 1) -> Optional[float]:
        """
        Get price change over a period.

        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back

        Returns:
            Percentage change or None
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 5)

            df = self.get_historical_prices(ticker, start_date, end_date)

            if len(df) < 2:
                return None

            old_price = df["Close"].iloc[0]
            new_price = df["Close"].iloc[-1]

            return ((new_price - old_price) / old_price) * 100

        except Exception as e:
            print(f"Error calculating price change for {ticker}: {e}")
            return None
