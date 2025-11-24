"""
Price service for fetching market data from external APIs.
Uses yfinance for free stock market data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
from scipy.stats import t as student_t


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

    def get_drift_volatility_and_garch(self, ticker: str):
        """Get annualized drift and volatility for a ticker

        Args:
            ticker: relevant ticker
        """
        try:
            end_date = datetime.now()
            start_date_5_years = end_date - timedelta(days=5 * 365)
            ticker = ticker.upper()
            stock = yf.Ticker(ticker)

            # Get historical data and convert to timezone-naive
            history_data = stock.history(period="max")
            min_date = history_data.index.min()

            # Convert to timezone-naive if needed
            if min_date.tzinfo is not None:
                min_date = min_date.tz_localize(None)

            # Make start_date_5_years timezone-naive
            start_date_5_years = start_date_5_years.replace(tzinfo=None)

            start_date = min(start_date_5_years, min_date)

            # Fetch data
            df = stock.history(start=start_date, end=end_date, interval="1mo")
            if df.empty:
                print(f"No data found for {ticker}")
                return None

            mclose = df["Close"].resample("ME").last()
            r_m = np.log(mclose / mclose.shift(1)).dropna()
            garch_par = self._fit_garch11_norm(r_m)

            mu_m = r_m.mean()
            sigma_m = r_m.std(ddof=1)
            return (12 * mu_m, np.sqrt(12) * sigma_m), garch_par

        except Exception as e:
            print(f"Error fetching drift and volatility data for {ticker}: {e}")
            return None

    def get_correlations(self, ticker_list: List[str]):
        try:
            end_date = datetime.now()
            start_date_5y = end_date - timedelta(days=5 * 365)

            monthly_prices = []
            names = []

            for ticker in map(str.upper, ticker_list):
                stock = yf.Ticker(ticker)

                # full history to find min available date
                history_data = stock.history(period="max")
                if history_data.empty:
                    continue
                idx = history_data.index
                if idx.tz is not None:
                    idx = idx.tz_localize(None)
                min_date = idx.min()

                # timezone-naive start cut
                start_date = min(
                    start_date_5y.replace(tzinfo=None), min_date.to_pydatetime()
                )

                # fetch monthly, then resample to calendar month-end to be safe
                df = stock.history(start=start_date, end=end_date, interval="1mo")
                if df.empty or "Close" not in df:
                    continue
                px = df.copy()
                if px.index.tz is not None:
                    px.index = px.index.tz_localize(None)
                mclose = px["Close"].resample("ME").last()

                monthly_prices.append(mclose.rename(ticker))
                names.append(ticker)

            if len(monthly_prices) < 2:
                raise ValueError("Not enough valid tickers for correlation.")

            # align on overlap and compute monthly log-returns
            px_panel = pd.concat(monthly_prices, axis=1)
            r_m = np.log(px_panel / px_panel.shift(1)).dropna(how="any")

            # Kendall_tau_correlation for t copula fitting correlation and return as ndarray
            kendall_tau_matrix = r_m.corr(method="kendall").values
            r_star = np.sin((np.pi / 2) * kendall_tau_matrix)

            def adjust_correlation(r_star, delta=1e-8):
                # Step 1: Spectral decomposition
                eigenvalues, eigenvectors = np.linalg.eigh(r_star)

                # Step 2: Replace negative eigenvalues
                eigenvalues[eigenvalues < delta] = delta

                # Step 3: Reconstruct matrix
                Q = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

                # Step 4: Convert to correlation matrix
                D = np.diag(1 / np.sqrt(np.diag(Q)))
                R = D @ Q @ D

                return R

            if not np.all(np.linalg.eigvals(r_star) > 0):
                print(r"Not postive definite $\tau$")
                corr = adjust_correlation(r_star)
            else:
                corr = r_star

            return corr

        except Exception as e:
            print(f"Error computing correlations: {e}")
            return None

    def get_t_marginal_parameters(self, ticker_list: List[str]):
        try:
            end_date = datetime.now()
            start_date_5y = end_date - timedelta(days=5 * 365)

            monthly_prices = []
            names = []

            for ticker in map(str.upper, ticker_list):
                stock = yf.Ticker(ticker)

                # full history to find min available date
                history_data = stock.history(period="max")
                if history_data.empty:
                    continue
                idx = history_data.index
                if idx.tz is not None:
                    idx = idx.tz_localize(None)
                min_date = idx.min()

                # timezone-naive start cut
                start_date = min(
                    start_date_5y.replace(tzinfo=None), min_date.to_pydatetime()
                )

                # fetch monthly, then resample to calendar month-end to be safe
                df = stock.history(start=start_date, end=end_date, interval="1mo")
                if df.empty or "Close" not in df:
                    continue
                px = df.copy()
                if px.index.tz is not None:
                    px.index = px.index.tz_localize(None)
                mclose = px["Close"].resample("ME").last()

                monthly_prices.append(mclose.rename(ticker))
                names.append(ticker)

            # align on overlap and compute monthly log-returns
            px_panel = pd.concat(monthly_prices, axis=1)
            r_m = np.log(px_panel / px_panel.shift(1)).dropna(how="any")
            mu_m = r_m.mean(axis=0)
            sigma_m = r_m.std(axis=0, ddof=1)
            e_m = (r_m - mu_m) / sigma_m
            df_marginals = np.empty(e_m.shape[1])

            for j, col in enumerate(e_m.columns):
                df_hat, _, __ = student_t.fit(
                    e_m[col].values, floc=0.0, fscale=1.0  # fix mean 0  # fix scale 1
                )
                df_marginals[j] = df_hat
            return df_marginals

        except Exception as e:
            print(f"Error computing degrees of freedoms: {e}")
            return None

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

    def _fit_garch11_norm(self, ret: pd.Series):
        """
        Fit GARCH(1,1) with normal innovations to monthly log returns.
        Returns ANNUALIZED parameters for consistency with GBM simulation.
        """
        from arch.univariate import GARCH, ConstantMean, Normal

        ret_clean = ret.dropna()

        if len(ret_clean) < 24:
            print("Insufficient data for GARCH estimation")
            return "NO GARCH"
        model = ConstantMean(ret_clean * 100)
        model.volatility = GARCH(p=1, q=1)
        model.distribution = Normal()

        try:
            result = model.fit(disp="off", show_warning=False)
            params = result.params

            omega = params["omega"] / 10000
            alpha = params["alpha[1]"]
            beta = params["beta[1]"]
            mu = params.get("mu", ret_clean.mean() * 100) / 100

            persistence = alpha + beta
            if persistence >= 0.999:
                scale_factor = 0.98 / persistence
                alpha *= scale_factor
                beta *= scale_factor
                persistence = alpha + beta

            h0_monthly = omega / (1 - persistence)

            omega_annual = omega * 12
            h0_annual = h0_monthly * 12
            mu_annual = mu * 12

            return {
                "omega": omega_annual,
                "alpha": alpha,
                "beta": beta,
                "h0": h0_annual,
                "mu": mu_annual,
                "persistence": persistence,
            }

        except Exception as e:
            print(f"GARCH fitting failed: {e}")
            var_monthly = ret_clean.var()
            return {
                "omega": var_monthly * 0.1 * 12,
                "alpha": 0.1,
                "beta": 0.85,
                "h0": var_monthly * 12,
                "mu": ret_clean.mean() * 12,
                "persistence": 0.95,
            }
