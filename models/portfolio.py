"""
Portfolio model for managing a collection of assets.
Handles portfolio-level calculations and analytics.
"""

from typing import List, Dict, Optional
from collections import defaultdict
from .asset import Asset


class Portfolio:
    """
    Manages a collection of investment assets with analytics capabilities.

    Provides methods for calculating portfolio metrics, weights, and groupings.
    """

    def __init__(self, name: str = "My Portfolio"):
        """
        Initialize an empty portfolio.

        Args:
            name: Name of the portfolio
        """
        self.name = name
        self.assets: List[Asset] = []

    def add_asset(self, asset: Asset) -> None:
        """
        Add an asset to the portfolio.
        Same asset can be added multiple times
        Args:
            asset: Asset object to add
        """
        tickr = asset.ticker
        if not self.get_asset(tickr):
            print(f"Note that {tickr} is already within the portfolio")

        self.assets.append(asset)

    def remove_asset(self, ticker: str) -> bool:
        """
        Remove an asset from the portfolio by ticker.

        Args:
            ticker: Ticker symbol of asset to remove

        Returns:
            True if asset was removed, False if not found
        """
        ticker = ticker.upper()
        for i, asset in enumerate(self.assets):
            if asset.ticker == ticker:
                self.assets.pop(i)
                return True
        return False

    def get_asset(self, ticker: str) -> Optional[Asset]:
        """
        Get an asset by ticker symbol.

        Args:
            ticker: Ticker symbol to search for

        Returns:
            Asset if found, None otherwise
        """
        ticker = ticker.upper()
        for asset in self.assets:
            if asset.ticker == ticker:
                return asset
        return None

    @property
    def total_value(self) -> float:
        """Calculate total current portfolio value."""
        return sum(asset.current_value for asset in self.assets)

    @property
    def total_cost(self) -> float:
        """Calculate total original investment cost."""
        return sum(asset.transaction_value for asset in self.assets)

    @property
    def total_profit_loss(self) -> float:
        """Calculate total portfolio profit/loss."""
        return self.total_value - self.total_cost

    @property
    def total_profit_loss_pct(self) -> float:
        """Calculate total portfolio profit/loss percentage."""
        if self.total_cost == 0:
            return 0.0
        return (self.total_profit_loss / self.total_cost) * 100

    def get_weights(self) -> Dict[str, float]:
        """
        Calculate the weight of each asset in the portfolio.

        Returns:
            Dictionary mapping ticker to weight percentage
        """

        ticker_values = defaultdict(float)
        for asset in self.assets:
            ticker_values[asset.ticker] += asset.current_value

        if self.total_value == 0:
            return {}

        return {
            tickr: (value / self.total_value) * 100
            for tickr, value in ticker_values.items()
        }

    def get_sector_weights(self) -> Dict[str, float]:
        """
        Calculate portfolio weights grouped by sector.

        Returns:
            Dictionary mapping sector to weight percentage
        """
        sector_values = defaultdict(float)
        for asset in self.assets:
            sector_values[asset.sector] += asset.current_value

        if self.total_value == 0:
            return {}

        return {
            sector: (value / self.total_value) * 100
            for sector, value in sector_values.items()
        }

    def get_asset_class_weights(self) -> Dict[str, float]:
        """
        Calculate portfolio weights grouped by asset class.

        Returns:
            Dictionary mapping asset class to weight percentage
        """
        class_values = defaultdict(float)
        for asset in self.assets:
            class_values[asset.asset_class] += asset.current_value

        if self.total_value == 0:
            return {}

        return {
            asset_class: (value / self.total_value) * 100
            for asset_class, value in class_values.items()
        }

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get detailed summary by stock

        Returns:
            Dictionary with ticker metrics: (value, weight, return)
        """
        ticker_data = defaultdict(lambda: {"value": 0.0, "tv": 0.0, "pl": 0.0})

        for a in self.assets:
            ticker_data[a.ticker]["value"] += getattr(a, "current_value", 0.0)
            ticker_data[a.ticker]["tv"] += getattr(a, "transaction_value", 0.0)
            ticker_data[a.ticker]["pl"] += getattr(a, "profit_loss", 0.0)

        total_value = self.total_value
        out = defaultdict(lambda: {"value": 0.0, "weight": 0.0, "returns": 0.0})

        for tick, data in ticker_data.items():
            value = data["value"]
            tv = data["tv"]
            pl = data["pl"]

            out[tick]["value"] = value
            out[tick]["weight"] = (
                (value / total_value * 100) if total_value > 0 else 0.0
            )
            out[tick]["returns"] = (pl / tv * 100) if tv > 0 else 0

        return dict(ticker_data)

    def get_sector_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get detailed summary by sector.

        Returns:
            Dictionary with sector metrics (value, weight, count)
        """
        sector_data = defaultdict(lambda: {"value": 0.0, "count": 0})

        for asset in self.assets:
            sector_data[asset.sector]["value"] += asset.current_value
            sector_data[asset.sector]["count"] += 1

        total_value = self.total_value
        for sector in sector_data:
            sector_data[sector]["weight"] = (
                (sector_data[sector]["value"] / total_value * 100)
                if total_value > 0
                else 0
            )

        return dict(sector_data)

    def get_asset_class_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get detailed summary by asset class.

        Returns:
            Dictionary with asset class metrics (value, weight, count)
        """
        class_data = defaultdict(lambda: {"value": 0.0, "count": 0})

        for asset in self.assets:
            class_data[asset.asset_class]["value"] += asset.current_value
            class_data[asset.asset_class]["count"] += 1

        total_value = self.total_value
        for asset_class in class_data:
            class_data[asset_class]["weight"] = (
                (class_data[asset_class]["value"] / total_value * 100)
                if total_value > 0
                else 0
            )

        return dict(class_data)

    def __len__(self) -> int:
        """Return number of assets in portfolio."""
        return len(self.assets)

    def __repr__(self) -> str:
        """String representation of the portfolio."""
        return f"""Portfolio(name="{self.name}", assets={len(self.assets)}, \nvalue=€{self.total_value:,.2f})"""
