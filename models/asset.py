"""
Asset model for portfolio tracking.
Represents a single investment asset with all its properties and methods.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Asset:
    """
    Represents a single asset in the investment portfolio.

    Attributes:
        ticker: Stock ticker symbol
        sector: Industry sector
        asset_class: Type of asset
        quantity: Number of shares/units owned (can be float greater than 0 (no shorting))
        purchase_price: Price per share at purchase
        purchase_date: Date of purchase (timestamp)
        current_price: Current market price per share
    """

    ticker: str
    sector: str
    asset_class: str
    quantity: float
    purchase_price: float
    purchase_date: datetime = field(default_factory=datetime.now)
    current_price: Optional[float] = None

    def __post_init__(self):
        """Validate and normalize data after initialization."""
        self.ticker = self.ticker.upper()
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.purchase_price <= 0:
            raise ValueError("Purchase price must be positive")

    @property
    def transaction_value(self) -> float:
        """Calculate original purchase value."""
        return self.quantity * self.purchase_price

    @property
    def current_value(self) -> float:
        """Calculate current market value."""
        if self.current_price is None:
            return self.transaction_value
        return self.quantity * self.current_price

    @property
    def profit_loss(self) -> float:
        """Calculate absolute profit/loss."""
        return self.current_value - self.transaction_value

    @property
    def profit_loss_pct(self) -> float:
        """Calculate percentage profit/loss."""
        if self.transaction_value == 0:
            return 0.0
        return (self.profit_loss / self.transaction_value) * 100

    def update_price(self, new_price: float) -> None:
        """
        Update the current market price.

        Args:
            new_price: New price per share
        """
        if new_price <= 0:
            raise ValueError("Price must be positive")
        self.current_price = new_price

    def to_dict(self) -> dict:
        """Convert asset to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "sector": self.sector,
            "asset_class": self.asset_class,
            "quantity": self.quantity,
            "purchase_price": self.purchase_price,
            "purchase_date": self.purchase_date.isoformat(),
            "current_price": self.current_price,
            "transaction_value": self.transaction_value,
            "current_value": self.current_value,
            "profit_loss": self.profit_loss,
            "profit_loss_pct": self.profit_loss_pct,
        }

    def __repr__(self) -> str:
        """String representation of the asset."""
        return f"""Asset(ticker="{self.ticker}", quantity={self.quantity}, \nvalue=€{self.current_value:.2f})"""
