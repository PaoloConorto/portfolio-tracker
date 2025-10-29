"""
Chart view for creating visualizations.
Uses matplotlib for plotting price histories and portfolio charts.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
from typing import List, Dict
from datetime import datetime


class PlotView:
    """
    Handles creation of charts and visualizations.

    Uses matplotlib for generating graphs.
    """

    def __init__(self):
        """Initialize chart view."""
        plt.style.use("seaborn-v0_8-darkgrid")

    def plot_price_history(
        self, ticker: str, df: pd.DataFrame, save_path: str = None
    ) -> None:
        """
        Plot historical price data for a single ticker.

        Args:
            ticker: Stock ticker symbol
            df: DataFrame with historical price data
            save_path: Optional path to save the figure
        """
        if df.empty:
            print(f"No data to plot for {ticker}")
            return

        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

        # Price plot
        euro_fmt = FuncFormatter(lambda x, p: f"€{x:,.0f}")
        ax1.plot(
            df.index, df["Close"], label="Close Price", linewidth=2, color="#2E86AB"
        )
        ax1.fill_between(df.index, df["Low"], df["High"], alpha=0.2, color="#2E86AB")
        ax1.set_title(f"{ticker} Price History", fontsize=16, fontweight="bold")
        ax1.yaxis.set_major_formatter(euro_fmt)
        ax1.set_ylabel("Price", fontsize=12)
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.1)

        # Volume plot
        colors = [
            "#06A77D" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#D00000"
            for i in range(len(df))
        ]
        ax2.bar(df.index, df["Volume"], color=colors, alpha=0.6)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("Volume", fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")
            print(f"Chart saved to {save_path}")

        plt.show()

    def plot_multiple_tickers(
        self,
        data: Dict[str, pd.DataFrame],
        normalize: bool = True,
        save_path: str = None,
    ) -> None:
        """
        Plot multiple tickers on the same chart.

        Args:
            data: Dictionary mapping ticker to DataFrame
            normalize: If True, normalize prices to percentage change
            save_path: Optional path to save the figure
        """
        if not data:
            print("No data to plot")
            return

        _, ax = plt.subplots(figsize=(14, 8))

        colors = ["#2E86AB", "#A23B72", "#F18F01", "#06A77D", "#C73E1D"]

        for i, (ticker, df) in enumerate(data.items()):
            if df.empty:
                continue

            if normalize:
                # Normalize to percentage change from first value
                normalized = (df["Close"] / df["Close"].iloc[0] - 1) * 100
                ax.plot(
                    df.index,
                    normalized,
                    label=ticker,
                    linewidth=2,
                    color=colors[i % len(colors)],
                )
            else:
                ax.plot(
                    df.index,
                    df["Close"],
                    label=ticker,
                    linewidth=2,
                    color=colors[i % len(colors)],
                )

        euro_fmt = FuncFormatter(lambda x, p: f"€{x:,.0f}")
        pct_fmt = FuncFormatter(lambda x, p: f"{x:.1f}%")
        if normalize:
            ax.yaxis.set_major_formatter(pct_fmt)
        else:
            ax.yaxis.set_major_formatter(euro_fmt)
        ylabel = "Change" if normalize else "Price"
        title = "Normalized Price Comparison" if normalize else "Price Comparison"

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.1)

        if normalize:
            ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")
            print(f"Chart saved to {save_path}")

        plt.show()

    def plot_portfolio_allocation(
        self,
        weights: Dict[str, float],
        title: str = "Portfolio Allocation",
        save_path: str = None,
    ) -> None:
        """
        Create a pie chart of portfolio allocation.

        Args:
            weights: Dictionary mapping name to weight percentage
            title: Chart title
            save_path: Optional path to save the figure
        """
        if not weights:
            print("No weights to plot")
            return

        _, ax = plt.subplots(figsize=(10, 8))

        # Sort by weight
        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_items]
        sizes = [item[1] for item in sorted_items]

        # Color scheme
        colors = plt.cm.Set3(range(len(labels)))

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"fontsize": 10},
        )

        # Enhance text
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")
            print(f"Chart saved to {save_path}")

        plt.show()

    def plot_sector_comparison(
        self, sector_data: Dict[str, Dict[str, float]], save_path: str = None
    ) -> None:
        """
        Create a bar chart comparing sectors.

        Args:
            sector_data: Dictionary with sector metrics
            save_path: Optional path to save the figure
        """
        if not sector_data:
            print("No sector data to plot")
            return

        sectors = list(sector_data.keys())
        values = [data["value"] for data in sector_data.values()]
        weights = [data["weight"] for data in sector_data.values()]

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        euro_fmt = FuncFormatter(lambda x, p: f"€{x:,.0f}")
        pct_fmt = FuncFormatter(lambda x, p: f"{x:.1f}%")

        # Value bar chart
        colors_value = plt.cm.viridis(range(len(sectors)))
        bars1 = ax1.bar(
            sectors,
            values,
            color=colors_value,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
        )
        ax1.set_title("Sector Values", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Value", fontsize=12)
        ax1.tick_params(axis="x", rotation=45)
        ax1.yaxis.set_major_formatter(euro_fmt)
        ax1.margins(y=0.1)
        # Added labels
        ax1.bar_label(
            bars1, labels=[f"€{v:,.0f}" for v in values], padding=3, fontsize=9
        )

        # Weight bar chart
        colors_weight = plt.cm.plasma(range(len(sectors)))
        bars2 = ax2.bar(
            sectors,
            weights,
            color=colors_weight,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
        )
        ax2.set_title("Sector Weights", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Weight", fontsize=12)
        ax2.tick_params(axis="x", rotation=45)
        ax2.yaxis.set_major_formatter(pct_fmt)
        ax2.margins(y=0.1)
        # Added Labels
        ax2.bar_label(
            bars2, labels=[f"{w:.1f}%" for w in weights], padding=3, fontsize=9
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")
            print(f"Chart saved to {save_path}")

        plt.show()
