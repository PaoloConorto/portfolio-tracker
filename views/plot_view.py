"""
Chart view for creating visualizations.
Uses matplotlib for plotting price histories and portfolio charts.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime


class PlotView:
    """
    Handles creation of charts and visualizations.

    Uses matplotlib for generating graphs.
    """

    def __init__(self, style="seaborn-v0_8-darkgrid"):
        """Initialize chart view with optional style."""
        plt.style.use(style)
        self.colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]

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

    def plot_simulation_results(
        self,
        simulation_df: pd.DataFrame,
        confidence_levels: list = [5, 95],
        save_path: str = None,
    ) -> None:
        """
        Create comprehensive visualization for t-copula GBM simulation results.

        Args:
            simulation_df: DataFrame from TCopulaGBMSimulator.simulate() (n_paths, n_steps+1)
            title: Main title for the plot
            confidence_levels: Percentiles for confidence intervals [lower, upper]
            save_path: Optional path to save the figure
        """
        from matplotlib.ticker import FuncFormatter
        from scipy import stats

        # Convert to numpy for calculations
        data = simulation_df.values
        time_points = np.arange(data.shape[1])

        # Calculate statistics
        mean_path = np.mean(data, axis=0)
        median_path = np.median(data, axis=0)
        lower_ci = np.percentile(data, confidence_levels[0], axis=0)
        upper_ci = np.percentile(data, confidence_levels[1], axis=0)

        # Create figure with subplots - added more spacing
        fig = plt.figure(figsize=(16, 13))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.35, wspace=0.3)

        # Main paths plot
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_main_paths(
            ax1,
            data,
            time_points,
            mean_path,
            median_path,
            lower_ci,
            upper_ci,
            confidence_levels,
        )

        # Statistics subplots
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_distribution_at_horizon(ax2, data, simulation_df.columns[-1])

        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_path_percentiles(ax3, data, time_points)

        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_returns_distribution(ax4, data)

        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_performance_metrics(ax5, data, time_points)

        plt.subplots_adjust(top=0.96, bottom=0.05, left=0.08, right=0.95)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Chart saved to {save_path}")

        plt.show()

    def _plot_main_paths(
        self,
        ax,
        data,
        time_points,
        mean_path,
        median_path,
        lower_ci,
        upper_ci,
        confidence_levels,
    ):
        """Plot main simulation paths with statistics."""
        from matplotlib.ticker import FuncFormatter

        n_paths = data.shape[0]
        n_sample = min(100, n_paths)

        if n_paths > n_sample:
            rng = np.random.default_rng(42)
            sample_indices = rng.choice(n_paths, n_sample, replace=False)
        else:
            sample_indices = np.arange(n_paths)

        for i in sample_indices:
            ax.plot(time_points, data[i], alpha=0.1, color="#2E86AB", linewidth=0.5)

        ax.fill_between(
            time_points,
            lower_ci,
            upper_ci,
            alpha=0.3,
            color="#A23B72",
            label=f"{confidence_levels[0]}-{confidence_levels[1]}% CI",
        )

        # Plot mean and median
        ax.plot(
            time_points,
            mean_path,
            color="#F18F01",
            linewidth=2.5,
            label="Mean Path",
            linestyle="-",
        )
        ax.plot(
            time_points,
            median_path,
            color="#06A77D",
            linewidth=2.5,
            label="Median Path",
            linestyle="--",
        )

        # Formatting
        ax.set_xlabel("Time Steps", fontweight="bold", fontsize=11)
        ax.set_ylabel("Portfolio Value", fontweight="bold", fontsize=11)
        ax.set_title(
            "Simulation Paths with Confidence Intervals",
            fontweight="bold",
            fontsize=12,
            pad=15,
        )
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format y-axis
        euro_fmt = FuncFormatter(lambda x, p: f"€{x:,.0f}")
        ax.yaxis.set_major_formatter(euro_fmt)

    def _plot_distribution_at_horizon(self, ax, data, horizon_name):
        """Plot distribution of final values."""
        from matplotlib.ticker import FuncFormatter
        from scipy import stats

        final_values = data[:, -1]

        # Histogram with KDE
        ax.hist(
            final_values,
            bins=50,
            alpha=0.7,
            color="#2E86AB",
            density=True,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add KDE
        kde = stats.gaussian_kde(final_values)
        x_range = np.linspace(final_values.min(), final_values.max(), 200)
        ax.plot(x_range, kde(x_range), color="#A23B72", linewidth=2, label="KDE")

        # Add vertical lines for statistics
        mean_val = np.mean(final_values)
        median_val = np.median(final_values)

        ax.axvline(
            mean_val,
            color="#F18F01",
            linestyle="-",
            linewidth=2,
            label=f"Mean: €{mean_val:,.0f}",
        )
        ax.axvline(
            median_val,
            color="#06A77D",
            linestyle="--",
            linewidth=2,
            label=f"Median: €{median_val:,.0f}",
        )

        ax.set_xlabel("Final Portfolio Value", fontweight="bold", fontsize=10)
        ax.set_ylabel("Density", fontweight="bold", fontsize=10)
        ax.set_title(
            f"Distribution at {horizon_name}", fontweight="bold", fontsize=11, pad=12
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        euro_fmt = FuncFormatter(lambda x, p: f"€{x:,.0f}")
        ax.xaxis.set_major_formatter(euro_fmt)
        ax.tick_params(axis="x", rotation=15)

    def _plot_path_percentiles(self, ax, data, time_points):
        """Plot path percentiles over time."""
        from matplotlib.ticker import FuncFormatter

        # Calculate percentiles across time
        percentiles = np.percentile(data, [10, 25, 50, 75, 90], axis=0)

        # Plot percentiles
        labels = ["10th", "25th", "50th", "75th", "90th"]
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#06A77D", "#C73E1D"]

        for i, (p, label, color) in enumerate(zip(percentiles, labels, colors)):
            ax.plot(time_points, p, label=label, color=color, linewidth=2)

        ax.fill_between(
            time_points, percentiles[0], percentiles[-1], alpha=0.2, color="#2E86AB"
        )

        ax.set_xlabel("Time Steps", fontweight="bold", fontsize=10)
        ax.set_ylabel("Portfolio Value", fontweight="bold", fontsize=10)
        ax.set_title(
            "Path Percentiles Distribution", fontweight="bold", fontsize=11, pad=12
        )
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)

        euro_fmt = FuncFormatter(lambda x, p: f"€{x:,.0f}")
        ax.yaxis.set_major_formatter(euro_fmt)

    def _plot_returns_distribution(self, ax, data):
        """Plot distribution of log returns."""
        from scipy import stats

        # Calculate log returns
        log_returns = np.diff(np.log(data), axis=1)
        flat_returns = log_returns.flatten()

        # Remove outliers for better visualization
        q1, q3 = np.percentile(flat_returns, [1, 99])
        filtered_returns = flat_returns[(flat_returns >= q1) & (flat_returns <= q3)]

        # Plot histogram
        ax.hist(
            filtered_returns,
            bins=50,
            alpha=0.7,
            color="#2E86AB",
            density=True,
            edgecolor="black",
            linewidth=0.5,
        )

        mu, std = np.mean(filtered_returns), np.std(filtered_returns)
        x = np.linspace(filtered_returns.min(), filtered_returns.max(), 100)
        normal_pdf = stats.norm.pdf(x, mu, std)
        ax.plot(
            x,
            normal_pdf,
            color="#A23B72",
            linewidth=1,
            label=f"Normal fit\n(μ={mu:.4f}, σ={std:.4f})",
        )

        ax.set_xlabel("Log Returns", fontweight="bold", fontsize=10)
        ax.set_ylabel("Density", fontweight="bold", fontsize=10)
        # ax.set_title(
        #    "Distribution of Log Returns", fontweight="bold", fontsize=11, pad=12
        # )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_performance_metrics(self, ax, data, time_points):
        """Plot key performance metrics over time."""
        from matplotlib.ticker import FuncFormatter

        # Calculate metrics
        mean_values = np.mean(data, axis=0)
        volatility = np.std(data, axis=0)

        # Calculate rolling Sharpe ratio
        returns = np.diff(mean_values) / mean_values[:-1]
        rolling_vol = [
            np.std(returns[: i + 1]) if i > 0 else returns[0]
            for i in range(len(returns))
        ]
        sharpe_ratio = [
            returns[: i + 1].mean() / vol if vol > 0 else 0
            for i, vol in enumerate(rolling_vol)
        ]

        # Plot metrics
        color1 = "#2E86AB"
        color2 = "#A23B72"

        ax.plot(
            time_points[1:],
            sharpe_ratio,
            color=color1,
            linewidth=2,
            label="Sharpe Ratio",
        )
        ax.set_xlabel("Time Steps", fontweight="bold", fontsize=10)
        ax.set_ylabel("Sharpe Ratio", fontweight="bold", fontsize=10, color=color1)
        ax.tick_params(axis="y", labelcolor=color1)

        ax_twin = ax.twinx()
        ax_twin.plot(
            time_points,
            volatility,
            color=color2,
            linewidth=2,
            label="Volatility",
            linestyle="--",
        )
        ax_twin.set_ylabel("Volatility", fontweight="bold", fontsize=10, color=color2)
        ax_twin.tick_params(axis="y", labelcolor=color2)

        ax.set_title(
            "Risk-Adjusted Performance", fontweight="bold", fontsize=11, pad=12
        )

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

        ax.grid(True, alpha=0.3)

        # Format volatility axis
        euro_fmt = FuncFormatter(lambda x, p: f"€{x:,.0f}")
        ax_twin.yaxis.set_major_formatter(euro_fmt)
