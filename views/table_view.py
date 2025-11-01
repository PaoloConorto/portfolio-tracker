"""
table view for displaying portfolio data in formatted table
Uses rich library
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from typing import List, Dict
from models.asset import Asset
import numpy as np
import pandas as pd


class TableView:
    """
    Handles rendering of tabular dat in CLI
    Uses rich
    """

    def __init__(self):
        self.console = Console()

    def display_portfolio(
        self, assets: List[Asset], title: str = "Portfolio Overview"
    ) -> None:
        """Displays portfolio assets in a formatted table (Note that the formatting is fully chats)
        Args:
            assets (List[Asset]): List of asset objects
            title (str, optional): Title. Defaults to "Portfolio Overview".
        """
        if not assets:
            self.console.print("[yellow] Portfolio is empty[/yellow]")
            return

        table = Table(title=title, show_header=True, header_style="bold cyan")

        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Sector", style="cyan")
        table.add_column("Class", style="green")
        table.add_column("Quantity", justify="right")
        table.add_column("Purchase Price", justify="right")
        table.add_column("Current Price", justify="right")
        table.add_column("Trans. Value", justify="right")
        table.add_column("Current Value", justify="right", style="bold")
        table.add_column("P/L", justify="right")
        table.add_column("P/L %", justify="right")

        for asset in assets:
            pl_color = "green" if asset.profit_loss >= 0 else "red"
            pl_symbol = "+" if asset.profit_loss >= 0 else ""

            current_price_str = (
                f"€{asset.current_price:.2f}"
                if asset.current_price is not None
                else "N/A"
            )

            table.add_row(
                asset.ticker,
                asset.sector,
                asset.asset_class,
                f"{asset.quantity:.2f}",
                f"€{asset.purchase_price:.2f}",
                current_price_str,
                f"€{asset.transaction_value:,.2f}",
                f"€{asset.current_value:,.2f}",
                f"[{pl_color}]{pl_symbol}€{asset.profit_loss:,.2f}[/{pl_color}]",
                f"[{pl_color}]{pl_symbol}{asset.profit_loss_pct:.2f}%[/{pl_color}]",
            )

        self.console.print(table)

    def display_weights(
        self, weights: Dict[str, float], title: str = "Portfolio Weights"
    ) -> None:
        """Displays weights in a formatted table (Note that chat did the formatting)

        Args:
            weights (Dict[str, float]): dict with the weights with their respective ticker
            title (str, optional): Title. Defaults to "Portfolio Weights".
        """
        if not weights:
            self.console.print("[yellow]No weights to display[/yellow]")
            return

        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Name", style="cyan")
        table.add_column("Weight", justify="right", style="green")
        table.add_column("Bar", style="blue")

        # Sort by weight descending
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        for name, weight in sorted_weights:
            # Create simple bar chart
            bar_length = int(weight / 2)  # Scale to fit terminal
            bar = "█" * bar_length

            table.add_row(name, f"{weight:.2f}%", bar)

        self.console.print(table)

    def display_summary(
        self, summary: Dict[str, Dict[str, float]], title: str = "Portfolio Summary"
    ) -> None:
        """Displays porfolio summary by cat (Implementation is also by LLM)

        Args:
            summary (Dict[str, Dict[str, float]]): the summary in dict form (check portofolio methods)
            title (str, optional): Title. Defaults to "Portfolio Summary".
        """
        if not summary:
            self.console.print("[yellow]No summary data[/yellow]")
            return

        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Category", style="cyan")
        table.add_column("Assets", justify="right")
        table.add_column("Value", justify="right", style="green")
        table.add_column("Weight", justify="right", style="yellow")

        for category, data in sorted(summary.items()):
            table.add_row(
                category,
                str(data.get("count", 0)),
                f"€{data.get("value", 0):,.2f}",
                f"{data.get("weight", 0):.2f}%",
            )

        self.console.print(table)

    def display_portfolio_stats(
        self,
        total_value: float,
        total_cost: float,
        total_pl: float,
        total_pl_pct: float,
    ) -> None:
        """Displays the porfolios overall statistics

        Args:
            total_value (float): Total value
            total_cost (float): total cost
            total_pl (float): pl
            total_pl_pct (float): percentage pl
        """
        pl_color = "green" if total_pl >= 0 else "red"
        pl_symbol = "+" if total_pl >= 0 else ""

        stats_text = Text()
        stats_text.append("Total Investment: ", style="bold")
        stats_text.append(f"€{total_cost:,.2f}\n", style="cyan")

        stats_text.append("Current Value: ", style="bold")
        stats_text.append(f"€{total_value:,.2f}\n", style="cyan")

        stats_text.append("Total P/L: ", style="bold")
        stats_text.append(
            f"{pl_symbol}€{total_pl:,.2f} ({pl_symbol}{total_pl_pct:,.2f}%)",
            style=pl_color,
        )

        panel = Panel(
            stats_text,
            title="[bold]Portfolio Statistics[/bold]",
            border_style="blue",
            padding=(1, 2),
        )

        self.console.print(panel)

    def display_simulation_risk_metrics(
        self,
        simulation_df: pd.DataFrame,
        initial_value: float,
        n_years: int,
        confidence_levels: list = [99.5, 99, 95, 90],
        risk_free_rate: float = 0.02,
    ) -> None:
        """
        Display comprehensive risk metrics from Monte Carlo simulation.
        Solvency II compliant analysis for insurance/pension portfolios.

        Args:
            simulation_df: DataFrame from TCopulaGBMSimulator (n_paths, n_steps+1)
            initial_value: Initial portfolio value (V0)
            n_years: Simulation horizon in years
            confidence_levels: Confidence levels for VaR/CVaR (default includes 99.5% for Solvency II)
            risk_free_rate: Annualized risk-free rate for Sharpe calculation
        """
        # Calculate key metrics
        final_values = simulation_df.iloc[:, -1].values
        all_values = simulation_df.values

        # Returns calculations
        total_returns = (final_values - initial_value) / initial_value

        # Periodic returns for volatility
        period_returns = np.diff(all_values, axis=1) / all_values[:, :-1]

        # Calculate annualized metrics
        mean_return_annualized = ((1 + total_returns.mean()) ** (1 / n_years) - 1) * 100
        median_return_annualized = (
            (1 + np.median(total_returns)) ** (1 / n_years) - 1
        ) * 100
        volatility_annualized = (
            period_returns.std() * np.sqrt(12) * 100
        )  # Monthly to annual

        # Sharpe Ratio
        excess_return = mean_return_annualized / 100 - risk_free_rate
        sharpe_ratio = (
            excess_return / (volatility_annualized / 100)
            if volatility_annualized > 0
            else 0
        )

        # Sortino Ratio (downside deviation)
        downside_returns = period_returns[period_returns < 0]
        downside_deviation = (
            downside_returns.std() * np.sqrt(12) * 100
            if len(downside_returns) > 0
            else 0
        )
        sortino_ratio = (
            excess_return / (downside_deviation / 100) if downside_deviation > 0 else 0
        )

        # Solvency II Specific: SCR calculation (Simplified market risk module)
        # SCR = Solvency Capital Requirement at 99.5% over 1 year
        var_99_5 = np.percentile(final_values, 0.5)  # 99.5% VaR
        scr_market = initial_value - var_99_5
        scr_ratio = (scr_market / initial_value) * 100

        # Create main panel layout
        self.console.print("\n")
        self.console.rule(
            "[bold cyan]Monte Carlo Simulation - Solvency II Risk Analysis[/bold cyan]",
            style="cyan",
        )
        self.console.print("\n")

        # SECTION 1: RETURN METRICS

        returns_table = Table(
            title="Return Metrics",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
        )
        returns_table.add_column("Metric", style="cyan", width=38)
        returns_table.add_column("Value", justify="right", style="bold white", width=20)
        returns_table.add_column("Description", style="dim", width=40)

        returns_table.add_row(
            "Expected Return (Annualized)",
            f"[green]{mean_return_annualized:+.2f}%[/green]",
            "Geometric mean return",
        )
        returns_table.add_row(
            "Median Return (Annualized)",
            f"[green]{median_return_annualized:+.2f}%[/green]",
            "50th percentile return",
        )
        returns_table.add_row(
            "Expected Final Value",
            f"€{final_values.mean():,.0f}",
            f"Mean value after {n_years} years",
        )
        returns_table.add_row(
            "Median Final Value",
            f"€{np.median(final_values):,.0f}",
            f"Median value after {n_years} years",
        )
        returns_table.add_row(
            "Standard Deviation (Final)",
            f"€{final_values.std():,.0f}",
            "Dispersion of outcomes",
        )

        # Return distribution quartiles
        q1 = np.percentile(final_values, 25)
        q3 = np.percentile(final_values, 75)
        iqr = q3 - q1
        returns_table.add_row(
            "Interquartile Range (IQR)", f"€{iqr:,.0f}", "Q3 - Q1 spread"
        )

        self.console.print(returns_table)
        self.console.print("\n")

        # SECTION 2: VOLATILITY & RISK-ADJUSTED RETURNS

        risk_table = Table(
            title="Risk-Adjusted Performance Metrics",
            show_header=True,
            header_style="bold yellow",
            border_style="yellow",
        )
        risk_table.add_column("Metric", style="yellow", width=38)
        risk_table.add_column("Value", justify="right", style="bold white", width=20)
        risk_table.add_column("Interpretation", style="dim", width=40)

        risk_table.add_row(
            "Volatility (Annualized)",
            f"[yellow]{volatility_annualized:.2f}%[/yellow]",
            "Annual standard deviation",
        )
        risk_table.add_row(
            "Downside Deviation",
            f"[red]{downside_deviation:.2f}%[/red]",
            "Volatility of negative returns",
        )
        risk_table.add_row(
            "Sharpe Ratio",
            f"[cyan]{sharpe_ratio:.3f}[/cyan]",
            ">1.0 good, >2.0 excellent",
        )
        risk_table.add_row(
            "Sortino Ratio",
            f"[cyan]{sortino_ratio:.3f}[/cyan]",
            "Downside risk-adjusted return",
        )

        # Coefficient of Variation
        cv = (final_values.std() / final_values.mean()) * 100
        risk_table.add_row(
            "Coefficient of Variation", f"{cv:.2f}%", "Risk per unit of return"
        )

        # Probability of loss
        prob_loss = (total_returns < 0).sum() / len(total_returns) * 100
        loss_color = (
            "red" if prob_loss > 25 else "yellow" if prob_loss > 10 else "green"
        )
        risk_table.add_row(
            "Probability of Loss",
            f"[{loss_color}]{prob_loss:.2f}%[/{loss_color}]",
            f"P(loss) after {n_years} years",
        )

        # Maximum Drawdown
        max_values = all_values.max(axis=1, keepdims=True)
        drawdowns = (max_values - all_values) / max_values
        max_drawdown = drawdowns.max() * 100
        risk_table.add_row(
            "Maximum Drawdown",
            f"[red]{max_drawdown:.2f}%[/red]",
            "Largest peak-to-trough decline",
        )

        # Average Drawdown
        avg_drawdown = drawdowns.mean() * 100
        risk_table.add_row(
            "Average Drawdown", f"{avg_drawdown:.2f}%", "Mean drawdown across paths"
        )

        self.console.print(risk_table)
        self.console.print("\n")

        # SECTION 3: SOLVENCY II CAPITAL REQUIREMENTS

        solvency_table = Table(
            title="Solvency II Capital Requirements (Market Risk Module)",
            show_header=True,
            header_style="bold magenta",
            border_style="magenta",
        )
        solvency_table.add_column("Metric", style="magenta", width=38)
        solvency_table.add_column(
            "Value", justify="right", style="bold white", width=20
        )
        solvency_table.add_column("Regulatory Context", style="dim", width=40)

        # SCR (Solvency Capital Requirement)
        scr_color = "green" if scr_ratio < 30 else "yellow" if scr_ratio < 50 else "red"
        solvency_table.add_row(
            "SCR - 99.5% VaR (1 year)",
            f"[{scr_color}]€{scr_market:,.0f}[/{scr_color}]",
            "Solvency II capital requirement",
        )
        solvency_table.add_row(
            "SCR Ratio",
            f"[{scr_color}]{scr_ratio:.2f}%[/{scr_color}]",
            "SCR as % of initial capital",
        )

        # Own Funds requirement
        mcr_99 = initial_value - np.percentile(final_values, 1)  # 99% for MCR proxy
        solvency_table.add_row(
            "MCR Proxy (99% VaR)",
            f"€{mcr_99:,.0f}",
            "Minimum Capital Requirement proxy",
        )

        # Solvency Ratio (if capital equals initial value)
        solvency_ratio = (initial_value / scr_market) * 100 if scr_market > 0 else 999
        ratio_color = (
            "green"
            if solvency_ratio > 150
            else "yellow" if solvency_ratio > 100 else "red"
        )
        solvency_table.add_row(
            "Solvency Ratio",
            f"[{ratio_color}]{min(solvency_ratio, 999):.0f}%[/{ratio_color}]",
            "Available capital / SCR (>100% required)",
        )

        # Stress scenario: 40% equity shock (Solvency II standard)
        stress_40_value = initial_value * 0.6  # 40% loss
        prob_exceed_stress = (
            (final_values < stress_40_value).sum() / len(final_values) * 100
        )
        stress_color = (
            "green"
            if prob_exceed_stress < 1
            else "yellow" if prob_exceed_stress < 5 else "red"
        )
        solvency_table.add_row(
            "P(40% Loss) - Equity Shock",
            f"[{stress_color}]{prob_exceed_stress:.3f}%[/{stress_color}]",
            "Solvency II equity stress test",
        )

        self.console.print(solvency_table)

        # SECTION 4: VALUE-AT-RISK (VaR) & EXPECTED SHORTFALL (ES/CVaR)

        var_table = Table(
            title="Value-at-Risk (VaR) & Expected Shortfall Analysis",
            show_header=True,
            header_style="bold red",
            border_style="red",
        )
        var_table.add_column("Confidence", style="red", width=15)
        var_table.add_column("VaR (€)", justify="right", style="bold white", width=18)
        var_table.add_column("VaR (%)", justify="right", style="red", width=12)
        var_table.add_column(
            "ES/CVaR (€)", justify="right", style="bold white", width=18
        )
        var_table.add_column("ES (%)", justify="right", style="red", width=12)
        var_table.add_column("Regulatory Use", style="dim", width=25)

        for cl in confidence_levels:
            alpha = (100 - cl) / 100
            var_value = np.percentile(final_values, alpha * 100)
            var_pct = (var_value - initial_value) / initial_value * 100

            # Expected Shortfall (ES) / CVaR
            es_value = final_values[final_values <= var_value].mean()
            es_pct = (es_value - initial_value) / initial_value * 100

            # Regulatory context
            if cl == 99.5:
                reg_use = "Solvency II (EU)"

            var_table.add_row(
                f"{cl}%",
                f"€{var_value:,.0f}",
                f"{var_pct:+.2f}%",
                f"€{es_value:,.0f}",
                f"{es_pct:+.2f}%",
                reg_use,
            )

        self.console.print(var_table)
        self.console.print(
            "\n[dim]VaR: Maximum loss at given confidence level | "
            "ES/CVaR: Average loss when VaR is breached (coherent risk measure)[/dim]\n"
        )

        # SECTION 5: DISTRIBUTION ANALYSIS

        percentile_table = Table(
            title="Portfolio Value Distribution - Full Spectrum",
            show_header=True,
            header_style="bold blue",
            border_style="blue",
        )
        percentile_table.add_column("Percentile", style="blue", width=12)
        percentile_table.add_column(
            "Value (€)", justify="right", style="bold white", width=18
        )
        percentile_table.add_column(
            "Return (%)", justify="right", style="cyan", width=15
        )
        percentile_table.add_column("Probability Context", style="dim", width=40)

        percentiles = [0.5, 5, 10, 25, 50, 75, 90, 95, 99, 99.5]
        for p in percentiles:
            p_value = np.percentile(final_values, p)
            p_return = (p_value - initial_value) / initial_value * 100

            return_color = "green" if p_return > 0 else "red"

            # Context
            if p == 0.5:
                context = "Worst 0.5% (1-in-200, Solvency II)"
            elif p == 5:
                context = "Worst 5% (1-in-20)"
            elif p == 50:
                context = "Median outcome (50/50)"
            elif p == 99.5:
                context = "Best 0.5% (top tier)"
            else:
                context = f"{p}% of outcomes below this"

            percentile_table.add_row(
                f"{p}th",
                f"€{p_value:,.0f}",
                f"[{return_color}]{p_return:+.2f}%[/{return_color}]",
                context,
            )

        self.console.print(percentile_table)
        self.console.print("\n")

        # SECTION 6: TAIL RISK & HIGHER MOMENTS

        tail_table = Table(
            title="Tail Risk Analysis & Higher Moments",
            show_header=True,
            header_style="bold red",
            border_style="red",
        )
        tail_table.add_column("Metric", style="red", width=38)
        tail_table.add_column("Value", justify="right", style="bold white", width=20)
        tail_table.add_column("Interpretation", style="dim", width=40)

        # Skewness
        skewness = ((total_returns - total_returns.mean()) ** 3).mean() / (
            total_returns.std() ** 3
        )
        skew_interp = (
            "Left tail risk (losses)"
            if skewness < -0.5
            else (
                "Right tail potential (gains)"
                if skewness > 0.5
                else "Approximately symmetric"
            )
        )
        skew_color = (
            "red" if skewness < -0.5 else "green" if skewness > 0.5 else "yellow"
        )
        tail_table.add_row(
            "Skewness", f"[{skew_color}]{skewness:.3f}[/{skew_color}]", skew_interp
        )

        # Kurtosis
        kurtosis = ((total_returns - total_returns.mean()) ** 4).mean() / (
            total_returns.std() ** 4
        )
        excess_kurtosis = kurtosis - 3
        kurt_interp = (
            "Fat tails - high extreme risk"
            if excess_kurtosis > 1
            else (
                "Thin tails - low extreme risk"
                if excess_kurtosis < -1
                else "Normal-like tail behavior"
            )
        )
        kurt_color = "red" if excess_kurtosis > 1 else "green"
        tail_table.add_row(
            "Excess Kurtosis",
            f"[{kurt_color}]{excess_kurtosis:.3f}[/{kurt_color}]",
            kurt_interp,
        )

        # Worst case scenario
        worst_value = final_values.min()
        worst_return = (worst_value - initial_value) / initial_value * 100
        tail_table.add_row(
            "Worst Case Scenario",
            f"[red]€{worst_value:,.0f}[/red]",
            f"Minimum simulated outcome ({worst_return:+.2f}%)",
        )

        # Best case scenario
        best_value = final_values.max()
        best_return = (best_value - initial_value) / initial_value * 100
        tail_table.add_row(
            "Best Case Scenario",
            f"[green]€{best_value:,.0f}[/green]",
            f"Maximum simulated outcome ({best_return:+.2f}%)",
        )

        # Range
        outcome_range = best_value - worst_value
        tail_table.add_row(
            "Outcome Range", f"€{outcome_range:,.0f}", "Spread between extremes"
        )

        # Semi-deviation (downside only)
        semi_deviation = (
            np.sqrt(((total_returns[total_returns < 0]) ** 2).mean()) * 100
            if (total_returns < 0).any()
            else 0
        )
        tail_table.add_row(
            "Semi-Deviation",
            f"{semi_deviation:.2f}%",
            "Standard deviation of losses only",
        )

        self.console.print(tail_table)
        self.console.print("\n")

        # SECTION 7: EXECUTIVE SUMMARY & REGULATORY ASSESSMENT

        summary_text = Text()
        summary_text.append(
            "╔═══════════════════════════════════════════════════════════╗\n",
            style="bold cyan",
        )
        summary_text.append(
            "║              EXECUTIVE SUMMARY & ASSESSMENT                ║\n",
            style="bold cyan",
        )
        summary_text.append(
            "╚═══════════════════════════════════════════════════════════╝\n\n",
            style="bold cyan",
        )

        summary_text.append("Simulation Parameters:\n", style="bold white")
        summary_text.append(f"  • Paths Simulated: ", style="bold")
        summary_text.append(f"{len(simulation_df):,}\n", style="cyan")
        summary_text.append(f"  • Time Horizon: ", style="bold")
        summary_text.append(f"{n_years} years\n", style="cyan")
        summary_text.append(f"  • Initial Value: ", style="bold")
        summary_text.append(f"€{initial_value:,.0f}\n", style="cyan")
        summary_text.append(f"  • Risk-Free Rate: ", style="bold")
        summary_text.append(f"{risk_free_rate*100:.2f}%\n\n", style="cyan")

        summary_text.append("Key Metrics:\n", style="bold white")
        summary_text.append(f"  • Expected Return: ", style="bold")
        summary_text.append(
            f"{mean_return_annualized:+.2f}% p.a.\n",
            style="green" if mean_return_annualized > 0 else "red",
        )
        summary_text.append(f"  • Volatility: ", style="bold")
        summary_text.append(f"{volatility_annualized:.2f}%\n", style="yellow")
        summary_text.append(f"  • Sharpe Ratio: ", style="bold")
        summary_text.append(f"{sharpe_ratio:.3f}\n", style="cyan")
        summary_text.append(f"  • Probability of Loss: ", style="bold")
        summary_text.append(f"{prob_loss:.2f}%\n\n", style=loss_color)

        summary_text.append("Solvency II Assessment:\n", style="bold white")
        summary_text.append(f"  • SCR (99.5% VaR): ", style="bold")
        summary_text.append(f"€{scr_market:,.0f} ({scr_ratio:.1f}%)\n", style=scr_color)
        summary_text.append(f"  • Solvency Ratio: ", style="bold")
        summary_text.append(f"{min(solvency_ratio, 999):.0f}%\n", style=ratio_color)

        # Regulatory status
        if solvency_ratio >= 150:
            reg_status = "Strong Capital Position"
            reg_color = "green"
        elif solvency_ratio >= 100:
            reg_status = "Adequate Capital Position"
            reg_color = "yellow"
        else:
            reg_status = "Capital Concern - Below Minimum"
            reg_color = "red"

        summary_text.append(f"  • Regulatory Status: ", style="bold")
        summary_text.append(f"{reg_status}\n\n", style=reg_color)

        # Overall risk assessment
        summary_text.append("Overall Risk Profile: ", style="bold white")

        risk_score = 0
        if sharpe_ratio > 1.0:
            risk_score += 2
        elif sharpe_ratio > 0.5:
            risk_score += 1

        if prob_loss < 15:
            risk_score += 2
        elif prob_loss < 25:
            risk_score += 1

        if solvency_ratio >= 150:
            risk_score += 2
        elif solvency_ratio >= 100:
            risk_score += 1

        if risk_score >= 5:
            assessment = "FAVORABLE - Strong risk-return profile with adequate capital"
            assessment_color = "green"
        elif risk_score >= 3:
            assessment = "MODERATE - Acceptable risk-return with monitoring required"
            assessment_color = "yellow"
        else:
            assessment = "ELEVATED - Significant risk or capital concerns"
            assessment_color = "red"

        summary_text.append(f"{assessment}", style=f"bold {assessment_color}")

        summary_panel = Panel(
            summary_text,
            title="[bold white on blue] RISK ANALYSIS REPORT [/bold white on blue]",
            border_style="cyan",
            padding=(1, 2),
        )

        self.console.print(summary_panel)
        self.console.print("\n")

    def print_success(self, message: str) -> None:
        """Print success message."""
        self.console.print(f"[green]✓[/green] {message}")

    def print_error(self, message: str) -> None:
        """Print error message."""
        self.console.print(f"[red]✗[/red] {message}")

    def print_info(self, message: str) -> None:
        """Print info message."""
        self.console.print(f"[blue]i[/blue] {message}")

    def print_warning(self, message: str) -> None:
        """Print warning message."""
        self.console.print(f"[yellow]WARNING[/yellow] {message}")
