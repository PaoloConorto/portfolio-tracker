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
