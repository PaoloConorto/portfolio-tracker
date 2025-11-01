"""
CLI view for handling user interface and command input.
Provides the main interactive loop for the application.
"""

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from controllers.portfolio_controller import PortfolioController


class CLIView:
    """
    Command-line interface for the portfolio tracker.

    Handles user input and displays menus.
    """

    def __init__(self, controller: "PortfolioController"):
        """
        Initialize CLI view.

        Args:
            controller: Portfolio controller instance
        """
        self.controller = controller
        self.console = Console()
        self.running = True

    def display_banner(self) -> None:
        """Display welcome banner."""
        banner = """
[bold cyan]╔══════════════════════════════════════════════╗
║   Portfolio Tracker - Investment Manager    ║
║                   a.s.r.                    ║
╚══════════════════════════════════════════════╝[/bold cyan]
        """
        self.console.print(banner)

    def display_menu(self) -> None:
        """Display main menu options."""
        menu = """
[bold cyan]Main Menu:[/bold cyan]
  [green]1.[/green] Add Asset
  [green]2.[/green] View Portfolio
  [green]3.[/green] Show Weights (Asset/Sector/Class)
  [green]4.[/green] View Price History
  [green]5.[/green] Compare Multiple Tickers
  [green]6.[/green] Portfolio Analytics
  [green]7.[/green] Remove Asset
  [green]8.[/green] Update Prices
  [green]9.[/green] Run Monte Carlo Simulation
  [red]0.[/red] Exit
        """
        self.console.print(Panel(menu, border_style="blue"))

    def run(self) -> None:
        """Main application loop."""
        self.display_banner()

        while self.running:
            self.display_menu()
            choice = Prompt.ask(
                "[bold cyan]Select an option[/bold cyan]",
                choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                default="2",
            )

            self.console.print()  # Add spacing

            try:
                self.handle_choice(choice)
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")

            if self.running:
                self.console.print()
                input("Press Enter to continue...")
                self.console.clear()

    def handle_choice(self, choice: str) -> None:
        """
        Handle user menu choice.

        Args:
            choice: User's menu selection
        """
        if choice == "0":
            self.handle_exit()
        elif choice == "1":
            self.handle_add_asset()
        elif choice == "2":
            self.handle_view_portfolio()
        elif choice == "3":
            self.handle_show_weights()
        elif choice == "4":
            self.handle_price_history()
        elif choice == "5":
            self.handle_compare_tickers()
        elif choice == "6":
            self.handle_analytics()
        elif choice == "7":
            self.handle_remove_asset()
        elif choice == "8":
            self.handle_update_prices()
        elif choice == "9":
            self.handle_simulation()

    def handle_add_asset(self) -> None:
        """Handle adding a new asset."""
        self.console.print("[bold cyan]Add New Asset[/bold cyan]\n")

        ticker = Prompt.ask("Enter ticker symbol (e.g., AAPL)").upper()

        # Validate ticker
        if not self.controller.price_service.validate_ticker(ticker):
            self.console.print(f"[red]Invalid ticker: {ticker}[/red]")
            return

        # Get ticker info for suggestions
        info = self.controller.price_service.get_ticker_info(ticker)
        suggested_sector = info.get("sector", "Technology")

        sector = Prompt.ask(
            f"Enter sector (suggested: {suggested_sector})", default=suggested_sector
        )

        asset_class = Prompt.ask(
            "Enter asset class",
            choices=["Stock", "ETF", "Bond", "Crypto", "Other"],
            default="Stock",
        )

        quantity = float(Prompt.ask("Enter quantity"))
        purchase_price = float(Prompt.ask("Enter your Purchase Value"))

        self.controller.add_asset(ticker, sector, asset_class, quantity, purchase_price)

    def handle_view_portfolio(self) -> None:
        """Handle viewing the portfolio."""
        self.controller.view_portfolio()

    def handle_show_weights(self) -> None:
        """Handle displaying portfolio weights."""
        self.console.print("[bold cyan]Portfolio Weights[/bold cyan]\n")

        weight_type = Prompt.ask(
            "Show weights by", choices=["asset", "sector", "class"], default="asset"
        )

        self.controller.show_weights(by=weight_type)

    def handle_price_history(self) -> None:
        """Handle displaying price history."""
        ticker_in_portfolio = Confirm.ask(
            "check Ticker in the portfolio?", default=False
        )
        if not ticker_in_portfolio:
            ticker = Prompt.ask("Enter ticker symbol").upper()
        else:
            ticks = [a.ticker for a in self.controller.portfolio.assets]
            ticker = Prompt.ask("Which ticker do you want to check?", choices=ticks)
        days = int(Prompt.ask("Enter number of days", default="365"))

        self.controller.show_price_history(ticker, days)

    def handle_compare_tickers(self) -> None:
        """Handle comparing multiple tickers."""

        portfolio_tickers = Confirm.ask("Use Portfolio Tickers?", default=False)
        tickers = []
        if not portfolio_tickers:
            tickers_input = Prompt.ask(
                "Enter tickers separated by comma (e.g., AAPL,MSFT,GOOGL)"
            )
            tickers = [t.strip().upper() for t in tickers_input.split(",")]

        days = int(Prompt.ask("Enter number of days", default="365"))
        normalize = Confirm.ask("Normalize prices?", default=True)

        self.controller.compare_tickers(portfolio_tickers, tickers, days, normalize)

    def handle_analytics(self) -> None:
        """Handle portfolio analytics display."""
        self.controller.show_analytics()

    def handle_remove_asset(self) -> None:
        """Handle removing an asset."""
        tickers = [a.ticker for a in self.controller.portfolio.assets]
        ticker = Prompt.ask("Enter ticker to remove", choices=tickers).upper()

        if Confirm.ask(f"Are you sure you want to remove {ticker}?"):
            self.controller.remove_asset(ticker)

    def handle_update_prices(self) -> None:
        """Handle updating all asset prices."""
        self.controller.update_all_prices()

    def handle_simulation(self) -> None:
        """Handle Monte Carlo simulation."""
        self.console.print("[bold cyan]Monte Carlo Simulation[/bold cyan]\n")
        years = int(Prompt.ask("Simulation years", default="15"))
        paths = int(Prompt.ask("Number of paths", default="100,000"))

        self.console.print(
            f"\n[yellow]Simulation setup: {years} years, {paths:,} paths[/yellow]"
        )
        interval = float(
            Prompt.ask(r"What is the $\alpha$ for your Confidence interval", default=5)
        )
        garch = Confirm.ask("Do you want GARCH volatility?")
        self.controller.naive_simulation(garch, years, paths, interval)

    def handle_exit(self) -> None:
        """Handle application exit."""
        if Confirm.ask("Are you sure you want to exit?"):
            self.console.print(
                "\n[green]Thank you for using Portfolio Tracker! by Paolo Conorto[/green]"
            )
            self.running = False
