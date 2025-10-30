"""
Portfolio controller - orchestrates the application logic.
Manages interaction between models, views, and services.
"""

from datetime import datetime, timedelta
from models.portfolio import Portfolio
from models.asset import Asset
from models.simulation import TCopulaGBMSimulator
from services.API_prices import PriceService
from views.table_view import TableView
from views.plot_view import PlotView


class PortfolioController:
    """
    Main controller for the portfolio tracking application.

    Coordinates between models, views, and services following MVC pattern.
    """

    def __init__(self):
        """Initialize controller with all necessary components."""
        self.portfolio = Portfolio()
        self.price_service = PriceService()
        self.table_view = TableView()
        self.chart_view = PlotView()

    def add_asset(
        self,
        ticker: str,
        sector: str,
        asset_class: str,
        quantity: float,
        purchase_price: float,
    ) -> None:
        """
        Add a new asset to the portfolio.

        Args:
            ticker: Stock ticker symbol
            sector: Industry sector
            asset_class: Type of asset
            quantity: Number of shares
            purchase_price: Price per share at purchase
        """
        try:
            # Create asset
            asset = Asset(
                ticker=ticker.upper(),
                sector=sector,
                asset_class=asset_class,
                quantity=quantity,
                purchase_price=purchase_price,
                purchase_date=datetime.now(),
                drift=0.068,
                volatility=0.16,
            )

            # Fetch current price
            current_price = self.price_service.get_current_price(ticker)
            if current_price:
                asset.update_price(current_price)

            # Add real drift and volatility
            params = self.price_service.get_drift_volatility(ticker)
            asset.add_real_parameters(params)
            print(params)
            # Add to portfolio

            self.portfolio.add_asset(asset)

            self.table_view.print_success(
                f"Added {quantity} shares of {ticker} at €{purchase_price:.2f}"
            )

            if current_price:
                self.table_view.print_info(f"Current price: €{current_price:.2f}")

        except Exception as e:
            self.table_view.print_error(f"Failed to add asset: {e}")

    def remove_asset(self, ticker: str) -> None:
        """
        Remove an asset from the portfolio.

        Args:
            ticker: Ticker symbol of asset to remove
        """
        if self.portfolio.remove_asset(ticker):
            self.table_view.print_success(f"Removed {ticker} from portfolio")
        else:
            self.table_view.print_error(f"Asset {ticker} not found in portfolio")

    def view_portfolio(self) -> None:
        """Display the current portfolio with all assets."""
        if not self.portfolio.assets:
            self.table_view.print_warning("Portfolio is empty. Add some assets first!")
            return

        # Display portfolio table
        self.table_view.display_portfolio(self.portfolio.assets)

        # Display summary statistics
        self.table_view.display_portfolio_stats(
            total_value=self.portfolio.total_value,
            total_cost=self.portfolio.total_cost,
            total_pl=self.portfolio.total_profit_loss,
            total_pl_pct=self.portfolio.total_profit_loss_pct,
        )

    def show_weights(self, by: str = "asset") -> None:
        """
        Display portfolio weights.

        Args:
            by: How to group weights ("asset", "sector", or "class")
        """
        if not self.portfolio.assets:
            self.table_view.print_warning("Portfolio is empty")
            return

        if by == "asset":
            weights = self.portfolio.get_weights()
            title = "Asset Weights"
        elif by == "sector":
            weights = self.portfolio.get_sector_weights()
            title = "Sector Weights"
        elif by == "class":
            weights = self.portfolio.get_asset_class_weights()
            title = "Asset Class Weights"
        else:
            self.table_view.print_error(f"Invalid grouping: {by}")
            return

        self.table_view.display_weights(weights, title)

        # Also show pie chart
        self.chart_view.plot_portfolio_allocation(weights, title)

    def show_price_history(self, ticker: str, days: int = 365) -> None:
        """
        Display price history for a ticker.

        Args:
            ticker: Stock ticker symbol
            days: Number of days of history
        """
        ticker = ticker.upper()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        self.table_view.print_info(f"Fetching {days} days of data for {ticker}...")

        df = self.price_service.get_historical_prices(ticker, start_date, end_date)

        if df.empty:
            self.table_view.print_error(f"No data found for {ticker}")
            return

        self.chart_view.plot_price_history(ticker, df)

    def compare_tickers(
        self,
        portfolio_tickers: bool,
        tickers: list,
        days: int = 365,
        normalize: bool = True,
    ) -> None:
        """
        Compare multiple tickers on one chart.

        Args:
            portfolio_tickers: whether to use the portfolio tickers
            tickers: List of ticker symbols
            days: Number of days of history
            normalize: Whether to normalize prices
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        if portfolio_tickers:
            if not self.portfolio.assets:
                self.table_view.print_warning(
                    "Portfolio is empty. Add some assets first!"
                )
                return
            for a in self.portfolio.assets:
                print(a.ticker)
            tickers = [a.ticker for a in self.portfolio.assets]
        data = {}
        for ticker in tickers:
            ticker = ticker.upper()
            self.table_view.print_info(f"Fetching data for {ticker}...")
            df = self.price_service.get_historical_prices(ticker, start_date, end_date)
            if not df.empty:
                data[ticker] = df

        if not data:
            self.table_view.print_error("No data found for any ticker")
            return

        self.chart_view.plot_multiple_tickers(data, normalize)

    def show_analytics(self) -> None:
        """Display comprehensive portfolio analytics."""
        if not self.portfolio.assets:
            self.table_view.print_warning("Portfolio is empty")
            return

        # Show portfolio overview
        self.view_portfolio()

        print("\n")

        # Show sector summary
        sector_summary = self.portfolio.get_sector_summary()
        self.table_view.display_summary(sector_summary, "Sector Analysis")

        print("\n")

        # Show asset class summary
        class_summary = self.portfolio.get_asset_class_summary()
        self.table_view.display_summary(class_summary, "Asset Class Analysis")

        print("\n")

        # Show sector comparison chart
        if sector_summary:
            self.chart_view.plot_sector_comparison(sector_summary)

    def update_all_prices(self) -> None:
        """Update current prices for all assets in the portfolio."""
        if not self.portfolio.assets:
            self.table_view.print_warning("Portfolio is empty")
            return

        self.table_view.print_info("Updating prices for all assets...")

        tickers = [asset.ticker for asset in self.portfolio.assets]
        prices = self.price_service.get_current_prices(tickers)

        updated = 0
        for asset in self.portfolio.assets:
            if asset.ticker in prices:
                asset.update_price(prices[asset.ticker])
                updated += 1

        self.table_view.print_success(
            f"Updated prices for {updated}/{len(self.portfolio.assets)} assets"
        )

    def get_portfolio(self) -> Portfolio:
        """
        Get the portfolio object.

        Returns:
            Portfolio instance
        """
        return self.portfolio

    def naive_simulation(self, years: int, paths: int, interval: int = 5):
        self.update_all_prices()
        assets = self.portfolio.assets.copy()
        tickers = [a.ticker for a in assets]

        mu = [a.drift for a in assets]

        sigma = [a.volatility for a in assets]

        corr = self.price_service.get_correlations(tickers)

        w = self.portfolio.get_weights()

        weights = [w[a] / 100 for a in tickers]

        S0 = [a.current_price for a in assets]

        V0 = self.portfolio.total_value

        sim = TCopulaGBMSimulator(mu, sigma, corr, weights, S0, V0)

        df = sim.simulate(n_years=years, n_paths=paths)

        print("check df exists")
        self.chart_view.plot_simulation_results(
            simulation_df=df,
            title=f"{years}-Year Portfolio Simulation",
            confidence_levels=[interval, 100 - interval],
        )
