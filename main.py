"""
Portfolio Tracker - Main Entry Point

A command-line application for tracking investment portfolios.
"""

from controllers.portfolio_controller import PortfolioController
from views.cli_view import CLIView


def main():
    """Initialize and run the portfolio tracker application."""
    controller = PortfolioController()
    cli = CLIView(controller)
    cli.run()


if __name__ == "__main__":
    main()
