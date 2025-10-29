"""
Configuration settings for the Portfolio Tracker application.
"""

from pathlib import Path

# Application metadata
APP_NAME = "Portfolio Tracker"
APP_VERSION = "1.0.0"
ORGANIZATION = "a.s.r."

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
CHARTS_DIR = DATA_DIR / "charts"

DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

# API Settings
DEFAULT_CURRENCY = "USD"
PRICE_UPDATE_INTERVAL = 3600  # seconds (1 hour)

# Simulation setting for now temporaty
DEFAULT_SIMULATION_YEARS = 15
DEFAULT_SIMULATION_PATHS = 10_0000
DEFAULT_ANNUAL_RETURN = 0.08
DEFAULT_ANNUAL_VOLATILITY = 0.15

# Display settings
MAX_TABLE_WIDTH = 150
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Chart settings
CHART_DPI = 600
CHART_STYLE = "seaborn-v0_8-darkgrid"
