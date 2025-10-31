from models.simulation import TCopulaGBMSimulator
from views.plot_view import PlotView
import numpy as np

# 1. Set up your simulator
simulator = TCopulaGBMSimulator(
    mu=np.array([0.08, 0.10, 0.06]),  # Expected returns
    sigma=np.array([0.15, 0.20, 0.10]),  # Volatilities
    corr=np.array(
        [[1.0, 0.6, 0.4], [0.6, 1.0, 0.5], [0.4, 0.5, 1.0]]  # Correlation matrix
    ),
    weights=np.array([0.4, 0.4, 0.2]),  # Portfolio weights
    S0=np.array([100.0, 200, 300]),  # Initial prices
    shares=[2.0, 1.0, 1.0],
    nu=6,  # t-copula df
    dt=1 / 12,  # Monthly
    V0=700.0,  # €100k initial
    rng=42,  # Random seed
)

# 2. Run simulation
simulation_df = simulator.simulate(
    n_years=15, n_paths=1000, return_asset_paths=False  # Portfolio only
)

# 3. Plot results
chart_view = PlotView()
chart_view.plot_simulation_results(
    simulation_df=simulation_df,
    title="15-Year Portfolio Simulation",
    confidence_levels=[5, 95],
    save_path="results.png",
)
