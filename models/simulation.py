import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t


class TCopulaGBMSimulator:
    """
    Vectorized Geometric Brownian Motion simulator with t-copula dependence.

    Inputs
    ------
    mu : (k,) array-like
        Annualized drifts per asset.
    sigma : (k,) array-like
        Annualized vols per asset (>=0).
    corr : (k,k) array-like
        Positive-definite correlation matrix.
    weights : (k,) array-like
        Portfolio weights that apply to initial capital V0 (buy-and-hold).
        Sum does not need to be 1. Can include cash by adding a zero-vol asset.
    S0 : (k,) array-like or float, optional
        Initial prices per asset. If float, broadcasted. Default 1.0 for each asset.
    nu : float, optional
        Degrees of freedom for the t-copula. Typical 4-10. Default 6.
    dt : float, optional
        Time step in years. Monthly = 1/12. Default 1/12.
    V0 : float, optional
        Initial portfolio notional. Default 1.0.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Output of simulate(...)
    -----------------------
    DataFrame with shape (n_paths, n_steps+1). Each row is a path.
    Column 0 is t=0, subsequent columns are monthly marks.
    """

    def __init__(
        self, mu, sigma, corr, weights, S0=1.0, nu=6, dt=1 / 12, V0=1.0, rng=42
    ):
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.corr = np.asarray(corr, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.k = self.mu.shape[0]
        assert self.sigma.shape == (self.k,)
        assert self.corr.shape == (self.k, self.k)
        assert self.weights.shape == (self.k,)

        if np.isscalar(S0):
            self.S0 = np.full(self.k, float(S0))
        else:
            self.S0 = np.asarray(S0, dtype=float)
            assert self.S0.shape == (self.k,)

        # Cholesky for correlation
        self.L = np.linalg.cholesky(self.corr)
        self.nu = float(nu)
        assert self.nu > 2.0, "nu must be > 2 for finite variance in the copula driver."

        self.dt = float(dt)
        self.sqrt_dt = np.sqrt(self.dt)
        self.V0 = float(V0)
        self.rng = np.random.default_rng(seed=rng)

        # Precompute buy-and-hold shares from initial weights
        # Shares = (w_i * V0) / S0_i
        self.shares = (self.weights * self.V0) / self.S0

        # GBM drift term per step
        self.mu_step = (self.mu - 0.5 * self.sigma**2) * self.dt
        self.sigma_step = self.sigma * self.sqrt_dt

    def _t_copula_gaussian_shocks(self, n_rows):
        """
        Generate Z ~ N(0,1) margins with t-copula dependence across k assets.
        Vectorized:
          1) Draw standard normals E ~ N(0, I_k)
          2) Correlate: Z_tilde = E @ L^T
          3) Scale by sqrt(nu / ChiSquare_nu) per row -> multivariate t
          4) Convert componentwise to uniforms via t CDF
          5) Map to Gaussian via Phi^{-1} for GBM marginals
        """
        # Step 1–2: correlated normals
        E = self.rng.normal(size=(n_rows, self.k))
        Z_corr = E @ self.L.T
        chi2 = self.rng.chisquare(df=self.nu, size=(n_rows, 1))
        t_scaled = Z_corr / np.sqrt(chi2 / self.nu)  # multivariate t

        U = student_t.cdf(t_scaled, df=self.nu)

        # Step 5: map uniforms to standard normal
        eps = np.finfo(float).eps
        U = np.clip(U, eps, 1.0 - eps)
        Z = norm.ppf(U)
        return Z  # shape (n_rows, k)

    def simulate(
        self,
        n_years: int,
        n_paths: int,
        return_asset_paths: bool = False,
        batch_size: int = None,
    ):
        """
        Simulate portfolio GBM paths with t-copula dependence.

        n_years : int
        n_paths : int
        return_asset_paths : bool
            If True, also return a dict of DataFrames for asset-level paths.
        batch_size : int or None
            If set, compute in batches of paths to reduce peak memory.
        """
        n_steps = int(np.round(n_years / self.dt))
        col_names = [f"t_{i}" for i in range(n_steps + 1)]

        if batch_size is None:
            batch_size = n_paths

        portfolio_blocks = []
        asset_blocks = [] if return_asset_paths else None
        done = 0
        while done < n_paths:
            b = min(batch_size, n_paths - done)

            # Draw all shocks for this batch and all steps at once
            n_rows = b * n_steps
            Z = self._t_copula_gaussian_shocks(n_rows)  # (b*n_steps, k)
            Z = Z.reshape(b, n_steps, self.k)

            # Per-step log-returns
            # r_t = mu_step + sigma_step * Z_t
            r = self.mu_step + self.sigma_step * Z  # shape (b, n_steps, k)

            # Cumulate to prices, include t=0
            logS0 = np.log(self.S0)
            logS = logS0 + np.cumsum(r, axis=1)
            S = np.exp(
                np.concatenate([np.broadcast_to(logS0, (b, 1, self.k)), logS], axis=1)
            )  # (b, n_steps+1, k)

            # Portfolio value: buy-and-hold shares fixed from t=0
            V = (S * self.shares).sum(axis=2)  # (b, n_steps+1)

            portfolio_blocks.append(V)

            if return_asset_paths:
                # One DataFrame per asset in this batch
                asset_blocks.append(S)

            done += b

        V_all = np.vstack(portfolio_blocks)  # (n_paths, n_steps+1)
        df = pd.DataFrame(V_all, columns=col_names)

        if return_asset_paths:
            S_all = np.concatenate(asset_blocks, axis=0)  # (n_paths, n_steps+1, k)
            assets = {
                f"asset_{i}": pd.DataFrame(S_all[:, :, i], columns=col_names)
                for i in range(self.k)
            }
            return df, assets

        return df
