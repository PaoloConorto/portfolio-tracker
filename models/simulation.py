import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_t, t as student_t, norm


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
        self,
        mu,
        sigma,
        corr,
        weights,
        S0,
        shares,
        nu=6,
        dt=1 / 12,
        V0=1.0,
        rng=42,
    ):
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.corr = np.asarray(corr, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.k = self.mu.shape[0]
        assert self.sigma.shape == (self.k,)
        assert self.corr.shape == (self.k, self.k)
        assert self.weights.shape == (self.k,)
        self.S0 = np.asarray(S0, dtype=float)
        assert self.S0.shape == (self.k,)

        self.L = np.linalg.cholesky(self.corr)
        self.nu = float(nu)
        assert self.nu > 2.0, "nu must be > 2"
        self.dt = float(dt)
        self.sqrt_dt = np.sqrt(self.dt)
        self.V0 = float(V0)
        self.rng = np.random.default_rng(seed=rng)
        self.shares = np.array(shares)

        # GBM drift term per step
        self.mu_step = (self.mu - 0.5 * self.sigma**2) * self.dt
        self.sigma_step = self.sigma * self.sqrt_dt
        self._mvt = multivariate_t(
            loc=np.zeros(self.k), shape=self.corr, df=self.nu_cop, seed=self.rng
        )
        self.t_scale = np.sqrt(self.nu_marg / (self.nu_marg - 2.0))

    def _t_copula_gaussian_shocks(self, n_rows):
        T = self._mvt.rvs(size=n_rows)
        if T.ndim == 1:
            T = T[None, :]
        U = student_t.cdf(T, df=self.nu)
        eps = np.finfo(float).eps
        Z = norm.ppf(np.clip(U, eps, 1 - eps))
        return Z

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

    def garch_simulate(
        self,
        h0,
        garch,
        n_years: int,
        n_paths: int,
        return_asset_paths: bool = False,
        batch_size: int = None,
    ):
        """Monte-Carlo GBM simulation with t-copula dependent shocks and GARCH"""

        N = int(round(n_years / self.dt))
        cols = [f"t_{i}" for i in range(N + 1)]
        batch_size = n_paths if batch_size is None else batch_size

        V_blocks = []
        S_blocks = [] if return_asset_paths else None
        logS0 = np.log(self.S0)[None, None, :]  # (1,1,k)
        omega = np.asarray(garch["omega"], float)[None, None, :]  # (1,1,k)
        alpha = np.asarray(garch["alpha"], float)[None, None, :]
        beta = np.asarray(garch["beta"], float)
        h0 = np.asarray(h0, float)[None, None, :]
        mu = self.mu[None, None, :]

        done = 0
        while done < n_paths:
            b = min(batch_size, n_paths - done)

            # 1) Draw all shocks for this batch and reshape to (b,N,k)
            Z = self._t_copula_gaussian_shocks(b * N).reshape(b, N, self.k)

            # 2) g_t = alpha * z_t^2 + beta  (broadcast on asset dim)
            g = alpha * (Z**2) + beta  # (b,N,k)

            # 3) G = cumprod(g) along time; G_prev = [ones, G[:,:-1,:]]
            G = np.cumprod(g, axis=1)  # G[:, t, :] = ∏_{j=0..t} g_j
            ones = np.ones((b, 1, self.k), dtype=float)
            G_prev = np.concatenate(
                [ones, G[:, :-1, :]], axis=1
            )  # G_{t-1}, with G_{-1}=1

            # 4) prefix sums of 1/G for Σ_{m=0}^{t-1} 1/G_m
            invG = 1.0 / G  # (b,N,k)
            cs_invG = np.cumsum(invG, axis=1)  # sum up to index t
            cs_invG_prev = np.concatenate(
                [np.zeros((b, 1, self.k)), cs_invG[:, :-1, :]], axis=1
            )

            # 5) h_t for t=0..N-1 (annualized variance used in step t)
            h_t = G_prev * (h0 + omega * cs_invG_prev)  # (b,N,k)

            # 6) log-return increments and price paths (vectorized cumsum)
            dlogS = (mu - 0.5 * h_t) * self.dt + np.sqrt(h_t * self.dt) * Z  # (b,N,k)
            logS = np.concatenate(
                [
                    logS0.repeat(b, axis=0),
                    logS0.repeat(b, axis=0) + np.cumsum(dlogS, axis=1),
                ],
                axis=1,
            )
            S = np.exp(logS)  # (b,N+1,k)

            # 7) portfolio aggregation
            V = (S * self.shares[None, None, :]).sum(axis=2)  # (b,N+1)

            V_blocks.append(V)
            if return_asset_paths:
                S_blocks.append(S)

            done += b

        V_all = np.vstack(V_blocks)
        dfV = pd.DataFrame(V_all, columns=cols)

        if return_asset_paths:
            S_all = np.concatenate(S_blocks, axis=0)
            assets = {
                f"asset_{i}": pd.DataFrame(S_all[:, :, i], columns=cols)
                for i in range(self.k)
            }
            return dfV, assets
        return dfV
