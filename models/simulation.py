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
        dfs,
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
        self.df_marginals = dfs

        # GBM drift term per step
        self.mu_step = (self.mu - 0.5 * self.sigma**2) * self.dt
        self.sigma_step = self.sigma * self.sqrt_dt
        self._mvt = multivariate_t(
            loc=np.zeros(self.k), shape=self.corr, df=self.nu, seed=self.rng
        )

    def _t_copula_gaussian_shocks(self, n_rows):
        T = self._mvt.rvs(size=n_rows)
        if T.ndim == 1:
            T = T[None, :]
        U = student_t.cdf(T, df=self.nu)
        eps = np.finfo(float).eps
        Z = norm.ppf(np.clip(U, eps, 1 - eps))
        return Z

    def _t_copula_t_shocks(self, n_rows):
        T = self._mvt.rvs(size=n_rows)
        if T.ndim == 1:
            T = T[None, :]
        U = student_t.cdf(T, df=self.nu)
        eps = np.finfo(float).eps
        Z = np.clip(U, eps, 1 - eps)
        df = self.df_marginals
        t_marg = student_t.ppf(Z, df=df)
        std_factor = np.sqrt(df / (df - 2.0))
        std_factor = std_factor[None, :]
        shocks = t_marg / std_factor
        return shocks

    def simulate(
        self,
        n_years: int,
        n_paths: int,
        t_marg: bool = False,
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
            if t_marg:
                Z = self._t_copula_t_shocks(n_rows)
            else:
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
        garch_params: dict,
        n_years: int,
        n_paths: int,
        return_asset_paths: bool = False,
        batch_size: int = None,
    ):
        """
        Monte Carlo GBM with GARCH(1,1) volatility and t-copula dependence.

        GARCH: h_t = omega + alpha * epsilon_{t-1}^2 + beta * h_{t-1}
        Returns: r_t = mu * dt + sqrt(h_t * dt) * Z_t
        """
        N = int(round(n_years / self.dt))
        cols = [f"t_{i}" for i in range(N + 1)]
        batch_size = n_paths if batch_size is None else batch_size

        omega = np.asarray(garch_params["omega"], dtype=float).reshape(1, 1, self.k)
        alpha = np.asarray(garch_params["alpha"], dtype=float).reshape(1, 1, self.k)
        beta = np.asarray(garch_params["beta"], dtype=float).reshape(1, 1, self.k)
        h0 = np.asarray(garch_params["h0"], dtype=float).reshape(1, 1, self.k)
        mu = np.asarray(garch_params["h0"], dtype=float).reshape(1, 1, self.k)

        persistence = alpha + beta
        if np.any(persistence >= 1.0):
            print(f"WARNING: GARCH not stationary. Adjusting...")
            scale = 0.98 / persistence
            alpha *= scale
            beta *= scale

        logS0 = np.log(self.S0).reshape(1, 1, self.k)

        V_blocks = []
        S_blocks = [] if return_asset_paths else None

        done = 0
        while done < n_paths:
            b = min(batch_size, n_paths - done)

            Z = self._t_copula_gaussian_shocks(b * N).reshape(b, N, self.k)

            h_t = np.zeros((b, N, self.k))
            log_returns = np.zeros((b, N, self.k))

            h_t[:, 0, :] = h0.reshape(1, self.k)

            # First return
            log_returns[:, 0, :] = (
                (mu - 0.5 * h_t[:, 0:1, :]) * self.dt
                + np.sqrt(h_t[:, 0:1, :] * self.dt) * Z[:, 0:1, :]
            ).squeeze(1)

            # GARCH recursion
            for t in range(1, N):

                residual_sq = (log_returns[:, t - 1 : t, :] - mu * self.dt) ** 2
                epsilon_sq = residual_sq / (h_t[:, t - 1 : t, :] * self.dt + 1e-10)

                h_t[:, t : t + 1, :] = (
                    omega + alpha * epsilon_sq + beta * h_t[:, t - 1 : t, :]
                )
                h_t[:, t : t + 1, :] = np.maximum(h_t[:, t : t + 1, :], 1e-8)

                log_returns[:, t : t + 1, :] = (
                    mu - 0.5 * h_t[:, t : t + 1, :]
                ) * self.dt + np.sqrt(h_t[:, t : t + 1, :] * self.dt) * Z[
                    :, t : t + 1, :
                ]

            cumulative_log_returns = np.cumsum(log_returns, axis=1)
            logS = np.concatenate(
                [
                    logS0.repeat(b, axis=0),
                    logS0.repeat(b, axis=0) + cumulative_log_returns,
                ],
                axis=1,
            )

            S = np.exp(logS)
            V = (S * self.shares.reshape(1, 1, self.k)).sum(axis=2)
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
