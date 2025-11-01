import numpy as np
import pandas as pd
from scipy.stats import multivariate_t, t as student_t, norm


class TCopulaGARCHGBM:
    def __init__(
        self,
        mu,
        garch,
        h0,
        corr,
        weights,
        S0=1.0,
        nu_cop=6,
        nu_marg=7,
        dt=1 / 12,
        V0=1.0,
        rng=None,
    ):
        self.mu = np.asarray(mu, float)
        self.k = self.mu.size
        self.omega = np.asarray(garch["omega"], float)
        self.alpha = np.asarray(garch["alpha"], float)
        self.beta = np.asarray(garch["beta"], float)
        self.h0 = np.asarray(h0, float)
        self.corr = np.asarray(corr, float)
        self.weights = np.asarray(weights, float)
        self.S0 = (
            np.full(self.k, float(S0)) if np.isscalar(S0) else np.asarray(S0, float)
        )
        assert (
            self.omega.shape
            == self.alpha.shape
            == self.beta.shape
            == self.h0.shape
            == self.weights.shape
            == (self.k,)
        )
        assert self.corr.shape == (self.k, self.k)
        self.nu_cop = float(nu_cop)
        self.nu_marg = float(nu_marg)
        assert self.nu_cop > 2 and self.nu_marg > 2
        self.dt = float(dt)
        self.sqrt_dt = np.sqrt(self.dt)
        self.V0 = float(V0)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.shares = (self.weights * self.V0) / self.S0
        self._mvt = multivariate_t(
            loc=np.zeros(self.k), shape=self.corr, df=self.nu_cop, seed=self.rng
        )
        self.std_t_scale = np.sqrt(self.nu_marg / (self.nu_marg - 2.0))

    def _copula_std_t_innovations(self, n_rows):
        T = self._mvt.rvs(size=n_rows)
        if T.ndim == 1:
            T = T[None, :]
        U = student_t.cdf(T, df=self.nu_cop)
        eps = np.finfo(float).eps
        Z = norm.ppf(np.clip(U, eps, 1 - eps))
        return Z  # shape (n_rows, k)

    def simulate(
        self,
        n_years: int,
        n_paths: int,
        return_asset_paths: bool = False,
        batch_size: int = None,
    ):
        N = int(round(n_years / self.dt))
        cols = [f"t_{i}" for i in range(N + 1)]
        batch_size = n_paths if batch_size is None else batch_size

        V_blocks = []
        S_blocks = [] if return_asset_paths else None
        logS0 = np.log(self.S0)[None, None, :]  # (1,1,k)
        omega = self.omega[None, None, :]  # (1,1,k)
        alpha = self.alpha[None, None, :]
        beta = self.beta[None, None, :]
        h0 = self.h0[None, None, :]
        mu = self.mu[None, None, :]

        done = 0
        while done < n_paths:
            b = min(batch_size, n_paths - done)

            # 1) Draw all shocks for this batch and reshape to (b,N,k)
            Z = self._copula_std_t_innovations(b * N).reshape(b, N, self.k)

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


from views.plot_view import PlotView
from views.table_view import TableView

if __name__ == "__main__":
    k = 3
    mu = np.array([0.06, 0.04, 0.20])
    garch = {
        "omega": np.array([0.02, 0.015, 0.025]),  # annual var floor
        "alpha": np.array([0.05, 0.07, 0.06]),
        "beta": np.array([0.90, 0.88, 0.89]),
    }
    h0 = np.array([0.20**2, 0.15**2, 0.30**2])  # initial annualized variance
    corr = np.array([[1.00, 0.5, 0.3], [0.5, 1.00, 0.2], [0.3, 0.2, 1.00]])
    weights = np.array([0.4, 0.4, 0.2])
    S0 = np.array([100.0, 80.0, 50.0])

    sim = TCopulaGARCHGBM(
        mu,
        garch,
        h0,
        corr,
        weights,
        S0=S0,
        nu_cop=5,
        nu_marg=7,
        dt=1 / 12,
        V0=10000.0,
        rng=np.random.default_rng(7),
    )
    df_paths = sim.simulate(n_years=15, n_paths=10_000, batch_size=1000)
    PlotView().plot_simulation_results(df_paths, confidence_levels=[5, 95])
    TableView().display_simulation_risk_metrics(df_paths, 10000.0, 15)
