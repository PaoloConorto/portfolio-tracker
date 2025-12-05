# portfolio-tracker

cli portfolio tracker with mvc and quick risk sim. no db. local only.

## quick start

- python 3.11+
- setup:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
- tweak params in `config.py` if needed
- run:
  ```bash
  python main.py
  ```

## what it does

- add tickers and builds a small portfolio
- pull prices and plots
- plots all prices within the portfolio
- shows weights and values by asset/class/sector
- runs a long-horizon multi-path risk simulation (follows from GBM with dependence structure governed by t-copula (fixed $\nu=6$)  shocks, optional GARCH inclusion)
- prints tables and charts in the cli

## layout

`controllers/  models/  services/  tests/  utils/  views/  config.py  main.py  requirements.txt`

## notes

- read code, tweak `config.py`, run `main.py`
- Code done with the the assistance of LLMs
- tickers should follow from [Yahoo finance](https://finance.yahoo.com/?guccounter=1) syntax
- no shortting provided
- the prefered forecast of use is the non-GARCH forecast due to stability concerns
- the degrees of freedom for the copula are low to overestimate tail-dependence given:$$\lambda = 2t_{\nu+1} \left(-\sqrt{\frac{(\nu +1)(1-\rho)}{1+\rho}}\right)$$ in future versions we estimate $\nu$ through MLE (not stable yet)
- the GARCH implementation is not completely stable and may inflate volatility specifically because of the fix to $GARCH(1,1)$ and the added multivariate shocks

# Simulation Specification

## 1. Overview

This simulation models a portfolio of $ k $ risky assets whose prices follow correlated **Geometric Brownian Motions (GBMs)**.  
Unlike the standard Gaussian correlation structure, dependence across assets is introduced through a **t-copula**, which allows for tail dependence (joint extreme events).  
The simulation evolves prices in discrete time with time step $ \Delta t $ and a finite time horizon $ T $.

---

## 2. Model Setup and Parameters

Let:
\[
\mu = (\mu_1, \dots, \mu_k)^\top, \quad 
\sigma = (\sigma_1, \dots, \sigma_k)^\top, \quad
R \in \mathbb{R}^{k\times k}
\]
where:
- $ \mu_i $ is the annual drift (expected return) of asset $ i $,
- $ \sigma_i $ is its annual volatility,
- $ R $ is the correlation matrix between assets.

Other parameters:
- $ S_0 \in \mathbb{R}^k_+ $: initial prices,  
- $ q \in \mathbb{R}^k_+ $: number of shares held (buy-and-hold),  
- $ \nu > 2 $: degrees of freedom of the t-copula,  
- $ \Delta t = 1/12 $: monthly time step (by default).

The model is simulated over $ n = T / \Delta t $ steps and for $ N $ independent Monte Carlo paths.

---

## 3. Generation of Correlated Shocks via a t-Copula

At each time step, we require a vector of correlated standard normal shocks $ Z_t = (Z_t^1, \dots, Z_t^k) $.  
The code does not generate these directly; instead, it proceeds as follows:

1. **Draw multivariate Student-t samples:**
   $$
   T \sim t_k(\nu, 0, R)
   $$
   Each component $ T_i $ has a univariate $ t_\nu $ distribution, and correlation structure $ R $.

2. **Transform to uniform marginals using the CDF:**
   $$
   U_i = F_{t,\nu}(T_i)
   $$
   where $ F_{t,\nu} $ is the CDF of a univariate Student-t distribution.

3. **Transform to Gaussian marginals via the inverse normal CDF:**
   $$
   Z_i = \Phi^{-1}(U_i)
   $$
   where $ \Phi $ is the standard normal CDF.

This yields Gaussian marginals with a **t-copula** dependence structure.  
Finite $ \nu $ implies non-zero *tail dependence*, meaning that extreme moves tend to occur jointly across assets.

---

## 4. GBM Step Dynamics

Each asset $ i $ follows a discretized GBM process:
$$
\log S_{t+\Delta t}^i = \log S_t^i 
+ \left( \mu_i - \frac{1}{2}\sigma_i^2 \right)\Delta t 
+ \sigma_i \sqrt{\Delta t} \, Z_t^i
$$

Define:
$$
\mu_{\text{step}} = (\mu - 0.5 \, \sigma^2)\Delta t, \quad
\sigma_{\text{step}} = \sigma \sqrt{\Delta t}
$$

Then, the vectorized step for all assets is:
$$
r_t = \mu_{\text{step}} + \sigma_{\text{step}} \odot Z_t
$$
and cumulative log-prices:
$$
\log S_t = \log S_0 + \sum_{j=1}^t r_j
$$
so that:
$$
S_t = \exp(\log S_t)
$$

---

## 5. Portfolio Valuation

The portfolio value at time $ t $ is:
$$
V_t = \sum_{i=1}^k q_i S_t^i
$$

This corresponds to a buy-and-hold strategy initialized with fixed shares $ q_i $.  
Weights are not used in the portfolio calculation, only the fixed number of shares.

---

## 6. Simulation Algorithm

For $ N $ Monte Carlo paths and $ n $ time steps:

1. Compute constants:
   $$
   \mu_{\text{step}}, \; \sigma_{\text{step}}, \; \text{and} \; \sqrt{\Delta t}
   $$
2. For each batch of paths:
   - Generate $ b \times n $ t-copula Gaussian shocks $ Z $ (shape $ (b,n,k) $).
   - Compute per-step log-returns $ r $.
   - Accumulate returns to get price paths $ S $.
   - Compute portfolio paths $ V = (S \odot q) \mathbf{1} $.
3. Concatenate all batches into full simulation output.

Complexity per batch:  
- Time $ O(bnk) $  
- Memory $ O(bnk) $

---

## 7. Dependence and Limiting Cases

- As $ \nu \to \infty $, the t-copula converges to the Gaussian copula (no tail dependence).
- As $ \Delta t \to 0 $, the discrete process converges to continuous-time GBM.

Tail dependence coefficient for the bivariate t-copula with correlation $ \rho $ is:
$$
\lambda_U = \lambda_L = 2 \, t_{\nu+1}\!\left(- \sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right)
$$

---

## 8. Statistical Properties

For each asset $ i $:
$$
\mathbb{E}[r_t^i] = (\mu_i - 0.5 \sigma_i^2)\Delta t, \quad
\mathrm{Var}(r_t^i) = \sigma_i^2 \Delta t
$$

For $ i \neq j $:
$$
\mathrm{Cov}(r_t^i, r_t^j) = \sigma_i \sigma_j \Delta t \, \mathrm{Corr}(Z_i, Z_j)
$$
where $ \mathrm{Corr}(Z_i, Z_j) $ depends on the parameters $ (\nu, R_{ij}) $.

---

## 9. Implementation Notes

- Correlation matrix $ R $ is validated via Cholesky decomposition and estimated through *Kendall's $\tau$*.
- $\nu > 2$ ensures finite variance of the t-distribution (there is no MLE fit of $\nu$ because of the lack of storage in the Portofolio objects for multiple assets).
- Small clipping ($ \varepsilon $) avoids infinite values when computing $ \Phi^{-1}(U) $.
- The `multivariate_t` random generator is used for efficient sampling.

---

## 10. Summary Equations

$$
\begin{aligned}
T &\sim t_k(\nu, 0, R) \\
U &= F_{t,\nu}(T) \\
Z &= \Phi^{-1}(\mathrm{clip}(U, \varepsilon, 1-\varepsilon)) \\
r_t &= (\mu - 0.5 \sigma^2)\Delta t + \sigma \sqrt{\Delta t} \odot Z_t \\
S_t &= S_{t-1} \odot \exp(r_t) \\
V_t &= \sum_{i=1}^k q_i S_t^i
\end{aligned}
$$

