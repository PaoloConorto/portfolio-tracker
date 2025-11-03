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
- runs a long-horizon multi-path risk simulation (follows from GBM with t-copula shocks, optional GARCH inclusion)
- prints tables and charts in the cli

## layout
`controllers/  models/  services/  tests/  utils/  views/  config.py  main.py  requirements.txt`

## notes
- read code, tweak `config.py`, run `main.py`
- the prefered forecast of use is the non GARCH forecast
- the degrees of freedom for the copula are low to overestimate tail-dependence given:
$$\lambda = 2t_{\nu+1} \left(-\sqrt{\frac{(\nu +1)(1-\rho)}{1+\rho}}\right)$$
in future versions we estimate $\nu$ through MLE (not stable yet)
- the GARCH implementation is not completely stable and may over predict volatility specifically because of the fix to $GARCH(1,1)
