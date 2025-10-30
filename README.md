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
- add tickers and build a small portfolio
- pull prices and plot
- show weights and values by asset/class/sector
- run a long-horizon multi-path risk simulation (follows from GBM with t-copula shcoks)
- print tables and charts in the cli

## layout
`controllers/  models/  services/  tests/  utils/  views/  config.py  main.py  requirements.txt`

## notes
- read code, tweak `config.py`, run
