import yfinance as yf
import pandas as pd

print("yfinance", yf.__version__)
df = yf.download("AAPL", period="5d", interval="1d", auto_adjust=True, progress=False)
print(df.tail())
print("Info has keys:", len(yf.Ticker("AAPL").info))
