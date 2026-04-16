# BTC-Volatility-Forecasting-using-GARCH-1-1-
## This project focuses on modelling and forcasting bitcoin volatility using GARCH(1,1)

## 🧰 Tech Stack
  - Python 3.13
  - numpy as np numerical operations
  - pandas as pd #For data manipulation
  - Matplotlib.pyplot as plt for plotting
  - Arch for GARCH modeling

## Overview
The script performs the following:
  - Implements rolling window validation for robust out-of-sample testing

## Data collection
  - Source: Yahoo Finance API (yfinance)
  - Period: September 2014 - December 2025 (~2,500 trading days)

## Volatility Calculation
- Realized volatility is calculated as the standard deviation of log returns over specific interval windows, following established financial econometrics practices.
 
## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

## 📄 License

MIT License  
See `LICENSE` file for details.
