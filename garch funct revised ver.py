"""
BTC Volatility Forecasting using GARCH(1,1)

Author: Bea Parish
Date(start): 2025-12-21
"""

import numpy as np #For numerical operations
import pandas as pd #For data manipulation
import matplotlib.pyplot as plt #For plotting
from arch import arch_model #For GARCH modeling
import os #For changing working directory

def load_data(file_path):
    df = pd.read_csv(file_path) #Load BTC data
    df["date"] = pd.to_datetime(df["date"]) #Convert date column to datetime (strings to dates we can work with)
    df = df.sort_values("date").dropna().reset_index(drop=True) #Sort by date, drop missing values, and reset index
    #Log returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1)) #Calculate log returns (log of price ratio) and shift by 1 to get returns for each day compared to the previous day
    df = df.dropna().reset_index(drop=True) #Drop the first row which will have NaN log return due to the shift, and reset index again after dropping.
    return df

def scale_returns(returns):
    #Scale returns (important for GARCH)
    returns = returns * 100 #Percentages are easier to interpret and often used in volatility modeling
    return returns

def fit_garch_model(returns):
    #Fit GARCH(1,1)
    model = arch_model(
        returns,
        vol="Garch",
        p=1,
        q=1,
        mean="Zero",
        dist="normal"
    )
    res = model.fit(disp="off")
    return res
    
def calculate_volatility(df, returns, res):
    #Conditional volatility (in-sample)
    df["garch_vol"] = res.conditional_volatility #This gives us the estimated volatility for each day based on the GARCH model, which we can compare to the realized volatility
    #Conditional volatility is the volatility estimate for each day based on the GARCH model, which takes into account past returns and past volatility to provide a dynamic estimate of volatility over time. This allows us to see how the GARCH model captures changes in volatility in response to market conditions

    # Realized volatility (30-day)
    df["realized_vol_30d"] = returns.rolling(30).std() #.rolling(30) creates a rolling window of 30 days, and .std() calculates the standard deviation of returns within that window. This gives us a measure of realized volatility based on the past 30 days of returns
    #.std means standard deviation, which is a common measure of volatility. By rolling over a 30-day window, we get the realized volatility for each day based on the past 30 days of returns
    return df

def plot_volatility(df):
    fig, ax = plt.subplots(figsize=(10, 5)) #Create a figure and axis for plotting with a specified size (10 inches wide and 5 inches tall)
    ax.plot(df["date"], df["realized_vol_30d"], label="Realized Vol (30d)")
    ax.plot(df["date"], df["garch_vol"], label="GARCH Volatility", alpha=0.8)
    ax.set_title("BTC Volatility: GARCH vs Realized")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (%)")
    ax.legend() #Labels the coloured lines to differentiate
    plt.tight_layout() #Adjusts spacing automatically to prevent overlap
    plt.show()

def calculate_mse(df):
    valid = df.dropna() #Drop any rows with NaN values to ensure we are only comparing valid data points where both GARCH volatility and realized volatility are available
    mse = ((valid["garch_vol"] - valid["realized_vol_30d"]) ** 2).mean() #Calculate Mean Squared Error (MSE) between the GARCH volatility and the realized volatility
    print(f"\nMean Squared Error: {mse:.4f}") #{mse:.4f} print MSE to 4dp

def main():
    os.chdir(os.path.dirname(__file__)) #Change working directory to script's directory
    df = load_data("btcusd_d.csv") #Load BTC data
    returns = scale_returns(df["log_return"]) #Scale returns for GARCH modeling
    res = fit_garch_model(returns) #Fit GARCH model and get results
    print(res.summary()) #Print the summary of the GARCH model fit, which includes parameter (alpha, beta, omega) estimates and statistical significance
    df = calculate_volatility(df, returns, res) #Calculate GARCH and realized volatility
    plot_volatility(df) #Plot the volatility estimates
    calculate_mse(df) #Calculate and print MSE between GARCH and realized volatility

if __name__ == "__main__":
    main()



