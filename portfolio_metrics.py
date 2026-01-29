import numpy as np
import pandas as pd
import yfinance as yf
portfolio_of_tickers = ['AAPL','GOOGL','TSLA','MSFT']
weights = "equal"
value_of_investment = 10000

prices = yf.download(portfolio_of_tickers,
    start="2024-01-01",
    end="2025-01-01"
    )['Close']
def weights_into_df(weights,portfolio_of_tickers):
    n = len(portfolio_of_tickers)

    if weights == "equal": 
        weights = np.full(n,1/n)
    else:
        weights = np.array(weights)
    return pd.DataFrame([weights],columns=portfolio_of_tickers)
df_weight = weights_into_df(weights,portfolio_of_tickers)
def idividual_stock_metrics(prices,portfolio_of_tickers,df_weight,value_of_investment,alpha=0.05):


    log_returns = np.log(prices/prices.shift(1)).dropna()
    weights = df_weight.iloc[0].values

    cumulative_returns = np.exp(np.cumsum(log_returns))
    weighted_values = cumulative_returns * weights * value_of_investment

    final_values = weighted_values.iloc[-1]
    VaR = np.percentile(-log_returns,(1-alpha)*100,axis=0)
    expected_value = log_returns.mean(axis=0)
    volatility = log_returns.std(axis=0)
    
    return pd.DataFrame(
        {
        f"VaR_{int((1-alpha)*100)}": VaR,
        "Final_values_of_investment": final_values,
        "Expected_value (%)": expected_value * 100,
        "Volatility (%)":volatility * 100 
        },
        index=portfolio_of_tickers
    )
idividual_stock_metrics(prices,portfolio_of_tickers,df_weight,value_of_investment)
def portfolio_metrics(prices,portfolio_of_tickers,df_weight,value_of_investment,alpha=0.05):


    returns = np.log(prices/prices.shift(1)).dropna()
    w = df_weight.iloc[0].values.reshape(-1, 1)


    portfolio_returns = returns @ w
    portfolio_returns = portfolio_returns.squeeze()
    expected_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()
      
    VaR = np.percentile(portfolio_returns,(1-alpha)*100)
    
    losses = -portfolio_returns
    CVaR = losses[losses >= VaR].mean()

    return pd.DataFrame(
        {
        "Expected Return  (%)": expected_return * 100,
        "Volatility (%)": volatility * 100,
        f"VaR_{int((1-alpha)*100)}": VaR * 100,
        f"CVaR_{int((1-alpha)*100)}": CVaR * 100
        },
        index=["Portfolio"]
    )
print(portfolio_metrics(prices,portfolio_of_tickers,df_weight,value_of_investment,alpha=0.05))
