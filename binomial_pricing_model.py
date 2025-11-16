import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date

n = 2 #leght of bionmial moderl
time = 1 # in years
starting_date = f"{date.today().year-time}-{date.today().month}-{date.today().day}" # date 1 year before
delta_t = time / n
ticker = "AAPL" # tikcer for stock, use only one at the time
K = 300 # Strike price
type_of_otpion = "call" # or put


def data_cleaning(ticker,date):
    data = yf.download(ticker,start=date)
    log_data = np.log(data['Close']/data['Close'].shift(1)).dropna()
    S0 = float(data['Close'][ticker].iloc[len(log_data[ticker])-1])
    log_std = float(log_data[ticker].std(axis=0))

    return log_std, S0, log_data, data

def variables(log_std, S0, log_data):
    sigma_hat = (np.sqrt(log_std / delta_t))
    up = np.exp(sigma_hat * (delta_t) ** 1/2)
    down  = 1/ up
    e_r = np.exp(.02 * delta_t)
    p_up = (e_r - down)/(up-down)
    q_down = 1 - p_up

    return sigma_hat, up, down, e_r, p_up, q_down

def price_tree(S0, up, down):
    binomial_tree =[]
    binomial_tree.append([S0])
    for i in range(1,n+1): 
        depth = []
        for j in range(i+1):
            price = float(S0 * (up ** j) * (down ** (i-j)))
            depth.append(price)
        binomial_tree.append(depth)

    return binomial_tree

def payoff_tree(binomial_tree, K, option_type="call"):
    terminal_prices = binomial_tree[n]
    payoff = []
    for S in terminal_prices:
        if option_type == "call":
            payoff.append(max(S - K, 0))
        elif option_type == "put":
            payoff.append(max(K - S, 0))
        else:
            raise ValueError("Please choose 'call' or 'put'.")
    
    return payoff

def option_price_tree(payoffs,p_up,q_down,e_r):
    option_price_tree = [ [None] * (i+1) for i in range(n+1) ]
    option_price_tree[n] = payoffs.copy()
    for level in range(n-1, -1, -1):
        for i in range(level+1):
            up_value = option_price_tree[level+1][i+1]
            down_value = option_price_tree[level+1][i]
            option_price_tree[level][i] = float((p_up * up_value + q_down * down_value) / e_r)

    return option_price_tree

log_std, S0, log_data, data  = data_cleaning(ticker,starting_date)
sigma_hat, up, down, e_r, p_up, q_down = variables(log_std, S0, log_data)
binomial_tree = price_tree(S0, up, down)
payoffs = payoff_tree(binomial_tree, K, type_of_otpion)
option_tree = option_price_tree(payoffs,p_up,q_down,e_r)


print(option_tree)
