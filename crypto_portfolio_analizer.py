import numpy as np
import pandas as pd
import yfinance as yf
import ccxt

api_keys = {"bybit":{
            "apiKey": "#",
            "secret": "#"}}

exchange_your = "bybit" 
exchange = getattr(ccxt,exchange_your)
exchange = exchange(api_keys[exchange_your])
def get_portfolio_weights(exchange):
    positions = exchange.fetchPositions()

    values = []
    tickers = []
    avr_prices = []
    side_of_trades = []
    leverage = []

    for p in positions:
        size = float(p['info'].get('positionIMByMp', 0))
        price =  float(p['info'].get('avgPrice', 0))
        side = p['info'].get('side', 0)
        symbol = p['symbol']
        lev = float(p['info'].get('leverage', 0))


        if size != 0:
            values.append(abs(size))
            tickers.append(symbol)
            avr_prices.append(price)
            side_of_trades.append(side)
            leverage.append(lev)
    total = sum(values)
    weights = np.array(values) / total if total != 0 else np.zeros(len(values))

    return pd.DataFrame(data=[weights,avr_prices,side_of_trades,leverage], columns=tickers,index=["weights","avr_price","side","leverage"]), total
df, total =  get_portfolio_weights(exchange)
df
def fetch_close_prices(df_weight,timeframe ="1h",number_of_data = 1000):
    close_prices = {}
    tickers = df_weight.columns.to_list()
    tickers.append("BTC/USDT:USDT")
    for ticker in tickers:
        df_prices = pd.DataFrame(exchange.fetchOHLCV(ticker, timeframe=timeframe,limit=number_of_data))
        df_prices.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        df_prices['Time'] = pd.to_datetime(df_prices['Time'], unit='ms')
        df_prices['Close'] = np.log(df_prices['Close'] / df_prices['Close'].shift(1))
        close_prices[ticker] = df_prices['Close'].dropna()
    
    return pd.DataFrame(close_prices)
def calculate_beta(df_log_prices, benchmark_ticker):
    """
    Calculate beta for each asset vs benchmark
    """
    if benchmark_ticker not in df_log_prices.columns:
        return pd.Series(np.nan, index=df_log_prices.columns)
    
    benchmark_returns = df_log_prices[benchmark_ticker]
    betas = {}
    
    for col in df_log_prices.columns:
        if col != benchmark_ticker:
            betas[col] = df_log_prices[col].cov(benchmark_returns) / benchmark_returns.var()
        else:
            betas[col] = 1.0
    
    return pd.Series(betas)
def individual_metrics(df_log_prices, df_weight=None, alpha=0.05, benchmark_ticker="BTC/USDT:USDT", apply_leverage=False):
    """
    Calculate individual asset metrics with optional leverage adjustment and side (long/short) handling
    
    For short positions:
    - Returns are inverted (negative returns become positive)
    - This represents profit/loss from the short position perspective
    """
    df_adjusted = df_log_prices.copy()
    
    # Apply leverage and side adjustments if requested
    if apply_leverage and df_weight is not None:
        leverage_row = df_weight.loc["leverage"] if "leverage" in df_weight.index else None
        side_row = df_weight.loc["side"] if "side" in df_weight.index else None
        
        for ticker in df_log_prices.columns:
            if ticker == benchmark_ticker:
                continue
                
            # Get leverage (default to 1x)
            lev = leverage_row[ticker] if leverage_row is not None and ticker in leverage_row.index else 1.0
            
            # Apply leverage
            df_adjusted[ticker] = df_log_prices[ticker] * lev
            
            # Invert returns for short positions
            if side_row is not None and ticker in side_row.index:
                if side_row[ticker].lower() == 'sell':
                    df_adjusted[ticker] = -df_adjusted[ticker]
    
    # Calculate metrics
    mean = df_adjusted.mean()
    std = df_adjusted.std()
    VaR = -df_adjusted.quantile(alpha)
    ES = -df_adjusted[df_adjusted <= -VaR].mean()
    
    cum_returns = np.exp(df_adjusted.cumsum())
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    
    sharpe_ratio = mean / std if std.any() else 0
    skewness = df_adjusted.skew()
    kurtosis = df_adjusted.kurtosis()
    
    # Calculate beta (on adjusted returns)
    betas = calculate_beta(df_adjusted, benchmark_ticker)
    
    # Build metrics dictionary
    metrics_dict = {
        "Mean (%)": mean * 100,
        "Volatility (%)": std * 100,
        "Sharpe Ratio": sharpe_ratio * np.sqrt(365 * 24),
        "VaR (%)": VaR * 100,
        "Expected Shortfall (%)": ES * 100,
        "Max Drawdown (%)": drawdown.min() * 100,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "Beta": betas if betas is not None else np.nan,
    }
    
    # Add leverage and side rows if available
    if df_weight is not None:
        if "leverage" in df_weight.index:
            metrics_dict["Leverage"] = df_weight.loc["leverage"]
        if "side" in df_weight.index:
            metrics_dict["Position Side"] = df_weight.loc["side"]
    
    metrics_df = pd.DataFrame(metrics_dict).T
    
    return metrics_df

def portfolio_metrics(df_weight, df_log_prices, alpha=0.05, benchmark_ticker="BTC/USDT:USDT", apply_leverage=False):
    """
    Calculate portfolio-level metrics with optional leverage adjustment and side (long/short) handling
    Uses variance-covariance matrix for proper portfolio std calculation
    
    For short positions:
    - Returns are inverted to represent profit/loss from short perspective
    - Weights remain positive (representing allocation size)
    """
    weights = df_weight.loc["weights"]
    leverage_row = df_weight.loc["leverage"] if "leverage" in df_weight.index else None
    side_row = df_weight.loc["side"] if "side" in df_weight.index else None
    
    # Filter to only positions (exclude BTC benchmark if not in portfolio)
    portfolio_tickers = [t for t in weights.index if t in df_log_prices.columns]
    portfolio_weights = weights[portfolio_tickers]
    portfolio_weights = portfolio_weights / portfolio_weights.sum()
    
    # Get returns for portfolio assets
    df_portfolio_prices = df_log_prices[portfolio_tickers].copy()
    
    # Create leverage and side-adjusted returns
    if apply_leverage and leverage_row is not None:
        # Apply leverage and side to returns
        leverages = []
        df_portfolio_prices_adj = df_portfolio_prices.copy()
        
        for ticker in portfolio_tickers:
            # Get leverage (default to 1x)
            lev = leverage_row[ticker] if ticker in leverage_row.index else 1.0
            leverages.append(lev)
            
            # Apply leverage
            df_portfolio_prices_adj[ticker] = df_portfolio_prices[ticker] * lev
            
            # Invert returns for short positions
            if side_row is not None and ticker in side_row.index:
                if side_row[ticker].lower() == 'sell':
                    df_portfolio_prices_adj[ticker] = -df_portfolio_prices_adj[ticker]
        
        leverages = np.array(leverages)
        
        # Calculate covariance matrix of adjusted returns
        cov_matrix = df_portfolio_prices_adj.cov()
        
        # Portfolio variance: w^T * Cov * w
        port_variance = np.dot(portfolio_weights.values, np.dot(cov_matrix.values, portfolio_weights.values))
        port_std = np.sqrt(port_variance)
        
        # Portfolio returns (for other metrics)
        portfolio_returns = (df_portfolio_prices_adj * portfolio_weights.values).sum(axis=1)
        
        # Effective leverage (weighted average)
        portfolio_leverage = np.dot(portfolio_weights.values, leverages)
        
        # Count long vs short positions
        if side_row is not None:
            long_count = sum([1 for t in portfolio_tickers if side_row[t].lower() == 'buy'])
            short_count = sum([1 for t in portfolio_tickers if side_row[t].lower() == 'sell'])
            long_weight = sum([portfolio_weights[t] for t in portfolio_tickers if side_row[t].lower() == 'buy'])
            short_weight = sum([portfolio_weights[t] for t in portfolio_tickers if side_row[t].lower() == 'sell'])
        else:
            long_count = len(portfolio_tickers)
            short_count = 0
            long_weight = 1.0
            short_weight = 0.0
        
    else:
        # Unleveraged case - but still handle short positions if present
        leverages = np.ones(len(portfolio_tickers))
        df_portfolio_prices_adj = df_portfolio_prices.copy()
        
        # Apply side adjustments even without leverage
        if side_row is not None:
            for ticker in portfolio_tickers:
                if ticker in side_row.index and side_row[ticker].lower() == 'sell':
                    df_portfolio_prices_adj[ticker] = -df_portfolio_prices[ticker]
        
        # Calculate covariance matrix
        cov_matrix = df_portfolio_prices_adj.cov()
        
        # Portfolio variance: w^T * Cov * w (Modern Portfolio Theory)
        port_variance = np.dot(portfolio_weights.values, np.dot(cov_matrix.values, portfolio_weights.values))
        port_std = np.sqrt(port_variance)
        
        # Portfolio returns
        portfolio_returns = (df_portfolio_prices_adj * portfolio_weights.values).sum(axis=1)
        
        portfolio_leverage = 1.0
        
        # Count long vs short positions
        if side_row is not None:
            long_count = sum([1 for t in portfolio_tickers if side_row[t].lower() == 'buy'])
            short_count = sum([1 for t in portfolio_tickers if side_row[t].lower() == 'sell'])
            long_weight = sum([portfolio_weights[t] for t in portfolio_tickers if side_row[t].lower() == 'buy'])
            short_weight = sum([portfolio_weights[t] for t in portfolio_tickers if side_row[t].lower() == 'sell'])
        else:
            long_count = len(portfolio_tickers)
            short_count = 0
            long_weight = 1.0
            short_weight = 0.0
    
    # Portfolio metrics
    port_mean = portfolio_returns.mean()
    port_sharpe = (port_mean / port_std * np.sqrt(365 * 24)) if port_std != 0 else 0
    port_VaR = -portfolio_returns.quantile(alpha)
    port_ES = -portfolio_returns[portfolio_returns <= -port_VaR].mean()
    
    cum_returns = (portfolio_returns.cumsum()).apply(lambda x: np.exp(x))
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    port_max_dd = drawdown.min()
    
    # Portfolio beta
    if benchmark_ticker in df_log_prices.columns:
        benchmark_returns = df_log_prices[benchmark_ticker]
        port_beta = portfolio_returns.cov(benchmark_returns) / benchmark_returns.var()
    else:
        port_beta = np.nan
    
    # Diversification ratio: (weighted avg of individual stds) / portfolio std
    individual_stds = df_portfolio_prices_adj.std()
    weighted_avg_std = np.dot(portfolio_weights.values, individual_stds.values)
    
    diversification_ratio = weighted_avg_std / port_std if port_std != 0 else 1.0
    
    return pd.Series({
        "Portfolio Mean (%)": port_mean * 100,
        "Portfolio Std (%) [VCV]": port_std * 100,  # Variance-Covariance based
        "Portfolio Sharpe": port_sharpe,
        "Portfolio VaR (%)": port_VaR * 100,
        "Portfolio ES (%)": port_ES * 100,
        "Portfolio Max DD (%)": port_max_dd * 100,
        "Portfolio Beta": port_beta,
        "Effective Leverage": portfolio_leverage,
        "Number of Positions": len(portfolio_tickers),
        "Long Positions": long_count,
        "Short Positions": short_count,
        "Long Weight (%)": long_weight * 100,
        "Short Weight (%)": short_weight * 100,
    })
df_weight, total = get_portfolio_weights(exchange)
df_log_prices = fetch_close_prices(df_weight)

print("=" * 80)
print("UNLEVERAGED METRICS (Spot-equivalent returns)")
print("=" * 80)

# Individual metrics without leverage
individual_metrics_unleveraged = individual_metrics(df_log_prices, df_weight, alpha=0.05, apply_leverage=False)
print("\nIndividual Asset Metrics (Unleveraged):")
print(individual_metrics_unleveraged)

# Portfolio metrics without leverage
portfolio_metrics_unleveraged = portfolio_metrics(df_weight, df_log_prices, alpha=0.05, apply_leverage=False)
print("\nPortfolio Metrics (Unleveraged):")
print(portfolio_metrics_unleveraged)
print("\n" + "=" * 80)
print("LEVERAGED METRICS (Actual position returns)")
print("=" * 80)

# Individual metrics with leverage
individual_metrics_leveraged = individual_metrics(df_log_prices, df_weight, alpha=0.05, apply_leverage=True)
print("\nIndividual Asset Metrics (Leveraged):")
print(individual_metrics_leveraged)

# Portfolio metrics with leverage
portfolio_metrics_leveraged = portfolio_metrics(df_weight, df_log_prices, alpha=0.05, apply_leverage=True)
print("\nPortfolio Metrics (Leveraged):")
print(portfolio_metrics_leveraged)
print("\n" + "=" * 80)
print("ADDITIONAL ANALYSIS")
print("=" * 80)

# Correlation matrix (always on unleveraged returns)
print("\nCorrelation Matrix (Unleveraged Returns):")
print(df_log_prices.corr().round(3))

# Leverage comparison table
print("\nLeverage Impact Comparison:")
comparison_df = pd.DataFrame({
    'Unleveraged Mean (%)': individual_metrics_unleveraged.loc['Mean (%)'],
    'Leveraged Mean (%)': individual_metrics_leveraged.loc['Mean (%)'],
    'Unleveraged Std (%)': individual_metrics_unleveraged.loc['Volatility (%)'],
    'Leveraged Std (%)': individual_metrics_leveraged.loc['Volatility (%)'],
    'Leverage': df_weight.loc['leverage'] if 'BTC/USDT:USDT' not in df_weight.columns else df_weight.loc['leverage'].drop('BTC/USDT:USDT', errors='ignore')
})
print(comparison_df)


# Portfolio comparison
print("\nPortfolio Leverage Impact:")
portfolio_comparison = pd.DataFrame({
    'Unleveraged': portfolio_metrics_unleveraged,
    'Leveraged': portfolio_metrics_leveraged
}).T
print(portfolio_comparison)
