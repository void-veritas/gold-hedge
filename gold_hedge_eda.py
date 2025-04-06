#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gold Miners Hedging Strategy - Exploratory Data Analysis
--------------------------------------------------------
This script performs comprehensive exploratory data analysis on GDX and GLD
to understand their relationships, correlations, and characteristics for
developing an effective hedging strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
from datetime import datetime
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import norm
from pathlib import Path


# Keep all your analysis functions in this module:
def create_charts_directory():
    charts_dir = Path('charts')
    if not charts_dir.exists():
        charts_dir.mkdir()
    return charts_dir


def fetch_data_fmp(api_key, start_date='2010-01-01', end_date=None, tickers=['GDX', 'GLD', 'GDXJ', 'SIL', 'SPY'],
                   use_cache=True):
    """
    Fetch historical price data from Financial Modeling Prep API.

    Parameters:
    -----------
    api_key : str
        FMP API key
    start_date : str
        Start date for data retrieval (default: '2010-01-01')
    end_date : str
        End date for data retrieval (default: today)
    tickers : list
        List of tickers to analyze (default: ['GDX', 'GLD', 'GDXJ', 'SIL', 'SPY'])

    Returns:
    --------
    Dictionary of DataFrames with price and return data
    """
    print(f"Fetching data from {start_date} to {end_date} using Financial Modeling Prep API...")

    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    # Create cache file path
    cache_dir = Path('cache')
    if not cache_dir.exists():
        cache_dir.mkdir()

    cache_file = cache_dir / f"data_{start_date}_to_{end_date}.pkl"

    # Check if cache exists and is not expired
    if use_cache and cache_file.exists():
        # Check if cache file is recent (e.g. within 24 hours if end_date is today)
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if end_date != datetime.today().strftime('%Y-%m-%d') or cache_age.total_seconds() < 86400:
            print(f"Loading data from cache: {cache_file}")
            return pd.read_pickle(cache_file)

    # If no cache or cache expired, fetch from API
    print(f"Fetching data from {start_date} to {end_date} using Financial Modeling Prep API...")

    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    data = {}
    price_data = {}

    # Download data for each ticker
    for ticker in tickers:
        try:
            # Construct the API URL for historical daily prices
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date}&to={end_date}&apikey={api_key}"

            # Make the API request
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Error fetching data for {ticker}: {response.status_code}")
                continue

            # Parse the JSON response
            json_data = response.json()

            if 'historical' not in json_data:
                print(f"No historical data available for {ticker}")
                continue

            # Convert to DataFrame
            df = pd.DataFrame(json_data['historical'])

            # Convert date string to datetime
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            # Sort by date (ascending)
            df.sort_index(inplace=True)

            # Rename columns to match yfinance format
            df.rename(columns={
                'close': 'Close',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'volume': 'Volume',
                'adjClose': 'Adj Close'
            }, inplace=True)

            # If 'adjClose' is not available, use 'close'
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']

            price_data[ticker] = df
            print(f"Downloaded {len(df)} rows for {ticker}")

        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")

    # Combine into a price DataFrame
    if not all(ticker in price_data for ticker in ['GDX', 'GLD']):
        raise ValueError("Could not fetch data for required tickers GDX and GLD")

    prices = pd.DataFrame({ticker: df['Adj Close'] for ticker, df in price_data.items() if ticker in price_data})
    volumes = pd.DataFrame({ticker: df['Volume'] for ticker, df in price_data.items() if ticker in price_data})

    # Calculate returns
    returns = prices.pct_change().dropna()

    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Calculate rolling statistics
    rolling_correlation = returns['GDX'].rolling(window=60).corr(returns['GLD'])
    rolling_beta = (
            returns['GDX'].rolling(window=60).cov(returns['GLD']) /
            returns['GLD'].rolling(window=60).var()
    )

    # Store data in dictionary
    data['prices'] = prices
    data['volumes'] = volumes
    data['returns'] = returns
    data['log_returns'] = log_returns
    data['rolling_correlation'] = rolling_correlation
    data['rolling_beta'] = rolling_beta

    # Add basic volatility measures for available tickers
    for ticker in prices.columns:
        # 21-day rolling volatility (annualized)
        data[f'{ticker}_vol_21d'] = returns[ticker].rolling(window=21).std() * np.sqrt(252)
        # 63-day rolling volatility (annualized)
        data[f'{ticker}_vol_63d'] = returns[ticker].rolling(window=63).std() * np.sqrt(252)
        # 252-day rolling volatility (annualized)
        data[f'{ticker}_vol_252d'] = returns[ticker].rolling(window=252).std() * np.sqrt(252)

    # Save to cache before returning
    if use_cache:
        pd.to_pickle(data, cache_file)
        print(f"Data saved to cache: {cache_file}")

    return data



def analyze_price_relationship(data, charts_dir):
    """
    Analyze the price relationship between GDX and GLD.

    Parameters:
    -----------
    data : dict
        Dictionary containing price and return data
    charts_dir : Path
        Directory to save chart outputs
    """
    prices = data['prices']
    returns = data['returns']

    # Create a figure with GDX on secondary axis
    fig, ax1 = plt.subplots()

    # Plot GLD on primary axis
    ax1.set_xlabel('Date')
    ax1.set_ylabel('GLD Price ($)', color='orange')
    ax1.plot(prices.index, prices['GLD'], color='orange', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='orange')

    # Create a secondary y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('GDX Price ($)', color='green')
    ax2.plot(prices.index, prices['GDX'], color='green', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='green')

    # Add title and grid
    plt.title('GDX and GLD Price Series')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(charts_dir / 'price_relationship.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot price ratio
    plt.figure()
    ratio = prices['GDX'] / prices['GLD']
    plt.plot(prices.index, ratio, color='purple', linewidth=2)
    plt.axhline(y=ratio.mean(), color='r', linestyle='--', label=f'Average: {ratio.mean():.2f}')
    plt.title('GDX/GLD Price Ratio')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True)
    plt.savefig(charts_dir / 'price_ratio.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate and print correlation
    correlation = returns['GDX'].corr(returns['GLD'])
    print(f"Correlation between GDX and GLD returns: {correlation:.4f}")

    # Show p-value
    r, p = stats.pearsonr(returns['GDX'].dropna(), returns['GLD'].dropna())
    print(f"Pearson correlation: r={r:.4f}, p={p:.6f}")

    # Calculate rolling correlations
    rolling_corr = returns['GDX'].rolling(window=252).corr(returns['GLD'])

    plt.figure()
    plt.plot(rolling_corr.index, rolling_corr, linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Correlation')
    plt.title('252-Day Rolling Correlation between GDX and GLD Returns')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True)
    plt.savefig(charts_dir / 'rolling_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate correlation across different timeframes
    timeframes = [21, 63, 126, 252]
    corr_values = []

    for tf in timeframes:
        if len(returns) < tf:
            print(f"Not enough data for {tf}-day correlation")
            corr_values.append(np.nan)
            continue
        corr = returns['GDX'].tail(tf).corr(returns['GLD'].tail(tf))
        corr_values.append(corr)
        print(f"Correlation over last {tf} trading days: {corr:.4f}")

    # Plot correlation by timeframe
    plt.figure()
    plt.bar([f"{t} days" for t in timeframes], corr_values, color=sns.color_palette("viridis", len(timeframes)))
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('GDX-GLD Return Correlation Across Timeframes')
    plt.ylabel('Correlation')
    plt.ylim(-1, 1)
    plt.grid(axis='y')
    plt.savefig(charts_dir / 'correlation_by_timeframe.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_beta_relationship(data, charts_dir):
    """
    Analyze the beta relationship between GDX and GLD.

    Parameters:
    -----------
    data : dict
        Dictionary containing price and return data
    charts_dir : Path
        Directory to save chart outputs
    """
    returns = data['returns']

    # Calculate beta using regression
    X = add_constant(returns['GLD'])
    y = returns['GDX']

    model = OLS(y, X).fit()
    beta = model.params[1]

    print(f"\nBeta of GDX to GLD (full period): {beta:.4f}")
    print(f"R-squared: {model.rsquared:.4f}")
    print("Summary Statistics:")
    print(model.summary().tables[1])

    # Plot scatterplot with regression line
    plt.figure()
    plt.scatter(returns['GLD'], returns['GDX'], alpha=0.5)

    # Add regression line
    x_range = np.linspace(returns['GLD'].min(), returns['GLD'].max(), 100)
    plt.plot(x_range, model.params[0] + model.params[1] * x_range, 'r', linewidth=2)

    plt.title(f'GDX vs GLD Return Scatter Plot\nBeta = {beta:.4f}, R² = {model.rsquared:.4f}')
    plt.xlabel('GLD Return')
    plt.ylabel('GDX Return')
    plt.grid(True)
    plt.savefig(charts_dir / 'beta_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate rolling beta
    rolling_beta = (
            returns['GDX'].rolling(window=252).cov(returns['GLD']) /
            returns['GLD'].rolling(window=252).var()
    )

    plt.figure()
    plt.plot(rolling_beta.index, rolling_beta)
    plt.axhline(y=beta, color='r', linestyle='--', label=f'Full Period Beta: {beta:.2f}')
    plt.title('252-Day Rolling Beta of GDX to GLD')
    plt.ylabel('Beta')
    plt.legend()
    plt.grid(True)
    plt.savefig(charts_dir / 'rolling_beta.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate beta across different timeframes
    timeframes = [21, 63, 126, 252]
    beta_values = []

    for tf in timeframes:
        X_sub = add_constant(returns['GLD'].tail(tf))
        y_sub = returns['GDX'].tail(tf)
        model_sub = OLS(y_sub, X_sub).fit()
        beta_sub = model_sub.params[1]
        beta_values.append(beta_sub)
        print(f"Beta over last {tf} trading days: {beta_sub:.4f} (R² = {model_sub.rsquared:.4f})")

    # Plot beta by timeframe
    plt.figure()
    plt.bar([f"{t} days" for t in timeframes], beta_values, color=sns.color_palette("viridis", len(timeframes)))
    plt.axhline(y=beta, color='r', linestyle='--', label=f'Full Period: {beta:.2f}')
    plt.title('GDX-GLD Beta Across Timeframes')
    plt.ylabel('Beta')
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(charts_dir / 'beta_by_timeframe.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_volatility(data, charts_dir):
    """
    Analyze the volatility patterns of GDX and GLD.

    Parameters:
    -----------
    data : dict
        Dictionary containing price and return data
    charts_dir : Path
        Directory to save chart outputs
    """
    returns = data['returns']

    # Calculate annualized volatility
    gdx_vol = returns['GDX'].std() * np.sqrt(252)
    gld_vol = returns['GLD'].std() * np.sqrt(252)

    print(f"\nAnnualized Volatility:")
    print(f"GDX: {gdx_vol:.4f} ({gdx_vol * 100:.2f}%)")
    print(f"GLD: {gld_vol:.4f} ({gld_vol * 100:.2f}%)")
    print(f"Ratio (GDX/GLD): {gdx_vol / gld_vol:.2f}x")

    # Plot rolling volatility
    plt.figure()
    plt.plot(data['GDX_vol_63d'].index, data['GDX_vol_63d'], 'g-', label='GDX', linewidth=2)
    plt.plot(data['GLD_vol_63d'].index, data['GLD_vol_63d'], 'orange', label='GLD', linewidth=2)
    plt.title('63-Day Rolling Volatility (Annualized)')
    plt.ylabel('Annualized Volatility')
    plt.legend()
    plt.grid(True)
    plt.savefig(charts_dir / 'rolling_volatility.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot volatility ratio
    vol_ratio = data['GDX_vol_63d'] / data['GLD_vol_63d']

    plt.figure()
    plt.plot(vol_ratio.index, vol_ratio, 'purple', linewidth=2)
    plt.axhline(y=vol_ratio.mean(), color='r', linestyle='--', label=f'Average: {vol_ratio.mean():.2f}x')
    plt.title('GDX/GLD Volatility Ratio (63-Day Window)')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True)
    plt.savefig(charts_dir / 'volatility_ratio.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Volatility regimes analysis
    plt.figure(figsize=(14, 8))

    # Clean and prepare data - remove NaN and outliers
    gdx_vol = data['GDX_vol_63d'].dropna()
    gld_vol = data['GLD_vol_63d'].dropna()

    # Filter for matching indexes
    common_idx = gdx_vol.index.intersection(gld_vol.index)
    gdx_vol = gdx_vol.loc[common_idx]
    gld_vol = gld_vol.loc[common_idx]

    # Create a DataFrame for the scatter plot
    volatility_df = pd.DataFrame({
        'GLD_Volatility': gld_vol,
        'GDX_Volatility': gdx_vol
    })

    # Remove outliers using quantiles to improve regression
    vol_df_clean = volatility_df[
        (volatility_df['GDX_Volatility'] < volatility_df['GDX_Volatility'].quantile(0.99)) &
        (volatility_df['GLD_Volatility'] < volatility_df['GLD_Volatility'].quantile(0.99))
        ]

    # Create scatter plot
    plt.scatter(vol_df_clean['GLD_Volatility'], vol_df_clean['GDX_Volatility'],
                alpha=0.5, color='#66c2a5', s=30)

    # Add regression line using robust regression to handle remaining outliers
    from sklearn.linear_model import TheilSenRegressor
    X = vol_df_clean['GLD_Volatility'].values.reshape(-1, 1)
    y = vol_df_clean['GDX_Volatility'].values

    # Fit Theil-Sen regression (robust to outliers)
    model = TheilSenRegressor().fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Create regression line
    x_values = np.linspace(vol_df_clean['GLD_Volatility'].min(), vol_df_clean['GLD_Volatility'].max(), 100)
    y_values = intercept + slope * x_values

    # Plot regression line
    plt.plot(x_values, y_values, 'r', linewidth=2,
             label=f'Slope: {slope:.2f}, Intercept: {intercept:.2f}')

    # Add guideline for the average volatility ratio
    avg_ratio = vol_df_clean['GDX_Volatility'].mean() / vol_df_clean['GLD_Volatility'].mean()
    plt.plot(x_values, x_values * avg_ratio, 'b--', linewidth=1.5,
             label=f'Average Ratio: {avg_ratio:.2f}x')

    plt.title('GDX vs GLD Volatility Regimes', fontsize=16)
    plt.xlabel('GLD Volatility', fontsize=14)
    plt.ylabel('GDX Volatility', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(charts_dir / 'volatility_regimes.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate correlation between volatilities
    vol_corr = vol_df_clean['GDX_Volatility'].corr(vol_df_clean['GLD_Volatility'])
    print(f"Correlation between GDX and GLD volatilities: {vol_corr:.4f}")
    print(f"Average GDX/GLD volatility ratio: {avg_ratio:.2f}x")
    print(f"Volatility relationship: GDX_Vol = {intercept:.3f} + {slope:.3f} * GLD_Vol")


def analyze_drawdowns(data, charts_dir):
    """
    Analyze drawdowns and recovery patterns for GDX and GLD.

    Parameters:
    -----------
    data : dict
        Dictionary containing price and return data
    charts_dir : Path
        Directory to save chart outputs
    """
    prices = data['prices']

    # Calculate drawdowns
    gdx_roll_max = prices['GDX'].cummax()
    gld_roll_max = prices['GLD'].cummax()

    gdx_drawdown = (prices['GDX'] / gdx_roll_max - 1) * 100
    gld_drawdown = (prices['GLD'] / gld_roll_max - 1) * 100

    print("\nDrawdown Analysis:")
    print(f"GDX Maximum Drawdown: {gdx_drawdown.min():.2f}%")
    print(f"GLD Maximum Drawdown: {gld_drawdown.min():.2f}%")

    # Plot drawdowns
    plt.figure()
    plt.plot(gdx_drawdown.index, gdx_drawdown, 'g-', label='GDX', linewidth=2)
    plt.plot(gld_drawdown.index, gld_drawdown, 'orange', label='GLD', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.title('Historical Drawdowns')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(charts_dir / 'drawdowns.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Identify major drawdown periods
    threshold = -15  # Define major drawdown as 15% or more

    gdx_major_dd = gdx_drawdown[gdx_drawdown <= threshold]
    gld_major_dd = gld_drawdown[gld_drawdown <= threshold]

    # Print major drawdown periods
    print("\nMajor Drawdown Periods for GDX:")
    for i, (date, value) in enumerate(gdx_major_dd.items()):
        print(f"{i + 1}. {date.strftime('%Y-%m-%d')}: {value:.2f}%")

    print("\nMajor Drawdown Periods for GLD:")
    for i, (date, value) in enumerate(gld_major_dd.items()):
        print(f"{i + 1}. {date.strftime('%Y-%m-%d')}: {value:.2f}%")

    # Calculate drawdown coincidence
    both_drawdown = ((gdx_drawdown <= threshold) & (gld_drawdown <= threshold))
    gdx_only = ((gdx_drawdown <= threshold) & (gld_drawdown > threshold))
    gld_only = ((gdx_drawdown > threshold) & (gld_drawdown <= threshold))

    print(f"\nDrawdown Coincidence Analysis (>={abs(threshold)}% drawdown):")
    print(f"Both GDX and GLD in drawdown: {both_drawdown.sum()} days")
    print(f"Only GDX in drawdown: {gdx_only.sum()} days")
    print(f"Only GLD in drawdown: {gld_only.sum()} days")

    # Create drawdown histogram
    plt.figure(figsize=(14, 8))
    plt.hist(gdx_drawdown, bins=50, alpha=0.5, label='GDX', color='green')
    plt.hist(gld_drawdown, bins=50, alpha=0.5, label='GLD', color='orange')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}%')
    plt.title('Drawdown Distribution')
    plt.xlabel('Drawdown (%)')
    plt.ylabel('Frequency (Days)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(charts_dir / 'drawdown_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_option_implications(data, charts_dir):
    """
    Analyze implications for option hedging strategies.
    Parameters:
    -----------
    data : dict
        Dictionary containing price and return data
    charts_dir : Path
        Directory to save chart outputs
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    # Import the option pricing function
    from option_pricing import calculate_put_option_price

    returns = data['returns']
    # Calculate implied volatility proxy (realized volatility)
    gdx_vol_252d = data['GDX_vol_252d']
    gld_vol_252d = data['GLD_vol_252d']

    # Create a dataframe of indicator variables for different GLD return scenarios
    gld_scenarios = pd.DataFrame(index=returns.index)
    gld_scenarios['Large_Down'] = returns['GLD'] < -0.02  # GLD down >2%
    gld_scenarios['Moderate_Down'] = (returns['GLD'] < -0.01) & (returns['GLD'] >= -0.02)  # GLD down 1-2%
    gld_scenarios['Small_Down'] = (returns['GLD'] < 0) & (returns['GLD'] >= -0.01)  # GLD down 0-1%
    gld_scenarios['Small_Up'] = (returns['GLD'] > 0) & (returns['GLD'] <= 0.01)  # GLD up 0-1%
    gld_scenarios['Moderate_Up'] = (returns['GLD'] > 0.01) & (returns['GLD'] <= 0.02)  # GLD up 1-2%
    gld_scenarios['Large_Up'] = returns['GLD'] > 0.02  # GLD up >2%

    # Calculate average GDX return for each scenario
    scenario_results = {}
    for scenario in gld_scenarios.columns:
        scenario_days = gld_scenarios[scenario]
        avg_gdx_return = returns.loc[scenario_days, 'GDX'].mean()
        count = scenario_days.sum()
        scenario_results[scenario] = {
            'Average GDX Return': avg_gdx_return,
            'Count': count,
            'Frequency': count / len(returns)
        }

    # Print results
    print("\nGDX Returns During Different GLD Scenarios:")
    print("-" * 70)
    for scenario, metrics in scenario_results.items():
        print(f"{scenario:15s}: {metrics['Average GDX Return']:+.4f} " +
              f"({metrics['Count']} days, {metrics['Frequency']:.2%} of time)")
    print("-" * 70)

    # Create a bar chart of average GDX returns in different GLD scenarios
    plt.figure(figsize=(12, 7))
    scenarios = list(scenario_results.keys())
    avg_returns = [scenario_results[s]['Average GDX Return'] for s in scenarios]
    # Color map based on return value
    colors = ['red' if r < 0 else 'green' for r in avg_returns]
    plt.bar(scenarios, avg_returns, color=colors)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Average GDX Returns Under Different GLD Scenarios')
    plt.ylabel('Average GDX Return')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(charts_dir / 'gdx_returns_by_gld_scenario.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate option strike implications
    print("\nOption Strike Price Implications:")
    latest_gld = data['prices']['GLD'].iloc[-1]
    strike_levels = [0.85, 0.90, 0.95]
    strike_results = []

    for level in strike_levels:
        strike = latest_gld * level
        result = {
            'Strike Level': f"{level * 100:.0f}%",
            'Strike Price': strike,
            'Protection %': 0,
            'Avg Protection': 0,
            'Option Cost': 0,
            'Cost %': 0
        }

        # Analyze historical protection
        would_protect = (data['prices']['GLD'] < strike).mean()
        result['Protection %'] = would_protect * 100

        # Calculate average protection when needed
        in_the_money = data['prices']['GLD'] < strike
        if in_the_money.sum() > 0:
            avg_protection = (strike - data['prices']['GLD'][in_the_money]).mean()
            result['Avg Protection'] = avg_protection

        # Use the modularized option pricing function
        vol = gld_vol_252d.iloc[-1]  # Use latest volatility as proxy
        time_to_expiry = 1.0  # 1 year
        put_price_estimate = calculate_put_option_price(
            spot_price=latest_gld,
            strike_price=strike,
            time_to_expiry=time_to_expiry,
            volatility=vol
        )

        result['Option Cost'] = put_price_estimate
        result['Cost %'] = put_price_estimate / latest_gld * 100
        strike_results.append(result)

        print(f"{level * 100:.0f}% OTM Put (Strike ${strike:.2f}):")
        print(f"  Would have provided protection {would_protect:.2%} of the time")
        if in_the_money.sum() > 0:
            print(f"  Average protection when ITM: ${avg_protection:.2f} per share")
        print(f"  Estimated option cost: ${put_price_estimate:.2f} per share " +
              f"({put_price_estimate / latest_gld:.2%} of spot)")
        print()

    # Create a comparison chart for strike levels
    strike_df = pd.DataFrame(strike_results)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    # Plot option cost on left axis
    x = np.arange(len(strike_levels))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, strike_df['Cost %'], width, color='red', alpha=0.7, label='Option Cost (% of spot)')
    ax1.set_ylabel('Option Cost (% of spot)', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    # Plot protection % on right axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, strike_df['Protection %'], width, color='blue', alpha=0.7,
                    label='Historical Protection (%)')
    ax2.set_ylabel('Historical Protection (%)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Add labels and title
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{level * 100:.0f}% OTM\n(${strike_df['Strike Price'][i]:.2f})" for i, level in enumerate(strike_levels)])
    plt.title('Option Strike Level Comparison')

    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(charts_dir / 'option_strike_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_full_analysis(api_key, start_date='2010-01-01', end_date=None):
    """
    Run a complete analysis workflow and return results

    Parameters:
    -----------
    api_key : str
        FMP API key
    start_date : str
        Start date for analysis (default: '2010-01-01')
    end_date : str
        End date for analysis (default: today)

    Returns:
    --------
    dict: Analysis results and data
    """
    # Create charts directory
    charts_dir = create_charts_directory()

    # Fetch data
    data = fetch_data_fmp(api_key, start_date=start_date, end_date=end_date)

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 70)
    for ticker in ['GDX', 'GLD']:
        print(f"{ticker} Price Statistics:")
        print(data['prices'][ticker].describe())
        print(f"\n{ticker} Return Statistics:")
        print(data['returns'][ticker].describe())
        print("-" * 70)

    # Run analyses
    analyze_price_relationship(data, charts_dir)
    analyze_beta_relationship(data, charts_dir)
    analyze_volatility(data, charts_dir)
    analyze_drawdowns(data, charts_dir)
    analyze_option_implications(data, charts_dir)

    print(f"\nEDA complete. Visualizations have been saved to the {charts_dir} directory.")

    return {
        'data': data,
        'charts_dir': charts_dir,
        'status': 'complete'
    }




