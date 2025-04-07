#!/usr/bin/env python3
"""
Gold Miners Hedging Strategy - Core Module
------------------------------------------
This module implements the core functionality for a gold miners ETF (GDX) 
strategy with put option hedging based on rolling beta vs GLD.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from typing import List, Dict, Tuple, Optional, Union
import os
import logging
import requests
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gold_hedge_strategy')

# Set visualization style
plt.style.use('fivethirtyeight')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

class GoldHedgeStrategy:
    """
    Implementation of a gold miners ETF (GDX) strategy with put option hedging
    based on rolling beta vs GLD.
    """
    
    def __init__(
        self,
        start_date='2015-01-01',
        end_date=None,
        beta_window=60,
        vol_window=30,
        risk_free_rate=0.02,
        tx_cost_stock=0.001,
        tx_cost_option=0.002,
        use_cache=True
    ):
        """
        Initialize the Gold Hedge Strategy.
        
        Args:
            start_date (str): Start date for data fetching (YYYY-MM-DD).
            end_date (str, optional): End date for data fetching. Defaults to current date.
            beta_window (int): Window size for rolling beta calculation. Defaults to 60.
            vol_window (int): Window size for rolling volatility calculation. Defaults to 30.
            risk_free_rate (float): Risk-free rate for option pricing. Defaults to 0.02.
            tx_cost_stock (float): Transaction cost for stock trades. Defaults to 0.001.
            tx_cost_option (float): Transaction cost for option trades. Defaults to 0.002.
            use_cache (bool): Whether to use cached data. Defaults to True.
        """
        # Convert string dates to datetime objects
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.to_datetime('today')
        
        self.beta_window = beta_window
        self.vol_window = vol_window
        self.risk_free_rate = risk_free_rate
        self.tx_cost_stock = tx_cost_stock
        self.tx_cost_option = tx_cost_option
        self.use_cache = use_cache
        
        # Initialize data attributes
        self.data = None
        self.prices = None
        self.returns = None
        self.volatility = None
        self.beta = None
        
        # Create cache directory if it doesn't exist
        os.makedirs('cache', exist_ok=True)
        
    def fetch_data(self, api_key=None):
        """
        Fetch historical price data for GDX and GLD.
        
        Args:
            api_key (str, optional): API key for Financial Modeling Prep.
        """
        # Try to load from cache first
        cache_file = os.path.join(
            'cache', 
            f"GDX-GLD_{self.start_date.strftime('%Y-%m-%d')}_{datetime.now().strftime('%Y-%m-%d')}.pkl"
        )
        
        if self.use_cache and os.path.exists(cache_file):
            try:
                logging.info(f"Using cached data from {cache_file}")
                data = pd.read_pickle(cache_file)
                self.data = data
                
                logging.info(f"Data shape: {self.data.shape}")
                logging.info(f"Data range: {min(self.data['date'])} to {max(self.data['date'])}")
                logging.info(f"Data columns: {self.data.columns.tolist()}")
                logging.info(f"Data sample:\n{self.data.head()}")
                
                self._prepare_data()
                return
            except Exception as e:
                logging.warning(f"Error loading cache: {str(e)}")
        
        if api_key is None:
            api_key = os.environ.get('FMP_API_KEY')
        
        if not api_key:
            raise ValueError("API key is required to fetch data. Please provide an API key.")
            
        logging.info(f"Using API key: {api_key[:5]}...{api_key[-3:]}")  # Only show part of key for security
        
        # Create a DataFrame to store the data
        data = pd.DataFrame()
        
        tickers = ['GDX', 'GLD']
        for ticker in tickers:
            try:
                url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={self.start_date.strftime('%Y-%m-%d')}&to={self.end_date.strftime('%Y-%m-%d')}&apikey={api_key}"
                logging.info(f"Requesting data from: {url.replace(api_key, 'API_KEY_HIDDEN')}")
                
                response = requests.get(url)
                
                if response.status_code != 200:
                    logging.error(f"Error fetching data for {ticker}: {response.status_code}")
                    logging.error(f"Response content: {response.text[:200]}...")  # Print first 200 chars of response
                    continue
                    
                ticker_data = response.json()
                
                if 'historical' not in ticker_data:
                    logging.error(f"No historical data found for {ticker}")
                    logging.error(f"Response structure: {list(ticker_data.keys())}")
                    continue
                    
                # Convert to DataFrame
                df = pd.DataFrame(ticker_data['historical'])
                
                # Add ticker column
                df['ticker'] = ticker
                
                # Convert date to datetime
                df['date'] = pd.to_datetime(df['date'])
                
                # Append to main DataFrame
                data = pd.concat([data, df], ignore_index=True)
                
            except Exception as e:
                logging.error(f"Error fetching data for {ticker}: {str(e)}")
        
        if len(data) == 0:
            raise ValueError("No data was fetched. Please check your API key and internet connection.")
            
        logging.info(f"Fetched data for {', '.join(data['ticker'].unique())} from {data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
        
        # Save to cache
        os.makedirs('cache', exist_ok=True)
        data.to_pickle(cache_file)
        logging.info(f"Data saved to cache: {cache_file}")
        
        self.data = data
        self._prepare_data()
    
    def _prepare_data(self):
        """
        Prepare data for analysis by calculating returns, volatility, and beta.
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data available. Please fetch data first.")
        
        # Pivot the data to get prices for each ticker
        prices = self.data.pivot(index='date', columns='ticker', values='adjClose')
        if 'adjClose' not in self.data.columns:
            prices = self.data.pivot(index='date', columns='ticker', values='close')
        
        # Make sure we have all required tickers
        for ticker in ['GDX', 'GLD']:
            if ticker not in prices.columns:
                raise ValueError(f"Ticker {ticker} not found in data")
        
        logging.info(f"Data shape: {prices.shape}")
        logging.info(f"Data range: {prices.index.min()} to {prices.index.max()}")
        
        # Sort by date
        prices.sort_index(inplace=True)
        
        # Store prices
        self.prices = prices
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        self.returns = returns
        
        # Calculate rolling volatility for GDX
        self.volatility = self._calculate_rolling_volatility('GDX')
        
        # Calculate rolling beta of GDX vs GLD
        self.beta = self._calculate_rolling_beta()
        
        logging.info(f"Data prepared successfully.")
    
    def _calculate_rolling_volatility(self, ticker: str) -> pd.Series:
        """
        Calculate rolling volatility for a given ticker.
        
        Args:
            ticker (str): Ticker symbol to calculate volatility for.
        
        Returns:
            pd.Series: Rolling volatility.
        """
        if ticker not in self.returns.columns:
            raise ValueError(f"Ticker {ticker} not found in returns data")
        
        # Calculate annual volatility using rolling window (sqrt of 252 trading days)
        vol = self.returns[ticker].rolling(window=self.vol_window).std() * np.sqrt(252)
        return vol
    
    def _calculate_rolling_beta(self) -> pd.Series:
        """
        Calculate rolling beta of GDX vs GLD.
        
        Returns:
            pd.Series: Rolling beta.
        """
        gdx_returns = self.returns['GDX']
        gld_returns = self.returns['GLD']
        
        if gdx_returns is None or gld_returns is None:
            raise ValueError("Returns data for GDX or GLD is missing")
        
        # Calculate rolling covariance and variance
        rolling_cov = gdx_returns.rolling(window=self.beta_window).cov(gld_returns)
        rolling_var = gld_returns.rolling(window=self.beta_window).var()
        
        # Calculate beta as covariance / variance
        beta = rolling_cov / rolling_var
        
        return beta

    def calculate_put_option_price(
        self, 
        spot_price: float, 
        strike_price: float, 
        time_to_expiry: float, 
        volatility: float,
        risk_free_rate: float = None
    ) -> float:
        """
        Calculate the price of a put option using the Black-Scholes model.

        Parameters:
        -----------
        spot_price : float
            Current price of the underlying asset
        strike_price : float
            Strike price of the option
        time_to_expiry : float
            Time to expiration in years
        volatility : float
            Annualized volatility of the underlying asset
        risk_free_rate : float, optional
            Risk-free rate. If None, use the instance's risk_free_rate. Default is None.

        Returns:
        --------
        float
            The estimated put option price
        """
        # Use the provided risk-free rate or fall back to the instance's risk_free_rate
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        # Calculate d1 and d2 parameters for Black-Scholes
        d1 = (np.log(spot_price / strike_price) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (
                    volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        # Calculate put price using Black-Scholes formula
        put_price = strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)

        return put_price
    
    def calculate_option_payoff(
        self,
        spot_price: float,
        strike_price: float,
        contracts: float = 1.0
    ) -> float:
        """
        Calculate the payoff of a put option at expiration.
        
        Parameters:
        -----------
        spot_price : float
            Price of the underlying asset at expiration
        strike_price : float
            Strike price of the option
        contracts : float, optional
            Number of option contracts (default: 1.0)
            
        Returns:
        --------
        float
            The option payoff (max(strike - spot, 0) * contracts)
        """
        return max(strike_price - spot_price, 0) * contracts 