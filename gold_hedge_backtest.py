#!/usr/bin/env python3
"""
Gold Miners Hedging Strategy - Backtest Module
----------------------------------------------
This module implements a backtest framework for a gold miners ETF (GDX) 
strategy with put option hedging based on rolling beta vs GLD.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
from gold_hedge_strategy import GoldHedgeStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gold_hedge_backtest')

# Set visualization style
plt.style.use('fivethirtyeight')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

class GoldHedgeBacktest:
    """
    Backtest for a gold miners ETF (GDX) strategy with put option hedging
    based on rolling beta vs GLD.
    """
    
    def __init__(
        self,
        strategy: GoldHedgeStrategy,
        initial_capital: float = 100000,
        rebalance_freq: str = 'monthly',
        moneyness: float = 0.95,  # Strike as percentage of spot (e.g., 0.95 = 5% OTM)
        use_dynamic_hedge: bool = True,
        min_hedge_ratio: float = 0.5,
        max_hedge_ratio: float = 1.0,
        option_multiplier: float = 100,  # Standard option contract size
        save_dir: str = 'figures'
    ):
        """
        Initialize the backtest.
        
        Parameters:
        -----------
        strategy : GoldHedgeStrategy
            Initialized strategy object with data loaded
        initial_capital : float
            Initial capital for the backtest (default: $100,000)
        rebalance_freq : str
            Rebalancing frequency: 'monthly', 'quarterly', or 'annually' (default: 'monthly')
        moneyness : float
            Strike price as percentage of spot price (default: 0.95, or 5% OTM)
        use_dynamic_hedge : bool
            Whether to use dynamic hedge sizing based on beta (default: True)
        min_hedge_ratio : float
            Minimum hedge ratio as percentage of portfolio (default: 0.5)
        max_hedge_ratio : float
            Maximum hedge ratio as percentage of portfolio (default: 1.0)
        option_multiplier : float
            Option contract multiplier (default: 100)
        save_dir : str
            Directory to save figures (default: 'figures')
        """
        # Validate strategy
        if not isinstance(strategy, GoldHedgeStrategy):
            raise TypeError("strategy must be a GoldHedgeStrategy object")
        if strategy.prices is None:
            raise ValueError("Strategy must have data loaded. Call strategy.fetch_data() first.")
        
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.moneyness = moneyness
        self.use_dynamic_hedge = use_dynamic_hedge
        self.min_hedge_ratio = min_hedge_ratio
        self.max_hedge_ratio = max_hedge_ratio
        self.option_multiplier = option_multiplier
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize results container
        self.results = None
        
        # Map rebalancing frequency to number of months
        self.rebalance_months = {
            'monthly': 1,
            'quarterly': 3,
            'annually': 12
        }.get(rebalance_freq, 1)
        
        # Check if we have required data
        self._validate_data()
    
    def _validate_data(self):
        """
        Validate that we have all required data.
        """
        required_data = ['prices', 'returns', 'volatility', 'beta']
        for attr in required_data:
            if getattr(self.strategy, attr) is None:
                raise ValueError(f"Strategy missing required data: {attr}")
    
    def _generate_rebalance_dates(self) -> List[pd.Timestamp]:
        """
        Generate rebalance dates based on rebalancing frequency.
        
        Returns:
        --------
        List[pd.Timestamp]
            List of rebalance dates
        """
        # Get all dates from price data
        all_dates = self.strategy.prices.index
        
        if self.rebalance_freq == 'monthly':
            # Group by year and month, take the first day of each month
            date_groups = all_dates.to_period('M').unique()
            rebalance_dates = [all_dates[all_dates.to_period('M') == period][0] for period in date_groups]
        elif self.rebalance_freq == 'quarterly':
            # Group by year and quarter, take the first day of each quarter
            date_groups = all_dates.to_period('Q').unique()
            rebalance_dates = [all_dates[all_dates.to_period('Q') == period][0] for period in date_groups]
        elif self.rebalance_freq == 'annually':
            # Group by year, take the first day of each year
            date_groups = all_dates.to_period('Y').unique()
            rebalance_dates = [all_dates[all_dates.to_period('Y') == period][0] for period in date_groups]
        else:
            raise ValueError(f"Invalid rebalance frequency: {self.rebalance_freq}")
        
        return rebalance_dates
    
    def run_backtest(self):
        """Run the backtest simulation."""
        # Initialize results DataFrame
        all_dates = self.strategy.prices.index
        results = pd.DataFrame(index=all_dates)
        results['GDX_Price'] = self.strategy.prices['GDX']
        results['GLD_Price'] = self.strategy.prices['GLD']
        results['Portfolio_Value'] = np.nan
        results['GDX_Shares'] = 0
        results['Put_Contracts'] = 0
        results['Strike_Price'] = np.nan
        results['Maturity'] = pd.NaT
        results['Put_Inflow'] = 0.0
        results['Option_Cost'] = 0.0
        results['Hedge_Ratio'] = 0.0
        
        # Initialize portfolio
        gdx_price = results.loc[all_dates[0], 'GDX_Price']
        gld_price = results.loc[all_dates[0], 'GLD_Price']
        
        # Calculate initial GDX position (accounting for transaction costs and initial option purchase)
        initial_option_allocation = 0.02  # Reserve 2% for initial option purchase
        available_for_stock = self.initial_capital * (1 - initial_option_allocation)
        gdx_shares = int(available_for_stock / (gdx_price * (1 + self.strategy.tx_cost_stock)))
        cash = self.initial_capital - (gdx_shares * gdx_price * (1 + self.strategy.tx_cost_stock))
        
        # Calculate initial hedge position
        beta = self.strategy.beta.iloc[0] if not self.strategy.beta.empty else 1.0
        if pd.isna(beta) or beta <= 0:
            beta = 1.0
        
        volatility = self.strategy.volatility.iloc[0] if not self.strategy.volatility.empty else 0.20
        if pd.isna(volatility) or volatility <= 0:
            volatility = 0.20
        
        # Calculate initial option hedge
        if self.use_dynamic_hedge:
            hedge_ratio = min(max(beta, self.min_hedge_ratio), self.max_hedge_ratio)
        else:
            hedge_ratio = 1.0
        
        notional_to_hedge = gdx_shares * gdx_price * hedge_ratio
        strike_price = gld_price * self.moneyness  # Use GLD price for strike since we're buying GLD puts
        
        # Calculate time to next rebalance for initial option pricing
        rebalance_dates = self._generate_rebalance_dates()
        if len(rebalance_dates) > 1:
            next_rebalance = rebalance_dates[1]
            time_to_next_rebalance = (next_rebalance - all_dates[0]).days / 365
        else:
            time_to_next_rebalance = 30 / 365
        
        # Calculate initial option position
        option_price = self.strategy.calculate_put_option_price(
            spot_price=gld_price,  # Use GLD price as spot since we're buying GLD puts
            strike_price=strike_price,
            time_to_expiry=time_to_next_rebalance,
            volatility=volatility,
            risk_free_rate=self.strategy.risk_free_rate
        )
        
        put_contracts = int(notional_to_hedge / (100 * gld_price))  # Use GLD price for contract sizing
        
        # Update cash for option purchase
        option_cost = put_contracts * option_price * (1 + self.strategy.tx_cost_option)
        cash -= option_cost
        
        # Set initial maturity
        maturity = next_rebalance if len(rebalance_dates) > 1 else (all_dates[0] + pd.DateOffset(months=1))
        
        # Log initial portfolio state
        logging.info(f"\nInitial Portfolio Setup:")
        logging.info(f"GDX Shares: {gdx_shares}, Put Contracts: {put_contracts}")
        logging.info(f"Cash: ${cash:.2f}, Strike Price: ${strike_price:.2f}")
        logging.info(f"Option Cost: ${option_cost:.2f}, Hedge Ratio: {hedge_ratio:.2f}")
        
        for date in results.index:
            gdx_price = results.loc[date, 'GDX_Price']
            
            # Log portfolio state at start of day
            logging.info(f"\nDate: {date}")
            logging.info(f"Start of day - Cash: ${cash:.2f}, GDX Shares: {gdx_shares}, Put Contracts: {put_contracts}")
            
            # Handle option expiration if applicable
            if maturity is not None and date >= maturity:
                if put_contracts > 0:
                    # Calculate option payoff using GLD price
                    gld_price = results.loc[date, 'GLD_Price']
                    payoff = max(strike_price - gld_price, 0) * put_contracts * 100  # Standard option multiplier
                    logging.info(f"Option expiration details - Strike: ${strike_price:.2f}, GLD Price: ${gld_price:.2f}, Contracts: {put_contracts}")
                    cash += payoff
                    results.loc[date, 'Put_Inflow'] = payoff  # Record the payoff
                    logging.info(f"Option expired - Payoff: ${payoff:.2f}")
                
                put_contracts = 0
                strike_price = None
                maturity = None
            
            # Calculate current portfolio value
            stock_value = gdx_shares * gdx_price
            option_value = 0
            if put_contracts > 0 and strike_price is not None and maturity is not None:
                days_to_expiry = (maturity - date).days
                time_to_expiry = max(days_to_expiry / 365, 0.01)  # At least 0.01 years (about 3-4 days)
                gld_price = results.loc[date, 'GLD_Price']
                volatility = self.strategy.volatility.loc[date] if date in self.strategy.volatility.index else 0.20
                option_value = self.strategy.calculate_put_option_price(
                    spot_price=gld_price,  # Use GLD price for option valuation
                    strike_price=strike_price,
                    time_to_expiry=time_to_expiry,
                    volatility=volatility,
                    risk_free_rate=self.strategy.risk_free_rate
                ) * put_contracts * self.option_multiplier
                results.loc[date, 'Option_Cost'] = option_value  # Record the option value
                logging.info(f"Option value on {date}: ${option_value:.2f} (Strike: ${strike_price:.2f}, GLD: ${gld_price:.2f}, Days to expiry: {days_to_expiry})")
            
            portfolio_value = cash + stock_value + option_value
            logging.info(f"Portfolio composition - Stock: ${stock_value:.2f}, Options: ${option_value:.2f}, Cash: ${cash:.2f}, Total: ${portfolio_value:.2f}")
            
            # Rebalance if needed
            if date in rebalance_dates:
                logging.info(f"Rebalancing on {date}")
                
                # Get beta and volatility
                beta = self.strategy.beta.loc[date] if date in self.strategy.beta.index else 1.0
                if pd.isna(beta):
                    beta = 1.0
                    logging.warning(f"Using default beta ({beta:.1f}) on {date} due to missing value.")
                
                volatility = self.strategy.volatility.loc[date] if date in self.strategy.volatility.index else 0.20
                if pd.isna(volatility) or volatility <= 0:
                    volatility = 0.20
                    logging.warning(f"Using default volatility ({volatility*100:.0f}%) on {date} due to missing or invalid value.")
                
                # Calculate target position
                target_stock_exposure = portfolio_value * 0.98  # Reserve 2% for options
                target_gdx_shares = int(target_stock_exposure / gdx_price) if not pd.isna(gdx_price) and gdx_price > 0 else 0
                
                # Calculate option hedge
                if self.use_dynamic_hedge:
                    hedge_ratio = min(max(beta, self.min_hedge_ratio), self.max_hedge_ratio)
                else:
                    hedge_ratio = 1.0
                
                notional_to_hedge = target_gdx_shares * gdx_price * hedge_ratio
                gld_price = results.loc[date, 'GLD_Price']
                strike_price = gld_price * self.moneyness  # Use GLD price for strike since we're buying GLD puts
                
                # Calculate time to next rebalance
                next_rebalance_idx = rebalance_dates.index(date) + 1
                if next_rebalance_idx < len(rebalance_dates):
                    next_rebalance = rebalance_dates[next_rebalance_idx]
                    time_to_next_rebalance = (next_rebalance - date).days / 365
                else:
                    time_to_next_rebalance = 30 / 365  # Default to 30 days if last rebalance
                
                # Calculate option contracts needed
                option_price = self.strategy.calculate_put_option_price(
                    spot_price=gld_price,  # Use GLD price as spot since we're buying GLD puts
                    strike_price=strike_price,
                    time_to_expiry=time_to_next_rebalance,
                    volatility=volatility,
                    risk_free_rate=self.strategy.risk_free_rate
                ) if strike_price > 0 else 0
                
                if option_price > 0 and notional_to_hedge > 0:
                    target_put_contracts = int(notional_to_hedge / (100 * gld_price))  # Use GLD price for contract sizing
                else:
                    target_put_contracts = 0
                    if option_price <= 0:
                        logging.warning(f"Invalid option price (${option_price:.2f}) - skipping hedge")
                
                # Calculate transaction costs
                stock_tx_cost = abs(target_gdx_shares - gdx_shares) * gdx_price * self.strategy.tx_cost_stock
                option_tx_cost = abs(target_put_contracts - put_contracts) * option_price * self.strategy.tx_cost_option
                total_tx_cost = stock_tx_cost + option_tx_cost
                
                # Log transaction details
                logging.info(f"Transaction costs - Stock: ${stock_tx_cost:.2f}, Option: ${option_tx_cost:.2f}")
                logging.info(f"New positions - GDX Shares: {target_gdx_shares}, Put Contracts: {target_put_contracts}")
                
                # Update positions
                cash -= (target_gdx_shares - gdx_shares) * gdx_price  # Stock purchase/sale
                cash -= (target_put_contracts - put_contracts) * option_price  # Option purchase/sale
                cash -= total_tx_cost  # Transaction costs
                
                gdx_shares = target_gdx_shares
                put_contracts = target_put_contracts
                if put_contracts > 0:
                    next_rebalance_idx = rebalance_dates.index(date) + 1
                    if next_rebalance_idx < len(rebalance_dates):
                        maturity = rebalance_dates[next_rebalance_idx]
                    else:
                        maturity = date + pd.DateOffset(months=1)
                
                logging.info(f"End of rebalance - Cash: ${cash:.2f}, Portfolio Value: ${portfolio_value:.2f}")
            
            # Record daily results
            results.loc[date, 'Portfolio_Value'] = portfolio_value
            results.loc[date, 'GDX_Shares'] = gdx_shares
            results.loc[date, 'Put_Contracts'] = put_contracts
            results.loc[date, 'Strike_Price'] = strike_price
            results.loc[date, 'Maturity'] = maturity
            results.loc[date, 'Option_Cost'] = option_value
            results.loc[date, 'Hedge_Ratio'] = hedge_ratio if 'hedge_ratio' in locals() else 0.0
        
        # Calculate daily returns
        results['Daily_Return'] = results['Portfolio_Value'].pct_change()
        
        # Add unhedged portfolio for comparison
        self._add_unhedged_portfolio(results)
        
        # Store results
        self.results = results
        
        # Print column names for debugging
        print(f"\nResults DataFrame Columns: {list(results.columns)}")
        
        return results
    
    def _add_unhedged_portfolio(self, results):
        """
        Add an unhedged portfolio to the results for comparison.
        
        Args:
            results (pd.DataFrame): Results DataFrame to add unhedged portfolio to.
        """
        # Calculate initial shares that can be bought with initial capital
        initial_gdx_price = results['GDX_Price'].iloc[0]
        initial_shares = int(self.initial_capital / (initial_gdx_price * (1 + self.strategy.tx_cost_stock)))
        initial_cost = initial_shares * initial_gdx_price * (1 + self.strategy.tx_cost_stock)
        
        # Calculate unhedged portfolio value over time
        results['Unhedged_Value'] = results['GDX_Price'] * initial_shares + (self.initial_capital - initial_cost)
        
        # Calculate daily returns
        results['Unhedged_Return'] = results['Unhedged_Value'].pct_change()
        
        return results
    
    def _calculate_performance_metrics(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance metrics for the backtest.
        
        Parameters:
        -----------
        results : pd.DataFrame
            Results DataFrame
            
        Returns:
        --------
        pd.DataFrame
            Updated results DataFrame
        """
        # Calculate daily returns
        results['Daily_Return'] = results['Portfolio_Value'].pct_change()
        results['Unhedged_Return'] = results['Unhedged_Value'].pct_change()
        
        # Calculate cumulative returns
        results['Cumulative_Return'] = (1 + results['Daily_Return']).cumprod() - 1
        results['Unhedged_Cumulative_Return'] = (1 + results['Unhedged_Return']).cumprod() - 1
        
        # Calculate drawdowns
        results['High_Watermark'] = results['Portfolio_Value'].cummax()
        results['Unhedged_High_Watermark'] = results['Unhedged_Value'].cummax()
        results['Drawdown'] = (results['Portfolio_Value'] - results['High_Watermark']) / results['High_Watermark']
        results['Unhedged_Drawdown'] = (results['Unhedged_Value'] - results['Unhedged_High_Watermark']) / results['Unhedged_High_Watermark']
        
        return results
    
    def calculate_metrics(self):
        """
        Calculate performance metrics for the backtest.
        
        Returns:
            dict: Dictionary of performance metrics.
        """
        if self.results is None:
            raise ValueError("No results available. Run the backtest first.")
        
        # Get the last portfolio value and initial capital
        final_value = self.results['Portfolio_Value'].iloc[-1]
        initial_value = self.initial_capital
        
        # Calculate total return
        total_return = (final_value / initial_value - 1) * 100
        
        # Calculate daily returns
        returns = self.results['Daily_Return'].dropna()
        
        # Calculate annualized return
        days = (self.results.index[-1] - self.results.index[0]).days
        years = days / 365.25
        ann_return = ((final_value / initial_value) ** (1 / years) - 1) * 100
        
        # Calculate annualized volatility
        ann_vol = returns.std() * np.sqrt(252) * 100
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate for simplicity)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Calculate Sortino ratio (downside risk only)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) * 100
        sortino = ann_return / downside_vol if downside_vol > 0 else 0
        
        # Calculate maximum drawdown
        rolling_max = self.results['Portfolio_Value'].cummax()
        drawdown = ((self.results['Portfolio_Value'] - rolling_max) / rolling_max) * 100
        max_drawdown = drawdown.min()
        
        # Calculate average option cost as % of portfolio
        option_costs = self.results['Option_Cost'].dropna()
        portfolio_values = self.results.loc[option_costs.index, 'Portfolio_Value']
        avg_option_cost_pct = (option_costs / portfolio_values * 100).mean() if not option_costs.empty else 0
        
        # For unhedged comparison
        if 'Unhedged_Value' in self.results.columns:
            unhedged_final = self.results['Unhedged_Value'].iloc[-1]
            unhedged_total_return = (unhedged_final / initial_value - 1) * 100
            unhedged_returns = self.results['Unhedged_Value'].pct_change().dropna()
            unhedged_ann_return = ((unhedged_final / initial_value) ** (1 / years) - 1) * 100
            unhedged_ann_vol = unhedged_returns.std() * np.sqrt(252) * 100
            unhedged_sharpe = unhedged_ann_return / unhedged_ann_vol if unhedged_ann_vol > 0 else 0
            unhedged_downside = unhedged_returns[unhedged_returns < 0]
            unhedged_downside_vol = unhedged_downside.std() * np.sqrt(252) * 100
            unhedged_sortino = unhedged_ann_return / unhedged_downside_vol if unhedged_downside_vol > 0 else 0
            unhedged_rolling_max = self.results['Unhedged_Value'].cummax()
            unhedged_drawdown = ((self.results['Unhedged_Value'] - unhedged_rolling_max) / unhedged_rolling_max) * 100
            unhedged_max_drawdown = unhedged_drawdown.min()
        else:
            unhedged_total_return = unhedged_ann_return = unhedged_ann_vol = unhedged_sharpe = unhedged_sortino = unhedged_max_drawdown = None
        
        # Create dictionary of metrics
        metrics = {
            'total_return': total_return,
            'ann_return': ann_return,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'avg_option_cost_pct': avg_option_cost_pct,
            'unhedged_total_return': unhedged_total_return,
            'unhedged_ann_return': unhedged_ann_return,
            'unhedged_ann_vol': unhedged_ann_vol,
            'unhedged_sharpe': unhedged_sharpe,
            'unhedged_sortino': unhedged_sortino,
            'unhedged_max_drawdown': unhedged_max_drawdown
        }
        
        return metrics
    
    def plot_results(self):
        """
        Plot backtest results.
        """
        if self.results is None:
            raise ValueError("No results available. Run the backtest first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Get results
        results = self.results
        
        # 1. Portfolio NAV
        plt.figure(figsize=(14, 8))
        plt.plot(results.index, results['Portfolio_Value'], label='Hedged Portfolio')
        plt.plot(results.index, results['Unhedged_Value'], label='Unhedged Portfolio')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'portfolio_nav.png'))
        
        # 2. Drawdown comparison
        plt.figure(figsize=(14, 8))
        
        # Calculate drawdowns on the fly
        hedged_rolling_max = results['Portfolio_Value'].cummax()
        hedged_drawdown = ((results['Portfolio_Value'] - hedged_rolling_max) / hedged_rolling_max) * 100
        
        unhedged_rolling_max = results['Unhedged_Value'].cummax()
        unhedged_drawdown = ((results['Unhedged_Value'] - unhedged_rolling_max) / unhedged_rolling_max) * 100
        
        plt.plot(results.index, hedged_drawdown, label='Hedged Portfolio')
        plt.plot(results.index, unhedged_drawdown, label='Unhedged Portfolio')
        plt.title('Drawdown Comparison')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'drawdown.png'))
        
        # 3. GDX price with option transactions
        plt.figure(figsize=(14, 8))
        plt.plot(results.index, results['GLD_Price'], label='GLD Price')
        
        # Plot option strike prices at rebalancing
        rebalance_dates = self._generate_rebalance_dates()
        strikes = []
        expiries = []
        
        for date in rebalance_dates:
            strike = results.loc[date, 'Strike_Price']
            expiry = results.loc[date, 'Maturity']
            
            if expiry and not pd.isna(expiry):
                plt.axvline(x=date, color='g', alpha=0.3)
                plt.axvline(x=expiry, color='r', alpha=0.3)
                
                if not pd.isna(strike) and strike > 0:
                    plt.scatter(date, strike, color='g', alpha=0.7)
                    plt.scatter(expiry, strike, color='r', alpha=0.7)
                    
                    # Draw line connecting strike price points
                    plt.plot([date, expiry], [strike, strike], 'k--', alpha=0.3)
        
        plt.title('GLD Price with Option Transactions')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend(['GLD', 'Option Start', 'Option End', 'Strike Price'])
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'option_transactions.png'))
        
        # 4. Monthly hedge notional as % of portfolio
        rebalance_results = results[results['Option_Cost'] > 0].copy()
        rebalance_results['Hedge_Notional_Pct'] = rebalance_results['Option_Cost'] / rebalance_results['Portfolio_Value']
        
        plt.figure(figsize=(14, 8))
        plt.bar(rebalance_results.index, rebalance_results['Hedge_Notional_Pct'] * 100)
        plt.title('Hedge Notional as % of Portfolio')
        plt.xlabel('Date')
        plt.ylabel('Percentage (%)')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'hedge_notional.png'))
        
        # 5. Rolling beta
        plt.figure(figsize=(14, 8))
        plt.plot(self.strategy.beta.index, self.strategy.beta, color='purple')
        plt.axhline(y=1.0, color='r', linestyle='--')
        plt.title('Rolling Beta of GDX vs GLD')
        plt.xlabel('Date')
        plt.ylabel('Beta')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'rolling_beta.png'))
        
        # 6. Distribution of option costs over time
        plt.figure(figsize=(14, 8))
        sns.histplot(rebalance_results['Option_Cost'], kde=True)
        plt.title('Distribution of Option Costs')
        plt.xlabel('Option Cost ($)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'option_cost_distribution.png'))
        
        # 7. Terminal performance comparison
        metrics = self.calculate_metrics()
        
        plt.figure(figsize=(10, 6))
        metric_pairs = [
            ('total_return', 'unhedged_total_return'),
            ('ann_return', 'unhedged_ann_return'),
            ('max_drawdown', 'unhedged_max_drawdown'),
            ('sharpe', 'unhedged_sharpe'),
            ('sortino', 'unhedged_sortino')
        ]
        
        for i, (hedged_key, unhedged_key) in enumerate(metric_pairs):
            hedged_val = metrics[hedged_key]
            unhedged_val = metrics[unhedged_key]
            
            if hedged_val is None or unhedged_val is None:
                continue
            
            # For drawdowns, multiply by -1 to make positive for display purposes
            if 'drawdown' in hedged_key:
                hedged_val *= -1
                unhedged_val *= -1
            
            plt.bar(i - 0.2, hedged_val * 100 if 'return' in hedged_key or 'drawdown' in hedged_key else hedged_val, 
                    width=0.4, color='blue', alpha=0.7)
            plt.bar(i + 0.2, unhedged_val * 100 if 'return' in unhedged_key or 'drawdown' in unhedged_key else unhedged_val, 
                    width=0.4, color='orange', alpha=0.7)
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xticks(range(len(metric_pairs)), ['Total Return', 'Ann. Return', 'Max DD', 'Sharpe', 'Sortino'])
        plt.legend(['Hedged', 'Unhedged'])
        plt.title('Performance Comparison')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, 'performance_comparison.png'))
        
        # 8. Create a dashboard with multiple plots
        fig, axs = plt.subplots(3, 2, figsize=(20, 15))
        
        # Portfolio NAV
        axs[0, 0].plot(results.index, results['Portfolio_Value'], color='blue')
        axs[0, 0].plot(results.index, results['Unhedged_Value'], color='orange')
        axs[0, 0].set_title('Portfolio Value')
        axs[0, 0].set_xlabel('Date')
        axs[0, 0].set_ylabel('Value ($)')
        axs[0, 0].legend(['Hedged', 'Unhedged'])
        axs[0, 0].grid(True)
        
        # Drawdown
        axs[0, 1].plot(results.index, hedged_drawdown, color='blue')
        axs[0, 1].plot(results.index, unhedged_drawdown, color='orange')
        axs[0, 1].set_title('Drawdown Comparison')
        axs[0, 1].set_xlabel('Date')
        axs[0, 1].set_ylabel('Drawdown (%)')
        axs[0, 1].legend(['Hedged', 'Unhedged'])
        axs[0, 1].grid(True)
        
        # Beta and Option Costs
        axs[1, 0].plot(self.strategy.beta.index, self.strategy.beta, color='purple')
        axs[1, 0].axhline(y=1.0, color='black', linestyle='--')
        axs[1, 0].set_title('Rolling Beta of GDX vs GLD')
        axs[1, 0].set_xlabel('Date')
        axs[1, 0].set_ylabel('Beta')
        axs[1, 0].grid(True)
        
        # Option costs as % of portfolio
        if not rebalance_results.empty:
            axs[1, 1].bar(rebalance_results.index, rebalance_results['Hedge_Notional_Pct'] * 100, color='green')
            axs[1, 1].set_title('Hedge Cost as % of Portfolio')
            axs[1, 1].set_xlabel('Date')
            axs[1, 1].set_ylabel('Percentage (%)')
            axs[1, 1].grid(True)
        
        # GDX price with options
        axs[2, 0].plot(results.index, results['GLD_Price'], color='blue')
        for date in rebalance_dates:
            strike = results.loc[date, 'Strike_Price']
            expiry = results.loc[date, 'Maturity']
            
            if expiry and not pd.isna(expiry):
                if not pd.isna(strike) and strike > 0:
                    axs[2, 0].scatter(date, strike, color='g', alpha=0.7)
                    axs[2, 0].scatter(expiry, strike, color='r', alpha=0.7)
                    
                    # Draw line connecting strike price points
                    axs[2, 0].plot([date, expiry], [strike, strike], 'k--', alpha=0.3)
        
        axs[2, 0].set_title('GLD Price with Option Strikes')
        axs[2, 0].set_xlabel('Date')
        axs[2, 0].set_ylabel('Price ($)')
        axs[2, 0].grid(True)
        
        # Performance metrics comparison
        for i, (hedged_key, unhedged_key) in enumerate(metric_pairs):
            hedged_val = metrics[hedged_key]
            unhedged_val = metrics[unhedged_key]
            
            if hedged_val is None or unhedged_val is None:
                continue
            
            # For drawdowns, multiply by -1 to make positive for display purposes
            if 'drawdown' in hedged_key:
                hedged_val *= -1
                unhedged_val *= -1
            
            axs[2, 1].bar(i - 0.2, hedged_val * 100 if 'return' in hedged_key or 'drawdown' in hedged_key else hedged_val, 
                         width=0.4, color='blue', alpha=0.7)
            axs[2, 1].bar(i + 0.2, unhedged_val * 100 if 'return' in unhedged_key or 'drawdown' in unhedged_key else unhedged_val, 
                         width=0.4, color='orange', alpha=0.7)
        
        axs[2, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axs[2, 1].set_xticks(range(len(metric_pairs)))
        axs[2, 1].set_xticklabels(['Total Return', 'Ann. Return', 'Max DD', 'Sharpe', 'Sortino'])
        axs[2, 1].legend(['Hedged', 'Unhedged'])
        axs[2, 1].set_title('Performance Metrics')
        axs[2, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'dashboard.png'))
        plt.close('all')
    
    def save_results(self, filename: str = 'backtest_results.xlsx'):
        """
        Save backtest results to an Excel file with multiple sheets.
        
        Parameters:
        -----------
        filename : str
            Filename to save results (default: 'backtest_results.xlsx')
        """
        if self.results is None:
            raise ValueError("Backtest not run yet. Call run_backtest() first.")
        
        # Create Excel writer
        writer = pd.ExcelWriter(filename, engine='openpyxl')
        
        # Save daily portfolio data
        daily_data = self.results[[
            'GDX_Price', 'GLD_Price', 'Portfolio_Value', 'GDX_Shares',
            'Put_Contracts', 'Strike_Price', 'Option_Cost', 'Hedge_Ratio',
            'Daily_Return', 'Unhedged_Value', 'Unhedged_Return'
        ]].copy()
        
        daily_data.to_excel(writer, sheet_name='Daily Portfolio', index=True)
        
        # Calculate and save monthly data
        monthly_data = daily_data.resample('M').agg({
            'GDX_Price': 'last',
            'GLD_Price': 'last',
            'Portfolio_Value': 'last',
            'GDX_Shares': 'last',
            'Put_Contracts': 'last',
            'Strike_Price': 'last',
            'Option_Cost': 'mean',
            'Hedge_Ratio': 'mean',
            'Daily_Return': 'sum',
            'Unhedged_Value': 'last',
            'Unhedged_Return': 'sum'
        })
        monthly_data.to_excel(writer, sheet_name='Monthly Summary', index=True)
        
        # Save rebalance dates data
        rebalance_dates = self._generate_rebalance_dates()
        rebalance_data = self.results.loc[rebalance_dates].copy()
        rebalance_data.to_excel(writer, sheet_name='Rebalance Dates', index=True)
        
        # Calculate and save performance metrics
        metrics = self.calculate_metrics()
        metrics_df = pd.DataFrame({
            'Metric': [
                'Total Return (%)',
                'Annualized Return (%)',
                'Annualized Volatility (%)',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Maximum Drawdown (%)',
                'Average Option Cost (% of Portfolio)'
            ],
            'Hedged Portfolio': [
                metrics['total_return'],
                metrics['ann_return'],
                metrics['ann_vol'],
                metrics['sharpe'],
                metrics['sortino'],
                metrics['max_drawdown'],
                metrics['avg_option_cost_pct']
            ],
            'Unhedged Portfolio': [
                metrics['unhedged_total_return'],
                metrics['unhedged_ann_return'],
                metrics['unhedged_ann_vol'],
                metrics['unhedged_sharpe'],
                metrics['unhedged_sortino'],
                metrics['unhedged_max_drawdown'],
                None
            ]
        })
        metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
        
        # Save strategy parameters
        params_df = pd.DataFrame({
            'Parameter': [
                'Initial Capital',
                'Rebalance Frequency',
                'Moneyness',
                'Dynamic Hedge',
                'Min Hedge Ratio',
                'Max Hedge Ratio',
                'Stock Transaction Cost',
                'Option Transaction Cost'
            ],
            'Value': [
                f"${self.initial_capital:,.2f}",
                self.rebalance_freq,
                f"{self.moneyness*100}%",
                str(self.use_dynamic_hedge),
                f"{self.min_hedge_ratio*100}%",
                f"{self.max_hedge_ratio*100}%",
                f"{self.strategy.tx_cost_stock*100}%",
                f"{self.strategy.tx_cost_option*100}%"
            ]
        })
        params_df.to_excel(writer, sheet_name='Strategy Parameters', index=False)
        
        # Save and close the Excel writer
        writer.close()
        logger.info(f"Results saved to {filename}")
        
    def print_metrics(self):
        """
        Print performance metrics.
        """
        metrics = self.calculate_metrics()
        
        print("\n=== Performance Metrics ===")
        print(f"Rebalance Frequency: {self.rebalance_freq}")
        print(f"Moneyness: {int(self.moneyness * 100)}%")
        print(f"Dynamic Hedge: {self.use_dynamic_hedge}")
        print("----------------------------")
        print(f"Total Return: {metrics['total_return']:.2f}% (Unhedged: {metrics['unhedged_total_return']:.2f}%)")
        print(f"Annualized Return: {metrics['ann_return']:.2f}% (Unhedged: {metrics['unhedged_ann_return']:.2f}%)")
        print(f"Annualized Volatility: {metrics['ann_vol']:.2f}% (Unhedged: {metrics['unhedged_ann_vol']:.2f}%)")
        print(f"Sharpe Ratio: {metrics['sharpe']:.2f} (Unhedged: {metrics['unhedged_sharpe']:.2f})")
        print(f"Sortino Ratio: {metrics['sortino']:.2f} (Unhedged: {metrics['unhedged_sortino']:.2f})")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}% (Unhedged: {metrics['unhedged_max_drawdown']:.2f}%)")
        print(f"Average Option Cost: {metrics['avg_option_cost_pct']:.2f}% of portfolio") 