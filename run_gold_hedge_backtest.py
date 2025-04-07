#!/usr/bin/env python3
"""
Gold Miners Hedging Strategy - Runner Script
--------------------------------------------
This script runs the gold miners ETF (GDX) strategy backtest with put option hedging.
"""

import os
import argparse
import logging
from datetime import datetime
import pandas as pd
from gold_hedge_strategy import GoldHedgeStrategy
from gold_hedge_backtest import GoldHedgeBacktest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gold_hedge_runner')

def main():
    """
    Main function to run the backtest.
    """
    parser = argparse.ArgumentParser(description='Run gold miners hedging strategy backtest')
    
    # Add arguments
    parser.add_argument('--api_key', type=str, default=None, help='Financial Modeling Prep API key')
    parser.add_argument('--start_date', type=str, default='2015-01-01',
                      help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                      help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--initial_capital', type=float, default=100000,
                      help='Initial capital for the strategy')
    parser.add_argument('--rebalance_freq', type=str, default='monthly',
                      choices=['monthly', 'quarterly', 'annually'],
                      help='Rebalancing frequency')
    parser.add_argument('--moneyness', type=float, default=0.95,
                      help='Strike price as a percentage of spot price')
    parser.add_argument('--min_hedge_ratio', type=float, default=0.5,
                      help='Minimum hedge ratio for dynamic hedging')
    parser.add_argument('--max_hedge_ratio', type=float, default=1.0,
                      help='Maximum hedge ratio for dynamic hedging')
    parser.add_argument('--tx_cost_stock', type=float, default=0.001,
                      help='Transaction cost for stock trades')
    parser.add_argument('--tx_cost_option', type=float, default=0.002,
                      help='Transaction cost for option trades')
    parser.add_argument('--static_hedge', action='store_true',
                      help='Use static hedge ratio instead of dynamic')
    
    args = parser.parse_args()
    
    # Convert dates to datetime
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date) if args.end_date else pd.Timestamp.now()
    
    # Try to load API key from environment if not provided
    if args.api_key is None:
        # Load environment variables from .env file if available
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logging.info("Loaded .env file")
        except ImportError:
            logging.warning("python-dotenv not installed; using environment variables directly")
        
        args.api_key = os.environ.get('FMP_API_KEY')
        if args.api_key:
            logging.info("Loaded API key from environment")
        else:
            raise ValueError("API key not provided and not found in environment variables")
    
    # Print configuration
    print("\n=== Gold Miners Hedging Strategy Backtest ===")
    print(f"Period: {args.start_date} to {'present' if args.end_date is None else args.end_date}")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print(f"Rebalance Frequency: {args.rebalance_freq}")
    print(f"Moneyness: {args.moneyness*100}% of spot price")
    print(f"Hedge Type: {'Static' if args.static_hedge else 'Dynamic (Beta-Based)'}")
    print(f"Transaction Costs: {args.tx_cost_stock*100:.2f}% (stock), {args.tx_cost_option*100:.2f}% (options)")
    print("==============================================\n")
    
    # Initialize strategy
    strategy = GoldHedgeStrategy(
        start_date=start_date,
        end_date=end_date,
        beta_window=60,
        vol_window=30,
        risk_free_rate=0.015,
        tx_cost_stock=args.tx_cost_stock,
        tx_cost_option=args.tx_cost_option,
        use_cache=True
    )
    
    # Fetch data
    logging.info("Fetching data...")
    strategy.fetch_data(api_key=args.api_key)
    
    # Initialize backtest
    backtest = GoldHedgeBacktest(
        strategy=strategy,
        initial_capital=args.initial_capital,
        rebalance_freq=args.rebalance_freq,
        moneyness=args.moneyness,
        use_dynamic_hedge=not args.static_hedge,
        min_hedge_ratio=args.min_hedge_ratio,
        max_hedge_ratio=args.max_hedge_ratio,
        save_dir='figures'
    )
    
    # Run backtest
    logging.info("Running backtest...")
    results = backtest.run_backtest()
    
    # Generate plots
    logging.info("Generating plots...")
    backtest.plot_results()
    
    # Save results
    logging.info("Saving results to backtest_results.csv...")
    backtest.save_results()
    
    # Print metrics
    backtest.print_metrics()
    
    print("\nBacktest complete. Figures saved to figures/")

def run_multiple_configs():
    """
    Run backtest with multiple configurations for comparison.
    """
    # Check for API key
    api_key = os.environ.get('FMP_API_KEY')
    if not api_key:
        logger.error("No API key found in environment. Set FMP_API_KEY in .env file")
        return
    
    # Base parameters
    base_params = {
        'api_key': api_key,
        'start_date': '2015-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 100000,
        'use_cache': True,
        'output_dir': 'comparison_figures'
    }
    
    # Initialize strategy with data
    strategy = GoldHedgeStrategy(
        start_date=base_params['start_date'],
        end_date=base_params['end_date'],
        use_cache=base_params['use_cache']
    )
    strategy.fetch_data(api_key=base_params['api_key'])
    
    # Define configurations to test
    configs = [
        {'rebalance_freq': 'monthly', 'moneyness': 0.95, 'use_dynamic_hedge': True, 'name': 'Monthly_Dynamic_95OTM'},
        {'rebalance_freq': 'quarterly', 'moneyness': 0.95, 'use_dynamic_hedge': True, 'name': 'Quarterly_Dynamic_95OTM'},
        {'rebalance_freq': 'monthly', 'moneyness': 0.90, 'use_dynamic_hedge': True, 'name': 'Monthly_Dynamic_90OTM'},
        {'rebalance_freq': 'monthly', 'moneyness': 0.95, 'use_dynamic_hedge': False, 'name': 'Monthly_Static_95OTM'},
    ]
    
    # Run each configuration
    results = []
    for config in configs:
        print(f"\n=== Running config: {config['name']} ===")
        backtest = GoldHedgeBacktest(
            strategy=strategy,
            initial_capital=base_params['initial_capital'],
            rebalance_freq=config['rebalance_freq'],
            moneyness=config['moneyness'],
            use_dynamic_hedge=config['use_dynamic_hedge'],
            save_dir=f"{base_params['output_dir']}/{config['name']}"
        )
        
        # Run backtest
        backtest_results = backtest.run_backtest()
        
        # Plot results
        backtest.plot_results()
        
        # Save results
        backtest.save_results(f"{config['name']}_results.csv")
        
        # Print metrics
        backtest.print_metrics()
        
        # Store metrics for comparison
        metrics = backtest.get_performance_metrics()
        metrics['Name'] = config['name']
        results.append(metrics)
    
    # Compare results
    compare_results(results)

def compare_results(results):
    """
    Compare and visualize results from multiple backtests.
    
    Parameters:
    -----------
    results : List[Dict]
        List of dictionaries with backtest metrics
    """
    # Create comparison directory
    os.makedirs('comparison', exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save comparison to CSV
    df.to_csv('comparison/backtest_comparison.csv', index=False)
    
    # Plot key metrics
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.style.use('fivethirtyeight')
    sns.set_palette('Set2')
    
    # Key metrics to compare
    key_metrics = [
        'Total Return', 
        'Annualized Return', 
        'Annualized Volatility', 
        'Sharpe Ratio', 
        'Maximum Drawdown',
        'Average Option Cost (% of Portfolio)'
    ]
    
    # Create bar charts for each metric
    for metric in key_metrics:
        plt.figure(figsize=(12, 6))
        
        # Format data for display
        if 'Return' in metric or 'Drawdown' in metric:
            data = df[metric] * 100  # Convert to percentage
            ylabel = f"{metric} (%)"
        else:
            data = df[metric]
            ylabel = metric
        
        # For drawdown, make positive for display
        if 'Drawdown' in metric:
            data = data * -1
        
        # Plot bars
        sns.barplot(x='Name', y=data, data=df)
        
        plt.title(f"Comparison: {metric}")
        plt.ylabel(ylabel)
        plt.xlabel('Strategy Configuration')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"comparison/{metric.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')}.png", bbox_inches='tight')
        plt.close()
    
    # Create summary dashboard
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    axs = axs.flatten()
    
    for i, metric in enumerate(key_metrics):
        # Format data for display
        if 'Return' in metric or 'Drawdown' in metric:
            data = df[metric] * 100  # Convert to percentage
            ylabel = f"{metric} (%)"
        else:
            data = df[metric]
            ylabel = metric
        
        # For drawdown, make positive for display
        if 'Drawdown' in metric:
            data = data * -1
        
        # Plot bars
        sns.barplot(x='Name', y=data, data=df, ax=axs[i])
        
        axs[i].set_title(metric)
        axs[i].set_ylabel(ylabel)
        axs[i].set_xlabel('')
        axs[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("comparison/summary_dashboard.png", bbox_inches='tight')
    plt.close()
    
    print("\nComparison complete. Results saved to comparison/")

if __name__ == "__main__":
    main() 