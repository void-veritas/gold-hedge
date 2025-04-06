#!/usr/bin/env python3
"""
Gold Miners Hedging Strategy - Main Entry Point
--------------------------------------------------------
This script runs a comprehensive exploratory data analysis on GDX and GLD
to understand their relationships, correlations, and characteristics for
developing an effective hedging strategy.
"""
import os
import argparse
from dotenv import load_dotenv
import warnings
import gold_hedge_eda as eda


def main():
    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Load environment variables
    load_dotenv()

    # Configure matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('fivethirtyeight')
    sns.set_palette('Set2')
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Gold Miners Hedging Strategy EDA')
    parser.add_argument('--api_key', type=str, help='Financial Modeling Prep API key')
    parser.add_argument('--start_date', type=str, default='2010-01-01', help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='End date for analysis (YYYY-MM-DD)')
    args = parser.parse_args()

    # Get API key from environment variable or command line
    api_key = os.environ.get('FMP_API_KEY')
    if args.api_key:
        api_key = args.api_key

    if not api_key:
        raise ValueError("API key must be provided via FMP_API_KEY environment variable or --api_key argument")

    # Run the complete analysis
    eda.run_full_analysis(api_key, start_date=args.start_date, end_date=args.end_date)


if __name__ == "__main__":
    main()