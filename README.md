# Gold Miners ETF Hedging Strategy Backtest

This project implements a backtest framework for evaluating a portfolio strategy involving:
- **Long Exposure**: Buy a gold miners equity ETF (GDX)
- **Hedge Exposure**: Buy put options on GDX to protect against downside risk

The strategy dynamically sizes the notional of put options based on the rolling beta of GDX vs GLD (as a proxy for spot gold). The intuition is to hedge the gold exposure embedded in GDX and adjust the size of the put hedge accordingly.

## Key Features

- Rolling beta calculation between GDX and GLD
- Dynamic option hedge sizing based on beta
- Black-Scholes option pricing simulation
- Flexible rebalancing periods (monthly, quarterly, annually)
- Configurable option moneyness levels
- Comprehensive performance analysis and visualization
- Transaction costs and slippage

## Requirements

- Python 3.6+
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scipy
  - requests
  - python-dotenv (optional, for environment variables)

Install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn scipy requests python-dotenv
```

## Files

- `gold_hedge_strategy.py`: Core strategy implementation (data handling, beta calculation, option pricing)
- `gold_hedge_backtest.py`: Backtest engine for running simulations
- `run_gold_hedge_backtest.py`: Runner script with command-line interface
- `README.md`: This documentation file

## Usage

### Basic Usage

```bash
python run_gold_hedge_backtest.py --api_key YOUR_API_KEY
```

### Full Options

```bash
python run_gold_hedge_backtest.py \
  --api_key YOUR_API_KEY \
  --start_date 2015-01-01 \
  --end_date 2023-12-31 \
  --beta_window 60 \
  --vol_window 30 \
  --initial_capital 100000 \
  --rebalance_freq monthly \
  --moneyness 0.95 \
  --min_hedge_ratio 0.5 \
  --max_hedge_ratio 1.0 \
  --tx_cost_stock 0.001 \
  --tx_cost_option 0.002 \
  --output_dir figures \
  --results_file backtest_results.csv
```

### API Key

You need a Financial Modeling Prep API key to download historical price data. You can provide it in three ways:

1. As a command-line argument: `--api_key YOUR_API_KEY`
2. In an environment variable: `FMP_API_KEY=YOUR_API_KEY`
3. In a `.env` file: `FMP_API_KEY=YOUR_API_KEY`

### Running Multiple Configurations

To compare different strategy configurations:

```bash
python -c "from run_gold_hedge_backtest import run_multiple_configs; run_multiple_configs()"
```

This will run the backtest with multiple configurations (different rebalancing periods, moneyness levels, and hedge types) and generate comparison visualizations.

## Backtest Methodology

1. **Initial Portfolio Composition**
   - Calculate beta of GDX vs GLD
   - Based on beta, determine the notional value to hedge
   - Buy put options to hedge the portfolio using the simulated option prices
   - Allocate the remaining capital to GDX

2. **Rebalancing Logic**
   - At the start of each rebalancing period:
     - If an option is in the money: Add profit to cash and recalculate the hedge
     - If an option is worthless: Adjust GDX position to afford new options
   - Track P&L from both GDX and put options

3. **Option Pricing**
   - Simulate put option prices using the Black-Scholes model
   - Inputs:
     - Spot price (GDX)
     - Strike price (configurable moneyness level)
     - Time to maturity (equal to rebalancing period)
     - Risk-free rate
     - Rolling historical volatility

## Output

### Results DataFrame

The backtest generates a DataFrame with:
- Date
- Portfolio value
- GDX shares owned
- GDX price
- GDX position value
- GLD price
- Option contracts bought at rebalancing
- Strike price
- Maturity
- Cash inflow from options at previous rebalancing
- Option cost
- Hedge ratio

### Visualizations

The backtest generates the following plots:
1. Portfolio NAV over time (with and without hedging)
2. Drawdown comparison (with and without puts)
3. Option deals visualization on the GDX chart
4. Hedge notional as % of portfolio
5. Rolling beta of GDX vs GLD
6. Distribution of option costs
7. Terminal performance comparison

## Performance Metrics

- Total return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Average option cost as % of portfolio

## Limitations

- Simulated option prices may not perfectly reflect market conditions
- No bid-ask spread modeling for options
- Limited liquidity considerations
- Historical relationships may not persist in the future

## Future Enhancements

- Add more sophisticated option pricing models
- Incorporate implied volatility surface
- Model bid-ask spreads for options
- Add more sophisticated transaction cost modeling
- Implement additional hedge instruments (e.g., futures)
- Add stress testing and scenario analysis 