# Gold Miners Hedging Strategy

This repository contains implementation of a systematic hedging strategy for gold mining stocks (GDX) using gold futures/ETF put options.

## Overview

Gold mining stocks typically have a leveraged exposure to gold prices (beta > 1). This repository explores a strategy that:

1. Invests in GDX (Gold Miners ETF) 
2. Hedges downside risk using GLD (Gold ETF) put options
3. Rebalances the hedge periodically (annually or quarterly)

The goal is to maintain exposure to gold miners' operational improvements and individual company performance while reducing the impact of significant gold price declines.

## Repository Contents

- **gold_strategy.py**: Core implementation of the hedging strategy
- **run_backtest.py**: Script to run the full backtest with various parameters
- **gold_hedge_eda.py**: Exploratory Data Analysis of GDX and GLD relationship
- **.env.example**: Example environment variables file

## Key Features

- Dynamic beta calculation between GDX and GLD
- Black-Scholes option pricing with volatility skew adjustments
- Transaction cost modeling
- Drawdown analysis and visualization
- Performance comparison with unhedged GDX

## Requirements

```
pandas
numpy
matplotlib
seaborn
requests
scipy
statsmodels
scikit-learn
```

## Setup

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your Financial Modeling Prep API key:
   - Sign up at [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs/)
   - Copy `.env.example` to `.env`
   - Add your API key to the `.env` file:
     ```
     FMP_API_KEY=your_api_key_here
     ```

## Usage

### Running the EDA

```bash
# Using environment variable for API key
python gold_hedge_eda.py --start_date 2010-01-01 --end_date 2023-12-31

# Or providing API key directly
python gold_hedge_eda.py --api_key YOUR_FMP_API_KEY --start_date 2010-01-01 --end_date 2023-12-31
```

### Running a Backtest

```bash
# Using environment variable for API key
python run_backtest.py --start_date 2016-01-01 --end_date 2023-12-31 --initial_capital 100000 --debug

# Or providing API key directly
python run_backtest.py --api_key YOUR_FMP_API_KEY --start_date 2016-01-01 --end_date 2023-12-31 --initial_capital 100000 --debug
```

## Key Findings

- GDX typically has a beta of ~2.0 to GLD, but this relationship varies over time
- The hedging strategy significantly reduces drawdowns during gold price declines
- Quarterly rebalancing outperforms annual rebalancing in terms of risk-adjusted returns
- Transaction costs impact is meaningful, especially for more frequent rebalancing

## License

MIT

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. 