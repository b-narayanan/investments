# Investment Returns Analysis

This notebook analyzes historical investment returns and compares them to a VOO (S&P 500) benchmark strategy.

## Step 1: Load Transaction Data

First, let's load all transaction files and understand the data structure:

```python
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load all transaction files
def load_transactions():
    """Load all transaction CSV files and combine them"""
    transactions_dir = Path('transactions')
    all_transactions = []
    
    for csv_file in sorted(transactions_dir.glob('*.csv')):
        df = pd.read_csv(csv_file)
        # Clean column names (remove BOM if present)
        df.columns = df.columns.str.replace('﻿', '')
        all_transactions.append(df)
    
    # Combine all years
    transactions = pd.concat(all_transactions, ignore_index=True)
    
    # Convert date column and sort
    transactions['Trade Date'] = pd.to_datetime(transactions['Trade Date'])
    transactions = transactions.sort_values('Trade Date')
    
    return transactions

# Load the data
transactions = load_transactions()
print(f"Loaded {len(transactions)} transactions")
print(f"Date range: {transactions['Trade Date'].min()} to {transactions['Trade Date'].max()}")
print("\nTransaction types:")
print(transactions['Type'].value_counts())
print("\nSample transactions:")
print(transactions.head(10))
```

## Step 2: Filter Relevant Transactions

Filter out money market funds and other non-equity transactions:

```python
# List of money market funds to exclude
MONEY_MARKET_FUNDS = ['VMFXX', 'QACDS', 'SPAXX', 'FDRXX']

def filter_equity_transactions(df):
    """Filter for equity/ETF transactions only"""
    # Remove money market funds
    df_filtered = df[~df['Ticker'].isin(MONEY_MARKET_FUNDS)].copy()
    
    # Remove bank transfers and other non-trading activities
    df_filtered = df_filtered[df_filtered['Type'].isin(['Buy', 'Sell', 'Dividend', 'Reinvest', 'CAP'])]
    
    # Remove rows without tickers
    df_filtered = df_filtered[df_filtered['Ticker'].notna()]
    
    return df_filtered

equity_transactions = filter_equity_transactions(transactions)
print(f"Filtered to {len(equity_transactions)} equity transactions")
print("\nUnique tickers:")
print(sorted(equity_transactions['Ticker'].unique()))
```

## Step 3: Set Up Nasdaq Data Link

Connect to Nasdaq Data Link for historical price data:

```python
import os
import requests
from typing import Dict, List
import time

# You'll need to set your API key as an environment variable
# export NASDAQ_DATA_LINK_API_KEY="your_key_here"

class NasdaqDataFetcher:
    """Fetch historical data from Nasdaq Data Link (Sharadar)"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('NASDAQ_DATA_LINK_API_KEY')
        if not self.api_key:
            raise ValueError("Please set NASDAQ_DATA_LINK_API_KEY environment variable")
        self.base_url = "https://data.nasdaq.com/api/v3"
        self.cache = {}
    
    def get_price_history(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get daily price history for a ticker"""
        
        # Check cache first
        cache_key = f"{ticker}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Sharadar SEP table for daily prices
        endpoint = f"{self.base_url}/datatables/SHARADAR/SEP"
        
        params = {
            'ticker': ticker,
            'date.gte': start_date,
            'date.lte': end_date,
            'api_key': self.api_key,
            'qopts.columns': 'ticker,date,closeadj,divamt'
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'datatable' in data and 'data' in data['datatable']:
                df = pd.DataFrame(data['datatable']['data'], 
                                columns=['ticker', 'date', 'closeadj', 'divamt'])
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                self.cache[cache_key] = df
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()
        
        # Rate limiting
        time.sleep(0.1)

# Initialize the fetcher (you'll need to set your API key)
# fetcher = NasdaqDataFetcher()
```

## Step 4: Calculate Actual Portfolio Returns

Calculate returns including dividend reinvestment:

```python
def calculate_portfolio_value(transactions_df: pd.DataFrame, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate portfolio value over time with dividend reinvestment
    """
    # Track holdings by ticker
    holdings = {}
    portfolio_history = []
    
    # Group transactions by date
    grouped = transactions_df.groupby('Trade Date')
    
    for date, day_transactions in grouped:
        # Process each transaction
        for _, txn in day_transactions.iterrows():
            ticker = txn['Ticker']
            txn_type = txn['Type']
            quantity = txn.get('Quantity', 0)
            
            if ticker not in holdings:
                holdings[ticker] = 0
            
            if txn_type in ['Buy', 'Reinvest']:
                holdings[ticker] += abs(quantity)
            elif txn_type == 'Sell':
                holdings[ticker] -= abs(quantity)
            # Dividends and CAP gains are handled through reinvestment
        
        # Calculate portfolio value for this date
        total_value = 0
        for ticker, shares in holdings.items():
            if shares > 0 and ticker in price_data:
                ticker_prices = price_data[ticker]
                price_on_date = ticker_prices[ticker_prices['date'] <= date]['closeadj'].iloc[-1] if len(ticker_prices) > 0 else 0
                total_value += shares * price_on_date
        
        portfolio_history.append({
            'date': date,
            'value': total_value,
            'holdings': dict(holdings)
        })
    
    return pd.DataFrame(portfolio_history)

# Example usage (when you have price data):
# portfolio_values = calculate_portfolio_value(equity_transactions, price_data_dict)
```

## Step 5: Simulate VOO Benchmark Portfolio

Create a function to simulate buying VOO instead:

```python
def simulate_voo_portfolio(transactions_df: pd.DataFrame, voo_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate portfolio if all Buy transactions went to VOO instead
    """
    voo_shares = 0
    voo_history = []
    
    # Get all buy transactions (including original reinvestments)
    buy_transactions = transactions_df[transactions_df['Type'].isin(['Buy', 'Reinvest'])].copy()
    
    for _, txn in buy_transactions.iterrows():
        date = txn['Trade Date']
        amount = abs(txn['Amount USD'])
        
        # Find VOO price on that date
        voo_price = voo_prices[voo_prices['date'] <= date]['closeadj'].iloc[-1] if len(voo_prices) > 0 else 500  # fallback
        
        # Calculate shares purchased
        shares_bought = amount / voo_price
        voo_shares += shares_bought
        
        # Track portfolio value
        voo_history.append({
            'date': date,
            'shares': voo_shares,
            'price': voo_price,
            'value': voo_shares * voo_price,
            'invested': amount
        })
    
    return pd.DataFrame(voo_history)

# Example usage:
# voo_portfolio = simulate_voo_portfolio(equity_transactions, voo_price_data)
```

## Step 6: Compare Returns

Calculate and compare annualized returns:

```python
def calculate_annualized_return(start_value: float, end_value: float, years: float) -> float:
    """Calculate annualized return"""
    if start_value <= 0 or years <= 0:
        return 0
    return (pow(end_value / start_value, 1/years) - 1) * 100

def compare_portfolios(actual_portfolio: pd.DataFrame, voo_portfolio: pd.DataFrame):
    """Compare actual vs VOO benchmark returns"""
    
    # Get total invested
    total_invested = abs(equity_transactions[equity_transactions['Type'] == 'Buy']['Amount USD'].sum())
    
    # Get final values
    actual_final = actual_portfolio['value'].iloc[-1]
    voo_final = voo_portfolio['value'].iloc[-1]
    
    # Calculate time period
    start_date = actual_portfolio['date'].min()
    end_date = actual_portfolio['date'].max()
    years = (end_date - start_date).days / 365.25
    
    # Calculate returns
    actual_return = calculate_annualized_return(total_invested, actual_final, years)
    voo_return = calculate_annualized_return(total_invested, voo_final, years)
    
    print(f"Investment Period: {start_date.date()} to {end_date.date()} ({years:.1f} years)")
    print(f"Total Invested: ${total_invested:,.2f}")
    print(f"\nActual Portfolio:")
    print(f"  Final Value: ${actual_final:,.2f}")
    print(f"  Total Return: {((actual_final/total_invested - 1) * 100):.2f}%")
    print(f"  Annualized Return: {actual_return:.2f}%")
    print(f"\nVOO Benchmark:")
    print(f"  Final Value: ${voo_final:,.2f}")
    print(f"  Total Return: {((voo_final/total_invested - 1) * 100):.2f}%")
    print(f"  Annualized Return: {voo_return:.2f}%")
    print(f"\nDifference:")
    print(f"  Outperformance: {actual_return - voo_return:+.2f}% annually")
    
    return {
        'actual_return': actual_return,
        'voo_return': voo_return,
        'outperformance': actual_return - voo_return
    }
```

## Step 7: Visualization

Create a chart comparing portfolio performance:

```python
import matplotlib.pyplot as plt

def plot_portfolio_comparison(actual_portfolio: pd.DataFrame, voo_portfolio: pd.DataFrame):
    """Plot portfolio values over time"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Portfolio values
    ax1.plot(actual_portfolio['date'], actual_portfolio['value'], 
             label='Actual Portfolio', linewidth=2)
    ax1.plot(voo_portfolio['date'], voo_portfolio['value'], 
             label='VOO Benchmark', linewidth=2, linestyle='--')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Portfolio Value Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Relative performance
    actual_returns = actual_portfolio['value'].pct_change().fillna(0)
    voo_returns = voo_portfolio['value'].pct_change().fillna(0)
    cumulative_diff = ((1 + actual_returns).cumprod() / (1 + voo_returns).cumprod() - 1) * 100
    
    ax2.plot(actual_portfolio['date'], cumulative_diff, 
             color='green' if cumulative_diff.iloc[-1] > 0 else 'red', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Outperformance vs VOO (%)')
    ax2.set_title('Cumulative Outperformance')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_portfolio_comparison(actual_portfolio, voo_portfolio)
```

## Main Execution Script

Put it all together:

```python
def main():
    """Main analysis function"""
    
    # Step 1: Load transactions
    print("Loading transactions...")
    transactions = load_transactions()
    equity_transactions = filter_equity_transactions(transactions)
    
    # Step 2: Get unique tickers and date range
    tickers = equity_transactions['Ticker'].unique().tolist()
    start_date = equity_transactions['Trade Date'].min().strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    # Step 3: Fetch price data from Nasdaq
    fetcher = NasdaqDataFetcher()
    price_data = {}
    
    # Always fetch VOO for benchmark
    print("Fetching VOO benchmark data...")
    voo_prices = fetcher.get_price_history('VOO', start_date, end_date)
    
    # Fetch data for all holdings
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        prices = fetcher.get_price_history(ticker, start_date, end_date)
        if not prices.empty:
            price_data[ticker] = prices
    
    # Step 4: Calculate actual portfolio performance
    print("\nCalculating actual portfolio performance...")
    actual_portfolio = calculate_portfolio_value(equity_transactions, price_data)
    
    # Step 5: Simulate VOO portfolio
    print("Simulating VOO benchmark portfolio...")
    voo_portfolio = simulate_voo_portfolio(equity_transactions, voo_prices)
    
    # Step 6: Compare results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    results = compare_portfolios(actual_portfolio, voo_portfolio)
    
    # Step 7: Create visualization
    plot_portfolio_comparison(actual_portfolio, voo_portfolio)
    
    return results

# Run the analysis
if __name__ == "__main__":
    results = main()
```

## Notes

- This analysis assumes all dividends are reinvested at the time they're received
- Money market funds (VMFXX, QACDS, etc.) are excluded as they represent cash holdings
- The VOO simulation invests the same dollar amounts on the same dates as your actual purchases
- Returns are calculated as annualized rates for fair comparison
- The Nasdaq Data Link API key needs to be set as an environment variable

## Next Steps

1. Set your Nasdaq Data Link API key: `export NASDAQ_DATA_LINK_API_KEY="your_key"`
2. Run the script section by section in Python REPL to test each component
3. Adjust the ticker filtering if needed for any special cases
4. Consider adding transaction costs if you want more precise calculations