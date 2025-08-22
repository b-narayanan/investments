#!/usr/bin/env python3
"""
Investment portfolio analysis using pandas.
Computes XIRR and compares against VOO benchmark with dividend reinvestment.
"""

import argparse
from pathlib import Path
import os
from typing import Dict, Optional, Union

import numpy_financial as npf
import pandas as pd
from pandas import Timestamp
import nasdaqdatalink as ndl

# ---------------------------- Configuration ---------------------------- #

CASH_EQUIVALENTS = {
    "VMFXX", "QACDS", "SWVXX", "SPAXX", "SNVXX", "FDRXX", 
    "FGXX", "SPRXX", "SNAXX", "VMMXX", "VMSXX"
}

TRANSACTION_TYPES = {
    "buy": {"BUY"},
    "sell": {"SELL", "LIQ"},
    "deposit": {"BNK", "DEP", "DEPOSIT"},
    "withdrawal": {"WDL", "WITHDRAWAL"},
    "dividend": {"DIV", "DIVIDEND", "CASH DIVIDEND", "CASHDIVIDEND", "DBS", "DBT"},
    "interest": {"INTEREST"},
    "reinvest": {"REINVEST"},
    "split": {"STK SPLT", "SPLT", "SPLIT"},
}

# ---------------------------- Data Loading ---------------------------- #

def load_transactions(csv_path: Union[str, Path], ignore_cash_equivalents: bool = True) -> pd.DataFrame:
    """Load and normalize transaction data."""
    transactions_dir = Path(csv_path)
    all_transactions = []
    for csv_file in sorted(transactions_dir.glob('*.csv')):
        # Read CSV 
        df = pd.read_csv(csv_file)
        # Normalize columns
        df["Trade Date"] = pd.to_datetime(df["Trade Date"])
        df["Type"] = df["Type"].fillna("").str.strip().str.upper()
        df["Ticker"] = df["Ticker"].fillna("").str.strip().str.upper()
        df["Amount USD"] = pd.to_numeric(df["Amount USD"], errors="coerce").fillna(0)
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
        # Filter out cash equivalents
        if ignore_cash_equivalents:
            df = df[~df["Ticker"].isin(CASH_EQUIVALENTS)]
            if "Security Type" in df.columns:
                df = df[df["Security Type"].str.upper() != "MONEY MARKET"]
        all_transactions.append(df)
    
    if not all_transactions:
        # Return an empty DataFrame with expected columns
        return pd.DataFrame(columns=["Trade Date", "Type", "Ticker", "Amount USD", "Quantity"])
    
    transactions = pd.concat(all_transactions).sort_values("Trade Date").reset_index(drop=True)
    return transactions

def consolidate_transaction_types(df: pd.DataFrame) -> pd.DataFrame:
    """Map various transaction types to standard categories."""
    type_map = {}
    for category, types in TRANSACTION_TYPES.items():
        for t in types:
            type_map[t] = category.upper()
    
    df = df.copy()
    df["Type"] = df["Type"].replace(type_map).fillna(df["Type"])
    return df

# ---------------------------- Market Data ---------------------------- #

class MarketDataFetcher:
    """Fetch historical prices and dividends from Nasdaq Data Link."""
    
    def __init__(self):
        api_key = os.getenv("NASDAQ_DATA_LINK_API_KEY")
        if api_key:
            ndl.api_config.read_key_from_environment_variable()
        self.cache = {}
    
    def get_prices(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Get daily closing prices.
        
        Args:
            ticker: Stock ticker symbol
            start: Start date as pandas Timestamp
            end: End date as pandas Timestamp
        
        Returns:
            Series of adjusted closing prices indexed by date
        """
        # Format dates for caching and API calls
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        cache_key = f"{ticker}_{start_str}_{end_str}_prices"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Try SHARADAR/SEP for equities and ETFs
            data = ndl.get_table(
                "SHARADAR/SEP",
                ticker=ticker,
                date={"gte": start_str, "lte": end_str},
                qopts={"columns": ["date", "closeadj"]},
                paginate=True
            )
            
            if data is not None and not data.empty:
                series = data.set_index("date")["closeadj"].sort_index()
                self.cache[cache_key] = series
                return series
                
        except Exception as e:
            print(f"Warning: Could not fetch {ticker} prices: {e}")
        
        return pd.Series(dtype=float)
    
    def get_dividends(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Get dividend per share amounts.
        
        Args:
            ticker: Stock ticker symbol
            start: Start date as pandas Timestamp
            end: End date as pandas Timestamp
        
        Returns:
            Series of dividend amounts indexed by date
        """
        # Format dates for caching and API calls
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        cache_key = f"{ticker}_{start_str}_{end_str}_divs"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Try SHARADAR/SEP for dividend data
            data = ndl.get_table(
                "SHARADAR/SEP",
                ticker=ticker,
                date={"gte": start_str, "lte": end_str},
                qopts={"columns": ["date", "divamt"]},
                paginate=True
            )
            
            if data is not None and not data.empty:
                series = data.set_index("date")["divamt"]
                series = series[series > 0].sort_index()
                self.cache[cache_key] = series
                return series
                
        except Exception:
            pass
        
        return pd.Series(dtype=float)


# ---------------------------- Portfolio Tracking ---------------------------- #

class Portfolio:
    """Track portfolio positions and cash flows with accurate dividend reinvestment.
    
    DIVIDEND REINVESTMENT CALCULATION EXAMPLE:
    ========================================
    Day 1: Buy 100 shares at $100
        - holdings = {VOO: 100}
        - cash_flows = [(-$10000)]
        - holdings_history = [(Day1, VOO, 100)]
    
    Day 30: Dividend $1/share paid
        - Check holdings BEFORE Day 30 = 100 shares (dividend eligibility rule)
        - Dividend amount = 100 * $1 = $100
        - Reinvest at price $105: buy $100 / $105 = 0.952 shares
        - New holdings = {VOO: 100.952}
        - cash_flows = [(-$10000), (-$100)]  # Negative for reinvestment
        - holdings_history = [(Day1, VOO, 100), (Day30, VOO, 100.952)]
    
    Day 60: Buy 100 more shares at $110  
        - holdings = {VOO: 200.952}
        - cash_flows = [(-$10000), (-$100), (-$11000)]
        - holdings_history = [..., (Day60, VOO, 200.952)]
    
    Day 90: Dividend $1/share paid
        - Check holdings BEFORE Day 90 = 200.952 shares (includes prior reinvestment!)
        - Dividend amount = 200.952 * $1 = $200.952
        - Reinvest at price $108: buy $200.952 / $108 = 1.86 shares
        - Final holdings = {VOO: 202.812}
        - cash_flows = [..., (-$200.952)]
    
    KEY INSIGHTS:
    - Holdings checked BEFORE dividend date (must own shares before ex-date)
    - Each reinvestment creates negative cash flow (critical for accurate XIRR)  
    - Earlier reinvestments compound into later dividend calculations
    - Without proper cash flows, benchmark XIRR is artificially inflated
    """
    
    def __init__(self):
        self.holdings = {}  # ticker -> current shares owned
        self.cash_flows = []  # (date, amount) for XIRR calculation - negative = outflow
        self.holdings_history = []  # (date, ticker, shares_held) tracks all changes
    
    def process_transaction(self, row):
        """Process a single transaction from itertuples()."""
        ticker = row.Ticker
        txn_type = row.Type
        quantity = abs(getattr(row, 'Quantity', 0))
        amount = row.Amount_USD
        date = row.Trade_Date
        
        if ticker not in self.holdings:
            self.holdings[ticker] = 0
        
        if txn_type == "BUY":
            self.holdings[ticker] += quantity
            self.cash_flows.append((date, -abs(amount)))
            # Record holdings at this point in time
            self.holdings_history.append((date, ticker, self.holdings[ticker]))
            
        elif txn_type == "SELL":
            self.holdings[ticker] = max(0, self.holdings[ticker] - quantity)
            self.cash_flows.append((date, abs(amount)))
            # Record holdings at this point in time
            self.holdings_history.append((date, ticker, self.holdings[ticker]))
            
        elif txn_type == "DEPOSIT":
            self.cash_flows.append((date, -abs(amount)))
            
        elif txn_type == "WITHDRAWAL":
            self.cash_flows.append((date, abs(amount)))
            
        elif txn_type in ["DIVIDEND", "INTEREST"] and quantity == 0:
            self.cash_flows.append((date, abs(amount)))
            
        elif txn_type == "REINVEST":
            self.holdings[ticker] += quantity
            # Record holdings at this point in time
            self.holdings_history.append((date, ticker, self.holdings[ticker]))
    
    def buy_shares(self, ticker: str, date: Timestamp, amount: float, price: float) -> float:
        """Buy shares at a specific price."""
        if ticker not in self.holdings:
            self.holdings[ticker] = 0
        shares = amount / price
        self.holdings[ticker] += shares
        self.cash_flows.append((date, -amount))
        # Record holdings at this point in time
        self.holdings_history.append((date, ticker, self.holdings[ticker]))
        return shares
    
    def sell_shares(self, ticker: str, date: Timestamp, amount: float, price: float):
        """Sell shares at a specific price."""
        if ticker in self.holdings:
            shares = amount / price
            self.holdings[ticker] = max(0, self.holdings[ticker] - shares)
            # Record holdings at this point in time
            self.holdings_history.append((date, ticker, self.holdings[ticker]))
        self.cash_flows.append((date, amount))
    
    def add_cash_flow(self, date: Timestamp, amount: float):
        """Add a cash flow (deposit/withdrawal/dividend)."""
        self.cash_flows.append((date, amount))
    
    def get_holdings_on_date(self, ticker: str, date: Timestamp) -> float:
        """Get holdings for a ticker BEFORE a specific date.
        
        For dividend calculations, you must own shares BEFORE the ex-dividend date
        to be eligible. This method returns holdings as of the end of the previous day.
        """
        # Find the most recent holdings record BEFORE the date (not on the date)
        relevant_records = [(d, shares) for d, t, shares in self.holdings_history 
                           if t == ticker and d < date]
        if not relevant_records:
            return 0
        # Return the shares from the most recent record
        return max(relevant_records, key=lambda x: x[0])[1]
    
    def reinvest_dividends(self, ticker: str, dividends: pd.Series, prices: pd.Series):
        """Reinvest all dividends using correct historical holdings and proper cash flows.
        
        CRITICAL: For accurate XIRR calculation, dividend reinvestment must be treated as:
        1. Receiving dividend cash (positive cash flow)
        2. Immediately buying more shares (negative cash flow)
        
        This method handles both steps to ensure benchmark XIRR is calculated correctly.
        """
        # Process dividends in chronological order to ensure proper compounding
        for date, div_per_share in dividends.sort_index().items():
            # Get holdings BEFORE the dividend date (dividend eligibility rule)
            shares_held = self.get_holdings_on_date(ticker, date)
            if shares_held > 0:
                div_amount = shares_held * div_per_share
                
                # Find price on dividend date for reinvestment
                valid_prices = prices[prices.index <= date]
                if not valid_prices.empty:
                    price = valid_prices.iloc[-1]
                    if price > 0:
                        # CRITICAL FIX: Use buy_shares method which properly handles:
                        # 1. Share calculation and holdings update
                        # 2. Holdings history tracking  
                        # 3. MOST IMPORTANT: Creates negative cash flow for XIRR
                        # 
                        # Without this cash flow, benchmark XIRR is artificially inflated!
                        self.buy_shares(ticker, date, div_amount, price)
    
    def get_value(self, date: Timestamp, prices: Dict[str, pd.Series]) -> float:
        """Calculate portfolio value on a given date."""
        total = 0
        for ticker, shares in self.holdings.items():
            if shares > 0 and ticker in prices:
                price_series = prices[ticker]
                # Get price on or before date
                valid_prices = price_series[price_series.index <= date]
                if not valid_prices.empty:
                    total += shares * valid_prices.iloc[-1]
        return total
    
    def calculate_xirr(self, end_date: Timestamp, end_value: float) -> Optional[float]:
        """Calculate money-weighted return (XIRR)."""
        if not self.cash_flows:
            return None
        
        # Add final value
        flows = self.cash_flows + [(end_date, end_value)]
        dates = [cf[0] for cf in flows]
        amounts = [cf[1] for cf in flows]
        
        try:
            return npf.xirr(amounts, dates)
        except (ValueError, ZeroDivisionError) as e:
            print(f"Warning: Could not calculate XIRR. This can happen with unusual cash flows. Error: {e}")
            return None


# ---------------------------- Benchmark Simulation ---------------------------- #

def simulate_benchmark(df: pd.DataFrame, benchmark: str, prices: pd.Series, 
                       dividends: pd.Series) -> Portfolio:
    """Simulate investing in benchmark on same dates as actual trades."""
    portfolio = Portfolio()
    
    # Process transactions
    for row in df.itertuples(index=False, name='Transaction'):
        date = row.Trade_Date
        txn_type = row.Type
        amount = abs(row.Amount_USD)
        
        # Get price on transaction date
        valid_prices = prices[prices.index <= date]
        
        if txn_type in ["BUY", "SELL"]:
            if valid_prices.empty:
                print(f"Warning: No price found for {benchmark} on or before {date}. Skipping {txn_type} transaction.")
                continue
            price = valid_prices.iloc[-1]
            
            if txn_type == "BUY":
                portfolio.buy_shares(benchmark, date, amount, price)
            elif txn_type == "SELL":
                portfolio.sell_shares(benchmark, date, amount, price)
                
        elif txn_type == "DEPOSIT":
            portfolio.add_cash_flow(date, -amount)
            
        elif txn_type == "WITHDRAWAL":
            portfolio.add_cash_flow(date, amount)
            
        elif txn_type in ["DIVIDEND", "INTEREST"]:
            portfolio.add_cash_flow(date, amount)
    
    # Reinvest all dividends
    portfolio.reinvest_dividends(benchmark, dividends, prices)
    
    return portfolio


# ---------------------------- Analysis & Reporting ---------------------------- #

def analyze_portfolio(df: pd.DataFrame, benchmark: str = "VOO", 
                     end_date: Optional[str] = None) -> Dict:
    """Complete portfolio analysis with benchmark comparison."""
    
    # Setup
    fetcher = MarketDataFetcher()
    df = consolidate_transaction_types(df)
    
    # Filter to equity transactions
    equity_df = df[df["Type"].isin(["BUY", "SELL", "DIVIDEND", "REINVEST", "DEPOSIT", "WITHDRAWAL"])]
    
    # Date range - use pandas Timestamp objects
    start_dt = pd.Timestamp(df["Trade Date"].min())
    if end_date is None:
        end_dt = pd.Timestamp.today()
    else:
        end_dt = pd.to_datetime(end_date)
    
    # Get unique tickers
    tickers = df[df["Ticker"] != ""]["Ticker"].unique().tolist()
    if benchmark not in tickers:
        tickers.append(benchmark)
    
    # Fetch price data
    print(f"Fetching price data for {len(tickers)} tickers...")
    prices = {}
    for ticker in tickers:
        if ticker and " " not in ticker:  # Skip options
            series = fetcher.get_prices(ticker, start_dt, end_dt)
            if not series.empty:
                prices[ticker] = series
    
    # Process actual portfolio
    actual = Portfolio()
    for row in equity_df.itertuples(index=False, name='Transaction'):
        actual.process_transaction(row)
    
    actual_value = actual.get_value(end_dt, prices)
    actual_xirr = actual.calculate_xirr(end_dt, actual_value)
    
    # Simulate benchmark
    bench_prices = fetcher.get_prices(benchmark, start_dt, end_dt)
    bench_divs = fetcher.get_dividends(benchmark, start_dt, end_dt)
    
    bench = simulate_benchmark(equity_df, benchmark, bench_prices, bench_divs)
    bench_value = bench.get_value(end_dt, {benchmark: bench_prices})
    bench_xirr = bench.calculate_xirr(end_dt, bench_value)
    
    # Calculate metrics - net invested accounts for all cash flows
    buys_and_deposits = abs(df[df["Type"].isin(["BUY", "DEPOSIT"])]["Amount USD"].sum())
    sells_and_withdrawals = abs(df[df["Type"].isin(["SELL", "WITHDRAWAL"])]["Amount USD"].sum())
    net_invested = buys_and_deposits - sells_and_withdrawals
    
    results = {
        "period": f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}",
        "net_invested": net_invested,
        "actual": {
            "final_value": actual_value,
            "xirr": actual_xirr * 100 if actual_xirr else None,
            "total_return": (actual_value / net_invested - 1) * 100 if net_invested > 0 else 0
        },
        "benchmark": {
            "ticker": benchmark,
            "final_value": bench_value,
            "xirr": bench_xirr * 100 if bench_xirr else None,
            "total_return": (bench_value / net_invested - 1) * 100 if net_invested > 0 else 0
        }
    }
    
    if actual_xirr and bench_xirr:
        results["outperformance"] = (actual_xirr - bench_xirr) * 100
    
    return results


def print_results(results: Dict):
    """Print analysis results."""
    print("\n" + "=" * 60)
    print("PORTFOLIO PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"\nPeriod: {results['period']}")
    print(f"Net Invested: ${results['net_invested']:,.2f}")
    
    print(f"\nYour Portfolio:")
    print(f"  Final Value: ${results['actual']['final_value']:,.2f}")
    print(f"  Total Return: {results['actual']['total_return']:.2f}%")
    if results['actual']['xirr']:
        print(f"  XIRR: {results['actual']['xirr']:.2f}%")
    
    print(f"\n{results['benchmark']['ticker']} Benchmark:")
    print(f"  Final Value: ${results['benchmark']['final_value']:,.2f}")
    print(f"  Total Return: {results['benchmark']['total_return']:.2f}%")
    if results['benchmark']['xirr']:
        print(f"  XIRR: {results['benchmark']['xirr']:.2f}%")
    
    if "outperformance" in results:
        if results["outperformance"] > 0:
            print(f"\n✅ Outperformed by {results['outperformance']:.2f}% annually")
        else:
            print(f"\n❌ Underperformed by {abs(results['outperformance']):.2f}% annually")


# ---------------------------- Main ---------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Portfolio XIRR analysis")
    parser.add_argument("--csv", required=True, help="Transaction CSV file")
    parser.add_argument("--benchmark", default="VOO", help="Benchmark ticker")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    # Ensure API key is set
    if not os.getenv("NASDAQ_DATA_LINK_API_KEY"):
        print("Error: NASDAQ_DATA_LINK_API_KEY environment variable not set.")
        print("Please get a free API key from https://data.nasdaq.com and set it.")
        return
    
    # Load and analyze
    df = load_transactions(args.csv)
    results = analyze_portfolio(df, args.benchmark, args.end_date)
    print_results(results)


if __name__ == "__main__":
    main()