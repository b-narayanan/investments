#!/usr/bin/env python3
"""
Investment portfolio analysis using pandas.
Computes XIRR and compares against VOO benchmark with dividend reinvestment.
"""

import argparse
import os
from datetime import datetime
from typing import Dict, Optional

import numpy_financial as npf
import pandas as pd
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

def load_transactions(csv_path: str, ignore_cash_equivalents: bool = True) -> pd.DataFrame:
    """Load and normalize transaction data."""
    df = pd.read_csv(csv_path)
    
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
    
    return df.sort_values("Trade Date").reset_index(drop=True)


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
    
    def get_prices(self, ticker: str, start: str, end: str) -> pd.Series:
        """Get daily closing prices."""
        cache_key = f"{ticker}_{start}_{end}_prices"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Try SHARADAR/SEP for equities and ETFs
            data = ndl.get_table(
                "SHARADAR/SEP",
                ticker=ticker,
                date={"gte": start, "lte": end},
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
    
    def get_dividends(self, ticker: str, start: str, end: str) -> pd.Series:
        """Get dividend per share amounts."""
        cache_key = f"{ticker}_{start}_{end}_divs"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Try SHARADAR/SEP for dividend data
            data = ndl.get_table(
                "SHARADAR/SEP",
                ticker=ticker,
                date={"gte": start, "lte": end},
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
    """Track portfolio positions and cash flows."""
    
    def __init__(self):
        self.holdings = {}  # ticker -> shares
        self.cash_flows = []  # (date, amount) for XIRR
    
    def process_transaction(self, row: pd.Series):
        """Process a single transaction."""
        ticker = row["Ticker"]
        txn_type = row["Type"]
        quantity = abs(row.get("Quantity", 0))
        amount = row["Amount USD"]
        date = row["Trade Date"]
        
        if ticker not in self.holdings:
            self.holdings[ticker] = 0
        
        if txn_type == "BUY":
            self.holdings[ticker] += quantity
            self.cash_flows.append((date, -abs(amount)))
            
        elif txn_type == "SELL":
            self.holdings[ticker] = max(0, self.holdings[ticker] - quantity)
            self.cash_flows.append((date, abs(amount)))
            
        elif txn_type == "DEPOSIT":
            self.cash_flows.append((date, -abs(amount)))
            
        elif txn_type == "WITHDRAWAL":
            self.cash_flows.append((date, abs(amount)))
            
        elif txn_type in ["DIVIDEND", "INTEREST"] and quantity == 0:
            self.cash_flows.append((date, abs(amount)))
            
        elif txn_type == "REINVEST":
            self.holdings[ticker] += quantity
    
    def buy_shares(self, ticker: str, date: datetime, amount: float, price: float):
        """Buy shares at a specific price."""
        if ticker not in self.holdings:
            self.holdings[ticker] = 0
        shares = amount / price
        self.holdings[ticker] += shares
        self.cash_flows.append((date, -amount))
        return shares
    
    def sell_shares(self, ticker: str, date: datetime, amount: float, price: float):
        """Sell shares at a specific price."""
        if ticker in self.holdings:
            shares = amount / price
            self.holdings[ticker] = max(0, self.holdings[ticker] - shares)
        self.cash_flows.append((date, amount))
    
    def add_cash_flow(self, date: datetime, amount: float):
        """Add a cash flow (deposit/withdrawal/dividend)."""
        self.cash_flows.append((date, amount))
    
    def reinvest_dividends(self, ticker: str, dividends: pd.Series, prices: pd.Series):
        """Reinvest all dividends for a ticker."""
        if ticker not in self.holdings:
            return
            
        for date, div_per_share in dividends.items():
            shares_held = self.holdings.get(ticker, 0)
            if shares_held > 0:
                div_amount = shares_held * div_per_share
                # Reinvest at ex-date price
                valid_prices = prices[prices.index <= date]
                if not valid_prices.empty:
                    price = valid_prices.iloc[-1]
                    new_shares = div_amount / price
                    self.holdings[ticker] += new_shares
    
    def get_value(self, date: datetime, prices: Dict[str, pd.Series]) -> float:
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
    
    def calculate_xirr(self, end_date: datetime, end_value: float) -> Optional[float]:
        """Calculate money-weighted return (XIRR)."""
        if not self.cash_flows:
            return None
        
        # Add final value
        flows = self.cash_flows + [(end_date, end_value)]
        dates = [cf[0] for cf in flows]
        amounts = [cf[1] for cf in flows]
        
        try:
            return npf.xirr(amounts, dates)
        except:
            return None


# ---------------------------- Benchmark Simulation ---------------------------- #

def simulate_benchmark(df: pd.DataFrame, benchmark: str, prices: pd.Series, 
                       dividends: pd.Series) -> Portfolio:
    """Simulate investing in benchmark on same dates as actual trades."""
    portfolio = Portfolio()
    
    # Process transactions
    for _, row in df.iterrows():
        date = row["Trade Date"]
        txn_type = row["Type"]
        amount = abs(row["Amount USD"])
        
        # Get price on transaction date
        valid_prices = prices[prices.index <= date]
        price = valid_prices.iloc[-1] if not valid_prices.empty else None
        
        if txn_type == "BUY" and price:
            portfolio.buy_shares(benchmark, date, amount, price)
            
        elif txn_type == "SELL" and price:
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
    
    # Date range
    start_date = df["Trade Date"].min().strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
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
            series = fetcher.get_prices(ticker, start_date, end_date)
            if not series.empty:
                prices[ticker] = series
    
    # Process actual portfolio
    actual = Portfolio()
    for _, row in equity_df.iterrows():
        actual.process_transaction(row)
    
    actual_value = actual.get_value(end_dt, prices)
    actual_xirr = actual.calculate_xirr(end_dt, actual_value)
    
    # Simulate benchmark
    bench_prices = fetcher.get_prices(benchmark, start_date, end_date)
    bench_divs = fetcher.get_dividends(benchmark, start_date, end_date)
    
    bench = simulate_benchmark(equity_df, benchmark, bench_prices, bench_divs)
    bench_value = bench.get_value(end_dt, {benchmark: bench_prices})
    bench_xirr = bench.calculate_xirr(end_dt, bench_value)
    
    # Calculate metrics
    total_invested = abs(df[df["Type"] == "BUY"]["Amount USD"].sum())
    
    results = {
        "period": f"{start_date} to {end_date}",
        "total_invested": total_invested,
        "actual": {
            "final_value": actual_value,
            "xirr": actual_xirr * 100 if actual_xirr else None,
            "total_return": (actual_value / total_invested - 1) * 100 if total_invested > 0 else 0
        },
        "benchmark": {
            "ticker": benchmark,
            "final_value": bench_value,
            "xirr": bench_xirr * 100 if bench_xirr else None,
            "total_return": (bench_value / total_invested - 1) * 100 if total_invested > 0 else 0
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
    print(f"Total Invested: ${results['total_invested']:,.2f}")
    
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
        print("Warning: Set NASDAQ_DATA_LINK_API_KEY for market data")
    
    # Load and analyze
    df = load_transactions(args.csv)
    results = analyze_portfolio(df, args.benchmark, args.end_date)
    print_results(results)


if __name__ == "__main__":
    main()