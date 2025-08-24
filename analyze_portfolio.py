#!/usr/bin/env python3
"""
Portfolio XIRR analysis script.

Input files:
- transactions/complete_transactions.csv - Clean transaction data from clean_data.py
- transactions/current_holdings.txt - Current portfolio holdings

Calculates portfolio XIRR (Internal Rate of Return) and compares to benchmarks.
"""

import pandas as pd
import numpy as np
from scipy import optimize
import re
from datetime import datetime


def xnpv(rate, values, dates):
    """Calculate net present value of cash flows."""
    if rate <= -1.0:
        return float('inf')
    d0 = dates[0]
    return sum([vi / (1.0 + rate)**((di - d0).days / 365.0) for vi, di in zip(values, dates)])


def xirr(amounts, dates):
    """Calculate XIRR (Internal Rate of Return) using scipy optimization."""
    try:
        return optimize.newton(lambda r: xnpv(r, amounts, dates), 0.1)
    except:
        try:
            return optimize.brentq(lambda r: xnpv(r, amounts, dates), -0.999, 10.0)
        except:
            return None


def parse_current_holdings():
    """Parse current holdings file and extract equity value."""
    print("Parsing current portfolio holdings...")
    
    with open('transactions/current_holdings.txt', 'r') as f:
        lines = f.readlines()
    
    holdings = {}
    cash_value = 0
    total_value = 0
    
    for line in lines:
        line = line.strip()
        
        # Extract cash position
        if 'Cash & sweep funds:' in line:
            match = re.search(r'\$([0-9,]+\.\d+)', line)
            if match:
                cash_value = float(match.group(1).replace(',', ''))
        
        # Extract total value
        elif 'Total Value:' in line:
            match = re.search(r'\$([0-9,]+\.\d+)', line)
            if match:
                total_value = float(match.group(1).replace(',', ''))
        
        # Parse individual holdings
        elif line and line[0].isalpha() and not line.startswith(('Cash', 'Total')):
            parts = line.split()
            if len(parts) >= 4:
                ticker = parts[0]
                try:
                    shares = float(parts[1].replace(',', ''))
                    price = float(parts[2].replace('$', '').replace(',', ''))
                    value = float(parts[3].replace('$', '').replace(',', ''))
                    holdings[ticker] = {
                        'shares': shares,
                        'price': price,
                        'value': value
                    }
                except ValueError:
                    continue
    
    # Calculate equity value (excluding cash)
    equity_value = sum(h['value'] for h in holdings.values())
    
    print(f"  Portfolio breakdown:")
    print(f"    Equity value: ${equity_value:,.2f}")
    print(f"    Cash & sweep: ${cash_value:,.2f}")
    print(f"    Total value: ${total_value:,.2f}")
    
    return holdings, equity_value, cash_value, total_value


def load_transactions():
    """Load and prepare transaction data."""
    print("Loading transaction data...")
    
    df = pd.read_csv('transactions/complete_transactions.csv')
    df['Trade Date'] = pd.to_datetime(df['Trade Date'])
    
    # Filter to equity transactions only (exclude money markets)
    money_markets = ['VMFXX', 'VMSXX', 'SWVXX', 'SPAXX', 'QACDS', 'FDRXX']
    equity_df = df[~df['Ticker'].isin(money_markets)].copy()
    
    print(f"  Loaded {len(df)} total transactions ({len(equity_df)} equity)")
    print(f"  Date range: {df['Trade Date'].min().date()} to {df['Trade Date'].max().date()}")
    
    return equity_df


def calculate_portfolio_xirr(transactions_df, equity_value):
    """Calculate portfolio XIRR using complete transaction history."""
    print("Calculating portfolio XIRR...")
    
    # Create cash flows
    cash_flows = []
    dates = []
    
    # Process each transaction
    for _, row in transactions_df.iterrows():
        if row['Type'] == 'Buy':
            cash_flows.append(-abs(row['Amount USD']))  # Negative for purchases (outflow)
        elif row['Type'] == 'Sell':
            cash_flows.append(abs(row['Amount USD']))   # Positive for sales (inflow)
        dates.append(row['Trade Date'])
    
    # Add current portfolio value as final cash flow
    end_date = pd.Timestamp.today()
    cash_flows.append(equity_value)
    dates.append(end_date)
    
    # Calculate summary statistics
    total_invested = sum([-cf for cf in cash_flows if cf < 0])
    total_received = sum([cf for cf in cash_flows if cf > 0 and cf != equity_value])
    net_invested = total_invested - total_received
    
    print(f"  Cash flow summary:")
    print(f"    Total invested: ${total_invested:,.2f}")
    print(f"    Total sold: ${total_received:,.2f}")
    print(f"    Net invested: ${net_invested:,.2f}")
    print(f"    Current equity value: ${equity_value:,.2f}")
    
    # Calculate XIRR
    portfolio_xirr = xirr(cash_flows, dates)
    
    if portfolio_xirr is None:
        print("  ❌ Could not calculate XIRR")
        return None, net_invested, total_invested - total_received
    
    # Calculate time period and other metrics
    first_date = transactions_df['Trade Date'].min()
    years = (end_date - first_date).days / 365.25
    total_return_pct = (equity_value / net_invested - 1) * 100
    
    print(f"  📊 Results:")
    print(f"    Portfolio XIRR: {portfolio_xirr * 100:.2f}%")
    print(f"    Total return: {total_return_pct:.1f}%")
    print(f"    Time period: {years:.1f} years")
    
    return portfolio_xirr, net_invested, equity_value - net_invested


def compare_to_benchmarks(portfolio_xirr, years):
    """Compare portfolio performance to common benchmarks."""
    print("Comparing to benchmarks...")
    
    benchmarks = {
        'S&P 500 (VOO)': 0.11,  # ~11% historical average with dividends
        'NASDAQ 100 (QQQ)': 0.13,  # ~13% historical average
        'Total Stock Market (VTI)': 0.10  # ~10% historical average
    }
    
    if portfolio_xirr is None:
        print("  Cannot compare - XIRR calculation failed")
        return
    
    print(f"\n  Portfolio XIRR: {portfolio_xirr * 100:.2f}%")
    print(f"  Benchmark comparison:")
    
    for name, benchmark_return in benchmarks.items():
        outperformance = (portfolio_xirr - benchmark_return) * 100
        status = "✅" if outperformance > 0 else "❌"
        print(f"    {status} vs {name}: {outperformance:+.2f}% annually")
    
    # Show compound effect
    if years > 1:
        print(f"\n  Compound effect over {years:.1f} years:")
        voo_value = 1000 * (1.11 ** years)  # $1000 invested in VOO
        portfolio_value = 1000 * ((1 + portfolio_xirr) ** years)
        print(f"    $1,000 in VOO would be: ${voo_value:.0f}")
        print(f"    $1,000 in your portfolio: ${portfolio_value:.0f}")


def analyze_transaction_patterns(transactions_df):
    """Analyze transaction patterns and key positions."""
    print("Analyzing transaction patterns...")
    
    # Group by ticker
    ticker_summary = transactions_df.groupby('Ticker').agg({
        'Amount USD': ['sum', 'count'],
        'Type': lambda x: f"{sum(x=='Buy')}B/{sum(x=='Sell')}S"
    }).round(2)
    
    ticker_summary.columns = ['Total_Amount', 'Transaction_Count', 'Buy_Sell_Ratio']
    ticker_summary = ticker_summary.sort_values('Total_Amount', ascending=False)
    
    print(f"\n  Top 10 positions by transaction volume:")
    print(f"  {'Ticker':<8} {'Amount':>12} {'Count':>7} {'B/S':>8}")
    print("  " + "-" * 35)
    
    for ticker, row in ticker_summary.head(10).iterrows():
        print(f"  {ticker:<8} ${row['Total_Amount']:>11,.0f} {row['Transaction_Count']:>7.0f} {row['Buy_Sell_Ratio']:>8}")
    
    # Transaction timing analysis
    transactions_df['Year'] = transactions_df['Trade Date'].dt.year
    yearly_summary = transactions_df.groupby(['Year', 'Type'])['Amount USD'].sum().unstack(fill_value=0)
    
    print(f"\n  Investment timeline by year:")
    print(f"  {'Year':<6} {'Buys':>12} {'Sells':>12} {'Net':>12}")
    print("  " + "-" * 42)
    
    for year in sorted(yearly_summary.index):
        buys = yearly_summary.loc[year, 'Buy'] if 'Buy' in yearly_summary.columns else 0
        sells = yearly_summary.loc[year, 'Sell'] if 'Sell' in yearly_summary.columns else 0
        net = buys - sells
        print(f"  {year:<6} ${buys:>11,.0f} ${sells:>11,.0f} ${net:>11,.0f}")


def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print("INVESTMENT PORTFOLIO XIRR ANALYSIS")
    print("=" * 80)
    
    # Load data
    holdings, equity_value, cash_value, total_value = parse_current_holdings()
    transactions_df = load_transactions()
    
    # Calculate XIRR
    print("\n" + "=" * 80)
    portfolio_xirr, net_invested, total_gain = calculate_portfolio_xirr(transactions_df, equity_value)
    
    # Compare to benchmarks
    years = (pd.Timestamp.today() - transactions_df['Trade Date'].min()).days / 365.25
    print("\n" + "=" * 80)
    compare_to_benchmarks(portfolio_xirr, years)
    
    # Analyze patterns
    print("\n" + "=" * 80)
    analyze_transaction_patterns(transactions_df)
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if portfolio_xirr:
        print(f"\n🎯 Your portfolio's annualized return (XIRR): {portfolio_xirr * 100:.2f}%")
        print(f"💰 Net invested: ${net_invested:,.2f}")
        print(f"📈 Current equity value: ${equity_value:,.2f}")
        print(f"💵 Total gain: ${total_gain:,.2f}")
        print(f"⏱️  Time period: {years:.1f} years")
        
        # Risk-adjusted context
        if portfolio_xirr > 0.15:
            print(f"\n🚀 Excellent performance! You're significantly outperforming the market.")
        elif portfolio_xirr > 0.10:
            print(f"\n✅ Good performance, beating most benchmarks.")
        else:
            print(f"\n📊 Market-level performance.")
            
    else:
        print(f"\n❌ Could not calculate XIRR - check transaction data")
    
    print(f"\nNote: Analysis excludes ${cash_value:,.2f} in cash/sweep funds")
    print(f"Total account value: ${total_value:,.2f}")


if __name__ == "__main__":
    main()