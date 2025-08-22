#!/usr/bin/env python3
"""
Test fetching and applying stock splits from SHARADAR/ACTIONS.
"""

import pandas as pd
from pathlib import Path
import nasdaqdatalink as ndl
from dotenv import load_dotenv
from typing import Dict, List, Tuple

# Load environment
load_dotenv()
ndl.api_config.read_key_from_environment_variable()

def fetch_splits(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch stock splits from SHARADAR/ACTIONS."""
    try:
        data = ndl.get_table(
            "SHARADAR/ACTIONS",
            ticker=ticker,
            action="split",
            date={"gte": start_date, "lte": end_date},
            qopts={"columns": ["date", "value"]},
            paginate=True
        )
        
        if data is not None and not data.empty:
            data['date'] = pd.to_datetime(data['date'])
            return data.sort_values('date')
    except Exception as e:
        print(f"Error fetching splits for {ticker}: {e}")
    
    return pd.DataFrame()

def calculate_split_adjustment(shares_before: float, split_from: float, split_to: float) -> float:
    """Calculate shares after a split.
    
    For a split_to:split_from ratio (e.g., 2:1 split has split_to=2, split_from=1)
    """
    return shares_before * (split_to / split_from)

def apply_splits_to_holdings(holdings: Dict[str, float], 
                            split_data: Dict[str, pd.DataFrame],
                            as_of_date: pd.Timestamp) -> Dict[str, float]:
    """Apply all splits up to a given date."""
    adjusted_holdings = holdings.copy()
    
    for ticker, shares in holdings.items():
        if ticker in split_data and not split_data[ticker].empty:
            # Apply all splits for this ticker up to as_of_date
            relevant_splits = split_data[ticker][split_data[ticker]['date'] <= as_of_date]
            
            for _, split in relevant_splits.iterrows():
                if pd.notna(split.get('value')):
                    # Use value field (multiplicative factor)
                    old_shares = adjusted_holdings.get(ticker, 0)
                    new_shares = old_shares * split['value']
                    adjusted_holdings[ticker] = new_shares
                    print(f"  {ticker}: {split['date'].date()} split factor {split['value']} "
                          f"-> {old_shares:.4f} becomes {new_shares:.4f}")
    
    return adjusted_holdings

def main():
    # Test stocks we know had issues
    test_tickers = ['SHOP', 'AMZN', 'NVDA', 'TSLA', 'DXCM', 'ANET', 'NFLX', 'TTD']
    
    print("Fetching splits from SHARADAR/ACTIONS...")
    print("=" * 60)
    
    split_data = {}
    for ticker in test_tickers:
        splits = fetch_splits(ticker, "2018-01-01", "2024-12-31")
        if not splits.empty:
            split_data[ticker] = splits
            print(f"\n{ticker} splits found:")
            for _, row in splits.iterrows():
                if pd.notna(row.get('value')):
                    print(f"  {row['date'].date()}: {row['value']}x split")
        else:
            print(f"\n{ticker}: No splits found")
    
    # Test with known holdings before splits
    print("\n" + "=" * 60)
    print("Testing split calculations:")
    print("=" * 60)
    
    # Example: SHOP had 30 shares before June 29, 2022 split
    test_holdings = {
        'SHOP': 30.0,  # Before 10:1 split
        'AMZN': 28.0,  # Before 20:1 split
        'NVDA': 155.27,  # Before June 2024 10:1 split (but after 2021 4:1 split)
        'TSLA': 40.0   # Before 2020 5:1 split
    }
    
    print("\nInitial holdings (before any splits):")
    for ticker, shares in test_holdings.items():
        print(f"  {ticker}: {shares:.4f}")
    
    print("\nApplying splits through July 31, 2024:")
    adjusted = apply_splits_to_holdings(
        test_holdings, 
        split_data, 
        pd.Timestamp("2024-07-31")
    )
    
    print("\nFinal holdings (after all splits):")
    for ticker, shares in adjusted.items():
        print(f"  {ticker}: {shares:.4f}")

if __name__ == "__main__":
    main()