import polars as pl
import polars.selectors as cs
import numpy as np
from pathlib import Path
from datetime import datetime, date
import warnings
import numpy_financial as npf
import os
import requests
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Config
pl.Config.set_tbl_rows(10)
pl.Config.set_fmt_str_lengths(50)
sns.set_style('whitegrid')
warnings.filterwarnings('ignore')

def load_transactions(dirname='/Users/bhargav/Git/investments/transactions'):
    """Load all transaction CSV files and combine them using Polars"""
    transactions_dir = Path(dirname)
    all_transactions = []
    
    # Define schema to ensure correct data types
    schema_overrides = {
        "Price USD": pl.Float64,
        "Quantity": pl.Float64,
        "Amount USD": pl.Float64
    }
    
    for csv_file in sorted(transactions_dir.glob('*.csv')):
        # Read CSV with Polars, specifying schema overrides
        df = pl.read_csv(csv_file, schema_overrides=schema_overrides)
        # Clean column names (remove BOM if present)
        df = df.rename({col: col.replace('﻿', '') for col in df.columns})
        all_transactions.append(df)
        print(f"Loaded {csv_file.name}: {len(df)} transactions")
    
    # Combine all years using Polars concat
    transactions = pl.concat(all_transactions, how="vertical")
    
    # Convert date column and sort (fixed date format)
    transactions = (
        transactions
        .with_columns(
            pl.col("Trade Date").str.to_date(format="%m/%d/%Y")  # Fixed date format
        )
        .sort("Trade Date")
    )
    
    return transactions

def consolidate_transaction_types(df):
    """
    Consolidate duplicate and similar transaction types using Polars
    """
    # Create mapping for consolidation
    type_mapping = {
        # Deposits - consolidate variations
        'DEPOSIT': 'Deposit',
        'Deposit': 'Deposit',
        
        # Splits - consolidate variations
        'STK SPLT': 'Split',
        'SPLT': 'Split',
        'Split': 'Split',
        
        # Dividends - keep as is
        'Dividend': 'Dividend',
        'DBS': 'Dividend',
        'DBT': 'Dividend',
        
        # Reinvestments
        'Reinvest': 'Reinvest',
        
        # Buys and Sells
        'Buy': 'Buy',
        'Sell': 'Sell',
        'LIQ': 'Sell',      # Liquidation
        
        # Interest
        'Interest': 'Interest',
        
        # Distributions and capital gains
        'Distribution': 'Distribution',
        'CAP': 'Capital Gain',
        
        # Tax-related
        'WHT': 'Tax Withheld',     # Withholding tax
        'FWT': 'Tax Withheld',     # Foreign withholding tax
        
        # Fees
        'ADR': 'Fee',          # ADR fee
        'MER': 'Fee',    # Management expense ratio
        
        # Corporate actions
        'WDL': 'Withdrawal',       # Withdrawal
        'BNK': 'Bank Transfer',    # Bank transfer
        'CIL': 'Corporate Action',     # Cash in lieu
        'Exchange': 'Corporate Action',     # Currency exchange or security exchange
    }
    
    # Apply the mapping using Polars replace and then uppercase
    df = df.with_columns(
        pl.col("Type")
        .replace(type_mapping, default=pl.col("Type"))
        .str.strip_chars()
        .str.to_uppercase()
    )

    return df

def filter_equity_transactions(df, MONEY_MARKET_FUNDS = ['VMFXX', 'QACDS', 'SPAXX', 'FDRXX', 'SWVXX', 'VMMXX']):
    """Filter for equity/ETF transactions only using Polars"""
    # Chain operations for efficient filtering
    df_filtered = (
        df
        # Remove money market funds
        .filter(~pl.col("Ticker").is_in(MONEY_MARKET_FUNDS))
        # Keep only relevant transaction types
        .filter(pl.col("Type").is_in(['BUY', 'SELL', 'DIVIDEND', 'REINVEST', 'CAPITAL GAIN']))
        # Remove rows without tickers
        .filter(pl.col("Ticker").is_not_null())
    )
    
    return df_filtered

if __name__ == "__main__":
    transactions = load_transactions()
    print(f"\nTotal transactions loaded: {len(transactions)}")

    transactions = consolidate_transaction_types(transactions)
    equity_transactions = filter_equity_transactions(transactions)
    print(f"Filtered to {len(equity_transactions)} equity transactions")
    
    print(f"\nUnique tickers ({equity_transactions['Ticker'].n_unique()}):")
    unique_tickers = (
        equity_transactions
        .select("Ticker")
        .unique()
        .sort("Ticker")
        .get_column("Ticker")
        .to_list()
    )
    print(sorted(unique_tickers))