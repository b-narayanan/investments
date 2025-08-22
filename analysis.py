import polars as pl
import polars.selectors as cs
import numpy as np
from pathlib import Path
from datetime import datetime, date
import warnings
import numpy_financial as npf
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import nasdaqdatalink

# Config
pl.Config.set_tbl_rows(10)
pl.Config.set_fmt_str_lengths(50)
sns.set_style('whitegrid')
warnings.filterwarnings('ignore')

########## Load and clean transaction data ##########

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

##########  Nasdaq Data Fetching ##########

class NasdaqDataFetcher:
    """Fetch historical data from Nasdaq Data Link using the official library - returns Polars DataFrames"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        nasdaqdatalink.ApiConfig.api_key = self.api_key
        # Configure robustness features
        nasdaqdatalink.ApiConfig.use_retries = True
        nasdaqdatalink.ApiConfig.number_of_retries = 3
        nasdaqdatalink.ApiConfig.retry_status_codes = [429, 500, 501, 502, 503, 504]
        self.cache = {}
    
    def get_price_history(self, ticker: str, start_date: str, end_date: str) -> pl.DataFrame:
        """Get daily price history for a ticker including adjusted close and dividends using SHARADAR/SEP"""
        # Check cache first
        cache_key = f"{ticker}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not self.api_key:
            return pl.DataFrame()

        try:
            # Use the nasdaq-data-link library to fetch from SHARADAR/SEP table
            # This provides adjusted close prices and dividend information
            data = nasdaqdatalink.get_table(
                'SHARADAR/SEP',
                ticker=ticker,
                date={'gte': start_date, 'lte': end_date},
                qopts={'columns': ['ticker', 'date', 'closeadj', 'divamt']},
                paginate=True  # Automatically handle pagination for large datasets
            )
            
            if data is not None and not data.empty:
                # Convert pandas DataFrame to Polars for consistency
                df = pl.from_pandas(data)
                
                # Ensure date column is properly typed
                if 'date' in df.columns:
                    df = df.with_columns(
                        pl.col('date').cast(pl.Date)
                    ).sort('date')
                
                self.cache[cache_key] = df
                return df
            else:
                print(f"No data found for {ticker} in the specified date range")
                return pl.DataFrame()
                
        except nasdaqdatalink.NotFoundError:
            print(f"Ticker {ticker} not found in SHARADAR/SEP database")
            return pl.DataFrame()
            
        except nasdaqdatalink.LimitExceededError:
            print(f"API limit exceeded. Please wait.")
            return pl.DataFrame()
            
        except nasdaqdatalink.AuthenticationError:
            print(f"Authentication failed. Please check your API key.")
            return pl.DataFrame()
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return pl.DataFrame()
    
    def get_fund_prices(self, ticker: str, start_date: str, end_date: str) -> pl.DataFrame:
        """Get fund prices from SHARADAR/SFP for ETFs and mutual funds"""
        
        cache_key = f"fund_{ticker}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not self.api_key:
            return self.get_price_history(ticker, start_date, end_date)  # Use dummy data
        
        try:
            # Try SHARADAR/SFP for funds (ETFs, mutual funds)
            data = nasdaqdatalink.get_table(
                'SHARADAR/SFP',
                ticker=ticker,
                date={'gte': start_date, 'lte': end_date},
                qopts={'columns': ['ticker', 'date', 'closeadj', 'divamt']},
                paginate=True
            )
            
            if data is not None and not data.empty:
                df = pl.from_pandas(data)
                if 'date' in df.columns:
                    df = df.with_columns(
                        pl.col('date').cast(pl.Date)
                    ).sort('date')
                self.cache[cache_key] = df
                return df
            else:
                return pl.DataFrame()
                
        except Exception:
            # If not found in SFP, fallback to SEP
            return self.get_price_history(ticker, start_date, end_date)
    
    def get_ticker_info(self, ticker: str) -> dict:
        """Get ticker metadata from SHARADAR/TICKERS table"""
        
        if not self.api_key:
            return {}
        
        try:
            data = nasdaqdatalink.get_table(
                'SHARADAR/TICKERS',
                ticker=ticker,
                qopts={'columns': ['ticker', 'name', 'exchange', 'isdelisted', 'category', 'sector', 'industry']}
            )
            
            if data is not None and not data.empty:
                return data.iloc[0].to_dict()
            return {}
            
        except Exception as e:
            print(f"Error fetching ticker info for {ticker}: {e}")
            return {}

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
    fetcher = NasdaqDataFetcher(os.getenv('NASDAQ_DATA_LINK_API_KEY'))
