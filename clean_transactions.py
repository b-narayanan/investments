#!/opt/miniconda3/envs/antichain/bin/python
"""
Transaction Reconciliation and Portfolio Analysis Tool

This script provides comprehensive transaction reconciliation by:
1. Classifying transactions as BUYs vs dividend REINVESTs using heuristics.
2. Loading and matching sell transactions from historical data (2018-2025).
3. Handling corporate actions (e.g., ZNGA->TTWO conversion).
4. Computing a complete position lifecycle and comparing it with current holdings.

The goal is to achieve near-perfect reconciliation between computed positions
from transaction history and actual current portfolio holdings.
"""

import pandas as pd
import glob
import os
from typing import Dict, Tuple, List

# ---------------------------- Configuration ---------------------------- #

CONFIG = {
    'files': {
        'current_holdings': 'transactions/current_holdings.txt',
        'transactions': 'transactions/buys.csv',
        'historical_pattern': 'transactions/[12][0-9][0-9][0-9].csv'
    },
    'classification': {
        'etf_threshold': 50,
        'stock_threshold': 75,
        'small_quantity_threshold': 0.5,
        'etf_extended_threshold': 100
    },
    'etf_tickers': {
        'VGT', 'VOO', 'VOX', 'VDC', 'SCHX', 'SCHA', 'SCHF', 'SCHZ', 'VTI', 'QQQ', 
        'FCOM', 'FTEC', 'FHLC', 'IYW'
    },
    'cash_equivalents': {
        "VMFXX", "QACDS", "SWVXX", "SPAXX", "SNVXX", "FDRXX", 
        "FGXX", "SPRXX", "SNAXX", "VMMXX", "VMSXX"
    },
    'corporate_actions': [
        {
            'date': '2022-05-24',
            'type': 'merger_conversion',
            'from_ticker': 'ZNGA',
            'to_ticker': 'TTWO',
            'conversion_ratio': 0.0406,
            'cash_in_lieu': 22.08,
            'description': 'ZNGA acquired by TTWO - 300 shares -> 12 shares + cash'
        },
        {
            'date': '2022-10-04',
            'type': 'stock_split',
            'ticker': 'NTDOY',
            'split_ratio': 5,
            'description': 'NTDOY 5-for-1 stock split'
        }
    ],
    'match_tolerances': {
        'exact': 0.0001,
        'close': 0.01,
        'minor': 1.0
    }
}

# ---------------------------- Data Loading ---------------------------- #

def load_current_holdings(filepath: str) -> Dict[str, float]:
    """
    Parse current portfolio holdings from a brokerage statement file.

    Args:
        filepath: Path to the current holdings text file.

    Returns:
        A dictionary mapping ticker symbols to share quantities,
        excluding cash equivalents.
    """
    holdings = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Skip header lines and parse holdings
    for line in lines[3:]:
        line = line.strip()
        if not line or line.startswith(('Cash', 'Total')):
            break
        
        parts = line.split()
        if len(parts) >= 4 and line[0].isalpha():
            ticker = parts[0]
            # Skip cash equivalents
            if ticker in CONFIG['cash_equivalents']:
                continue
            try:
                shares_str = parts[1].replace(',', '')
                holdings[ticker] = float(shares_str)
            except (ValueError, IndexError):
                print(f"   - Warning: Could not parse line: {line}")
                continue
    return holdings

def load_and_classify_transactions(filepath: str) -> pd.DataFrame:
    """
    Load main transaction data and classify each row as 'BUY' or 'REINVEST'.

    Args:
        filepath: Path to the main transactions CSV file.

    Returns:
        A DataFrame with an added 'Transaction_Type' column.
    """
    df = pd.read_csv(filepath)
    df['Transaction_Type'] = df.apply(classify_transaction_type, axis=1)
    return df

def load_historical_transactions(file_pattern: str) -> pd.DataFrame:
    """
    Load and combine all historical transaction files from a glob pattern.

    Args:
        file_pattern: Glob pattern for historical CSV files.

    Returns:
        A combined DataFrame of all historical transactions.
    """
    historical_files = glob.glob(file_pattern)
    if not historical_files:
        return pd.DataFrame()

    all_transactions = [pd.read_csv(f) for f in sorted(historical_files)]
    combined_df = pd.concat(all_transactions, ignore_index=True)

    if 'Trade Date' in combined_df.columns:
        combined_df['Trade_Date'] = pd.to_datetime(combined_df['Trade Date'])
    return combined_df

# ---------------------------- Data Processing ---------------------------- #

def classify_transaction_type(row: pd.Series) -> str:
    """
    Classify a transaction as 'BUY' or 'REINVEST' based on heuristics.

    Args:
        row: A Pandas Series representing a single transaction.

    Returns:
        'REINVEST' or 'BUY'.
    """
    amount = abs(row['Amount USD'])
    quantity = row['Quantity']
    ticker = row['Ticker']
    
    cfg = CONFIG['classification']
    is_etf = ticker in CONFIG['etf_tickers']
    
    small_amount_threshold = cfg['etf_threshold'] if is_etf else cfg['stock_threshold']
    
    if amount < small_amount_threshold:
        return 'REINVEST'
    if quantity < cfg['small_quantity_threshold']:
        return 'REINVEST'
    if is_etf and amount < cfg['etf_extended_threshold']:
        return 'REINVEST'
        
    return 'BUY'

def get_sell_transactions(historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all sell and liquidation transactions from historical data.

    Args:
        historical_df: DataFrame of historical transactions.

    Returns:
        A DataFrame containing only sell/liquidation transactions.
    """
    if historical_df.empty:
        return pd.DataFrame()
    
    sell_types = ['Sell', 'LIQ']
    return historical_df[historical_df['Type'].isin(sell_types)].copy()

def get_complete_historical_lifecycle(
    historical_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find the complete buy/sell history for any position that was ever sold.

    This function identifies all assets that have "Sell" or "LIQ" transactions
    and pulls their entire transaction history (both buys and sells),
    allowing the script to correctly calculate positions that were fully closed out.

    Args:
        historical_df: DataFrame of all historical transactions.

    Returns:
        A tuple of (buy_transactions_df, sell_transactions_df) for all
        positions that have at least one sell transaction in their history.
    """
    if historical_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    sells = get_sell_transactions(historical_df)
    if sells.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # Identify all unique tickers that have at least one sell transaction.
    tickers_with_sells = sells['Ticker'].unique()
    
    # Retrieve all historical buys for those specific tickers.
    historical_buys = historical_df[
        (historical_df['Ticker'].isin(tickers_with_sells)) & 
        (historical_df['Type'] == 'Buy')
    ]
    
    return historical_buys, sells

def handle_corporate_actions(actions: List[Dict]) -> pd.DataFrame:
    """
    Create transaction records for corporate actions defined in the config.

    Args:
        actions: A list of corporate action dictionaries from the config.

    Returns:
        A DataFrame of transactions representing corporate actions.
    """
    corporate_transactions = []
    for action in actions:
        if action['type'] == 'merger_conversion':
            # NOTE: Original share count is hardcoded based on known position.
            original_shares = 300
            shares_received = int(original_shares * action['conversion_ratio'])
            
            # Transaction for shares received in the new ticker.
            corporate_transactions.append({
                'Trade Date': action['date'], 'Type': 'Conversion',
                'Ticker': action['to_ticker'], 'Security Type': 'Stock',
                'Price USD': 0.0, 'Quantity': shares_received, 'Amount USD': 0.0,
                'Transaction_Type': 'CONVERSION'
            })
            # Transaction to zero out the original ticker position.
            corporate_transactions.append({
                'Trade Date': action['date'], 'Type': 'Conversion_Out',
                'Ticker': action['from_ticker'], 'Security Type': 'Stock', 
                'Price USD': 0.0, 'Quantity': -original_shares, 'Amount USD': 0.0,
                'Transaction_Type': 'CONVERSION'
            })
        elif action['type'] == 'stock_split':
            # For stock splits, we need to know the pre-split share count
            # NOTE: Hardcoding NTDOY position as 100 shares pre-split
            if action['ticker'] == 'NTDOY':
                original_shares = 100
                additional_shares = original_shares * (action['split_ratio'] - 1)
                
                # Add shares from the split
                corporate_transactions.append({
                    'Trade Date': action['date'], 'Type': 'Stock_Split',
                    'Ticker': action['ticker'], 'Security Type': 'Stock',
                    'Price USD': 0.0, 'Quantity': additional_shares, 'Amount USD': 0.0,
                    'Transaction_Type': 'SPLIT'
                })
            
    return pd.DataFrame(corporate_transactions)

def create_complete_transaction_history() -> pd.DataFrame:
    """
    Build a comprehensive transaction history from all available sources.

    This function combines:
    1. Main classified transactions from 'buys.csv'.
    2. The complete historical lifecycle (buys and sells) for any traded position.
    3. Transactions generated from corporate actions.

    Returns:
        A deduplicated and standardized DataFrame of all transactions.
    """
    # 1. Load main and historical transactions.
    main_df = load_and_classify_transactions(CONFIG['files']['transactions'])
    historical_df = load_historical_transactions(CONFIG['files']['historical_pattern'])
    
    # 2. Filter out cash equivalents from both dataframes
    main_df = main_df[~main_df['Ticker'].isin(CONFIG['cash_equivalents'])]
    historical_df = historical_df[~historical_df['Ticker'].isin(CONFIG['cash_equivalents'])]
    
    # 3. Get historical lifecycle and corporate actions.
    historical_buys, historical_sells = get_complete_historical_lifecycle(historical_df)
    corporate_df = handle_corporate_actions(CONFIG['corporate_actions'])
    
    # 4. Add transaction types to new dataframes, avoiding SettingWithCopyWarning.
    if not historical_buys.empty:
        historical_buys = historical_buys.assign(Transaction_Type='BUY')
    if not historical_sells.empty:
        historical_sells = historical_sells.assign(Transaction_Type='SELL')

    # 5. Combine all sources.
    complete_df = pd.concat([main_df, historical_buys, historical_sells, corporate_df], ignore_index=True)
    
    # 6. Standardize and deduplicate.
    complete_df['Trade Date'] = pd.to_datetime(complete_df['Trade Date'], format='mixed', errors='coerce')
    
    complete_df['Abs_Amount'] = complete_df['Amount USD'].abs()
    dedup_columns = ['Trade Date', 'Ticker', 'Quantity', 'Abs_Amount']
    
    original_count = len(complete_df)
    complete_df = complete_df.drop_duplicates(subset=dedup_columns, keep='first')
    deduped_count = len(complete_df)
    
    if original_count != deduped_count:
        print(f"   - Removed {original_count - deduped_count} duplicate transactions.")
        
    return complete_df.drop(columns=['Abs_Amount'])

# ---------------------------- Reconciliation ---------------------------- #

def compute_final_positions(complete_df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Compute final positions and transaction summaries from the complete history.

    Args:
        complete_df: The comprehensive transaction history DataFrame.

    Returns:
        A tuple of (positions_dict, transaction_summary_dict).
    """
    stock_df = complete_df[complete_df['Security Type'].isin(['Stock', 'ETF'])]
    positions = stock_df.groupby('Ticker')['Quantity'].sum().to_dict()

    summary_stats = {}
    for ticker, final_pos in positions.items():
        ticker_df = complete_df[complete_df['Ticker'] == ticker]
        buys_reinv = ticker_df[ticker_df['Transaction_Type'].isin(['BUY', 'REINVEST'])]
        sells = ticker_df[ticker_df['Transaction_Type'] == 'SELL']

        summary_stats[ticker] = {
            'final_position': final_pos,
            'buy_count': (ticker_df['Transaction_Type'] == 'BUY').sum(),
            'reinvest_count': (ticker_df['Transaction_Type'] == 'REINVEST').sum(),
            'sell_count': (ticker_df['Transaction_Type'] == 'SELL').sum(),
            'conversion_count': (ticker_df['Transaction_Type'] == 'CONVERSION').sum(),
            'total_invested': buys_reinv['Amount USD'].sum(),
            'total_sold': abs(sells['Amount USD'].sum())
        }
        
    return positions, summary_stats

def reconcile_positions(
    computed: Dict[str, float], 
    actual: Dict[str, float]
) -> pd.DataFrame:
    """
    Compare computed positions vs. actual holdings and determine match status.

    Args:
        computed: Positions calculated from transaction history.
        actual: Current holdings from the brokerage statement.

    Returns:
        A DataFrame with a detailed comparison for each position.
    """
    all_tickers = sorted(set(computed.keys()) | set(actual.keys()))
    tolerances = CONFIG['match_tolerances']
    
    results = []
    for ticker in all_tickers:
        computed_shares = computed.get(ticker, 0)
        holding_shares = actual.get(ticker, 0)
        difference = holding_shares - computed_shares
        
        abs_diff = abs(difference)
        if abs_diff < tolerances['exact']:
            status = "✅ EXACT"
        elif abs_diff < tolerances['close']:
            status = "✅ Match"
        elif abs_diff < tolerances['minor']:
            status = "🟡 Close"
        else:
            status = "❌ Mismatch"
        
        results.append({
            'Ticker': ticker,
            'Computed': computed_shares,
            'Holdings': holding_shares,
            'Difference': difference,
            'Status': status
        })
    
    return pd.DataFrame(results)

# ---------------------------- Reporting ---------------------------- #

def print_header():
    """Prints the main header for the script output."""
    print("=" * 80)
    print("PORTFOLIO RECONCILIATION & ANALYSIS")
    print("=" * 80)

def print_detailed_reconciliation(df: pd.DataFrame):
    """Prints the detailed table of reconciled positions."""
    print("\n" + "=" * 80)
    print("DETAILED POSITION RECONCILIATION")
    print("=" * 80)
    print(f"\n{'Ticker':<8} {'Computed':>12} {'Holdings':>12} {'Difference':>12} {'Status'}")
    print("-" * 60)
    
    df['AbsDiff'] = df['Difference'].abs()
    sorted_df = df.sort_values('AbsDiff', ascending=False)
    
    # Filter to show rows with active holdings, computed positions, or discrepancies.
    relevant_rows = sorted_df[
        (df['Holdings'].abs() > 0.0001) | 
        (df['Computed'].abs() > 0.0001) |
        (df['AbsDiff'].abs() > 0.0001)
    ]

    for _, row in relevant_rows.iterrows():
        print(
            f"{row['Ticker']:<8} {row['Computed']:>12.4f} {row['Holdings']:>12.4f} "
            f"{row['Difference']:>12.4f} {row['Status']}"
        )

def print_summary_and_mismatches(df: pd.DataFrame, summary: Dict):
    """Prints the summary section, including accuracy and mismatch details."""
    print("\n" + "=" * 80)
    print("RECONCILIATION SUMMARY")
    print("=" * 80)

    # Accuracy calculation for currently held positions
    positions_with_holdings = df[df['Holdings'] > 0]
    matches = positions_with_holdings['Status'].str.contains("✅").sum()
    total_positions = len(positions_with_holdings)
    accuracy = (matches / total_positions * 100) if total_positions > 0 else 100
    
    print(f"\nOverall Accuracy: {accuracy:.1f}% ({matches}/{total_positions} currently held positions match)")

    # Mismatch investigation
    mismatches = df[df['Status'] == "❌ Mismatch"]
    if not mismatches.empty:
        print("\n⚠️  Positions requiring investigation:")
        for _, row in mismatches.iterrows():
            ticker = row['Ticker']
            print(f"\n   - {ticker}: Expected {row['Computed']:.4f}, Found {row['Holdings']:.4f} (Diff: {row['Difference']:+.4f})")
            if ticker in summary:
                stats = summary[ticker]
                print(f"     History: {stats['buy_count']} Buys, {stats['reinvest_count']} Reinvests, {stats['sell_count']} Sells")
                print(f"     Flow: Invested ${stats['total_invested']:,.2f}, Sold ${stats['total_sold']:,.2f}")
            if row['Computed'] > 0 and row['Holdings'] == 0:
                print("     Hint: Position may have been fully sold, but not all sell transactions were found.")
            elif row['Holdings'] > 0 and row['Computed'] == 0:
                print("     Hint: Position is held but has no transaction record; check for missing data.")

    # NEW: Report on positions that were fully bought and sold
    closed_positions = df[
        (df['Holdings'] == 0) & 
        (df['Computed'].abs() < CONFIG['match_tolerances']['exact']) &
        (df['Ticker'].isin(summary.keys())) # Ensure it had transactions
    ]
    if not closed_positions.empty:
        print("\n🔄 Completely Closed Positions (Buy/Sell cycles complete):")
        tickers = ", ".join(closed_positions['Ticker'].tolist())
        print(f"   {tickers}")

    # Note on corporate actions
    if 'TTWO' in df['Ticker'].values:
        print("\n💡 Corporate Action Note: TTWO position includes ZNGA->TTWO conversion.")


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """
    Main function to orchestrate the entire reconciliation process.
    """
    print_header()

    try:
        # 1. Load data
        print("\n1. Loading and processing data...")
        actual_holdings = load_current_holdings(CONFIG['files']['current_holdings'])
        print(f"   - Found {len(actual_holdings)} positions in brokerage statement.")
        
        complete_history = create_complete_transaction_history()
        print(f"   - Built a complete history of {len(complete_history)} transactions.")
        
        type_counts = complete_history['Transaction_Type'].value_counts()
        print("   - Transaction breakdown:")
        for tx_type, count in type_counts.items():
            print(f"     - {tx_type}: {count} transactions")

        # 2. Compute expected positions from history
        print("\n2. Computing expected positions...")
        computed_positions, transaction_summary = compute_final_positions(complete_history)
        print(f"   - Calculated final positions for {len(computed_positions)} unique tickers.")
        
        # 3. Reconcile computed vs. actual
        print("\n3. Reconciling positions...")
        reconciliation_df = reconcile_positions(computed_positions, actual_holdings)
        mismatches = (reconciliation_df['Status'] == "❌ Mismatch").sum()
        print(f"   - Reconciliation complete. Found {mismatches} mismatches.")
        
        # 4. Report results
        print_detailed_reconciliation(reconciliation_df)
        print_summary_and_mismatches(reconciliation_df, transaction_summary)

        print(f"\n✅ Analysis complete.")

        return reconciliation_df

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Could not find a required file: {e.filename}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        return None


if __name__ == "__main__":
    reconciliation_results = main()
    # To save results, uncomment the following line:
    # if reconciliation_results is not None:
    #     reconciliation_results.to_csv('transactions/position_reconciliation.csv', index=False)