#!/usr/bin/env python3
"""Investigate transaction mismatches for specific tickers."""

import pandas as pd
import glob

# Load all historical files
files = glob.glob('transactions/[12][0-9][0-9][0-9].csv')
files.append('transactions/buys.csv')

problem_tickers = ['UPST', 'NTDOY', 'ADSK', 'APPN', 'JD']

print('=' * 80)
print('TRANSACTION HISTORY FOR PROBLEM TICKERS')
print('=' * 80)

for ticker in problem_tickers:
    print(f'\n{ticker} TRANSACTIONS:')
    print('-' * 60)
    
    found_any = False
    all_transactions = []
    
    for file in sorted(files):
        try:
            df = pd.read_csv(file)
            if 'Ticker' in df.columns:
                ticker_df = df[df['Ticker'] == ticker]
                if not ticker_df.empty:
                    found_any = True
                    print(f'\nFrom {file}:')
                    for _, row in ticker_df.iterrows():
                        date = row.get('Trade Date', 'N/A')
                        tx_type = row.get('Type', 'N/A')
                        quantity = row.get('Quantity', 0)
                        price = row.get('Price USD', 0)
                        amount = row.get('Amount USD', 0)
                        print(f'  {date}: {tx_type:10} Qty: {quantity:10.4f} Price: ${price:8.2f} Amount: ${amount:10.2f}')
                        
                        # Store for summary
                        all_transactions.append({
                            'Date': date,
                            'Type': tx_type,
                            'Quantity': quantity,
                            'Amount': amount,
                            'File': file
                        })
        except Exception as e:
            pass
    
    if not found_any:
        print(f'  No transactions found for {ticker}')
    else:
        # Calculate totals
        total_qty = sum(t['Quantity'] for t in all_transactions)
        buy_qty = sum(t['Quantity'] for t in all_transactions if t['Type'] in ['Buy', 'REINVEST'])
        sell_qty = sum(t['Quantity'] for t in all_transactions if t['Type'] in ['Sell', 'LIQ'])
        
        print(f'\n  SUMMARY for {ticker}:')
        print(f'    Total Buy Quantity:  {buy_qty:10.4f}')
        print(f'    Total Sell Quantity: {sell_qty:10.4f}')
        print(f'    Net Position:        {total_qty:10.4f}')
        print(f'    Transaction Count:   {len(all_transactions)}')

print('\n' + '=' * 80)
print('ANALYSIS OF DISCREPANCIES')
print('=' * 80)

print("""
Observations:
1. UPST: Shows 426 shares computed but 0 held - likely sold but sell transactions missing
2. NTDOY: Shows -400 shares computed - indicates more sells than buys recorded
3. ADSK: Shows 294 shares computed but 0 held - likely sold but sell transactions missing  
4. APPN: Shows 50 shares computed but 0 held - likely sold but sell transactions missing
5. JD: Shows -4 shares computed but 1 held - sell transactions may be overcounted
""")