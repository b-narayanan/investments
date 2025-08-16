Analyse @docling/20231231-statements-9262-.md and extract ALL transactions from December 2023 (12/01/2023 through 12/31/2023). Format them as CSV with these exact columns:
Trade Date,Type,Ticker,Security Type,Price USD,Quantity,Amount USD

  Requirements:
  1. Include ALL transaction types: Dividend, Reinvest, CAP (capital gains), DBS (debit sweep), WDL (withdrawal), DBT (debit), ADR (ADR fees), Interest, etc
  2. List in reverse chronological order (most recent date first)
  3. For dividends: Price USD=0, Quantity=0, Amount=positive
  4. For reinvestments: Price USD=0, Quantity=shares purchased, Amount=negative
  5. Include the exact transaction type code as shown in the statement (e.g., "CAP" not "Capital Gains")
  6. Security types should be: Stock, ETF, Mutual Fund, Money Market, Municipal Bond, or ADR as appropriate
  7. Look in multiple sections: ACTIVITY THIS PERIOD, INCOME, SWEEP PROGRAM ACTIVITY, and TRADE AND INVESTMENT ACTIVITY
  8. Match dividends with their corresponding reinvestments on the same date/security

Output the complete CSV data including all transactions found to december_2023_transactions.csv.