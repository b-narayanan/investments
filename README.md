# Investment Portfolio XIRR Analysis

Calculate your portfolio's internal rate of return (XIRR) and compare performance against benchmarks.

## Quick Start

1. **Clean your data:**
   ```bash
   python clean_data.py
   ```

2. **Analyze your portfolio:**
   ```bash
   python analyze_portfolio.py
   ```

## Input Data

The analysis requires these files in the `transactions/` directory:

### Required Files
- **`buys.csv`** - Buy transactions from tax lot data
- **`current_holdings.txt`** - Current portfolio holdings snapshot
- **`{year}.csv`** - Historical transaction files (2018.csv, 2019.csv, etc.)

### File Formats

**buys.csv:**
```csv
Trade Date,Type,Ticker,Security Type,Price USD,Quantity,Amount USD
2025-01-27,Buy,NVDA,Stock,118.94,15.0,1784.10
```

**current_holdings.txt:**
```
CURRENT PORTFOLIO HOLDINGS

Ticker  Shares      Price       Value
NVDA    1,850.97223 $177.99     $329,454.54
...
Cash & sweep funds: $151,639.35 
Total Value:        $2,419,235.56
```

## Pipeline Overview

### 1. Data Cleaning (`clean_data.py`)

- Fixes data quality issues (column scaling, format errors)
- Extracts sell transactions from historical files
- Finds missing buy transactions for sold positions
- Combines all transactions into clean dataset
- Validates against current holdings

**Output:** `transactions/complete_transactions.csv`

### 2. XIRR Analysis (`analyze_portfolio.py`)

- Calculates portfolio XIRR using complete transaction history
- Compares performance to benchmarks (S&P 500, NASDAQ 100)
- Analyzes transaction patterns and key positions
- Excludes cash positions for accurate equity returns

## Key Features

- **Accurate XIRR calculation** using actual cash flows and timing
- **Complete transaction history** including all buys and sells
- **Benchmark comparison** against major indices
- **Data validation** against current holdings
- **Transaction analysis** showing patterns and key positions

## Output

The analysis provides:
- Portfolio XIRR (annualized return)
- Total return and gain/loss
- Benchmark comparisons
- Transaction patterns by year and ticker
- Data quality validation

## Example Results

```
🎯 Your portfolio's annualized return (XIRR): 23.19%
💰 Net invested: $1,192,874
📈 Current equity value: $2,267,596
💵 Total gain: $1,074,722
⏱️  Time period: 7.2 years

✅ vs S&P 500 (VOO): +12.19% annually
✅ vs NASDAQ 100 (QQQ): +10.19% annually
```

## Notes

- Analysis excludes cash/sweep fund positions for pure equity returns
- Handles stock splits, dividend reinvestments, and position sales
- Uses actual transaction timing for precise XIRR calculation
- Validates data quality and provides error reporting