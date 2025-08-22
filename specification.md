# Investment Portfolio XIRR Analysis - Technical Specification

## Overview

The `xirr_analysis.py` script is designed to evaluate investment portfolio performance against market benchmarks using time-weighted and money-weighted return calculations. The primary goal is to answer the fundamental question: **"Is my investment strategy beating the market?"**

## Core Objectives

1. **Performance Evaluation**: Calculate actual portfolio performance using Extended Internal Rate of Return (XIRR)
2. **Benchmark Comparison**: Simulate equivalent investments in a market benchmark (e.g., VOO/S&P 500)
3. **Fair Comparison**: Ensure identical cash flow timing between actual and benchmark portfolios
4. **Comprehensive Reporting**: Provide clear performance metrics and outperformance analysis

## Key Financial Calculations

### 1. Net Invested Capital
- **Formula**: `(Sum of BUYS + DEPOSITS) - (Sum of SELLS + WITHDRAWALS)`
- **Purpose**: Represents total investor capital at risk
- **Importance**: Base for calculating total return percentages

### 2. Portfolio Market Value
- **Calculation**: Current price × shares held for each position
- **Data Source**: Nasdaq Data Link (SHARADAR/SEP)
- **Timing**: Calculated as of specified end date

### 3. Total Return (Simple)
- **Formula**: `(Final Value / Net Invested Capital - 1) × 100`
- **Limitation**: Does not account for cash flow timing
- **Use**: Basic performance overview

### 4. Extended Internal Rate of Return (XIRR)
- **Definition**: Annualized money-weighted rate of return
- **Method**: Uses `numpy_financial.xirr` function
- **Input**: Series of dates and corresponding cash flows
- **Output**: Annualized percentage return accounting for timing

## Critical Implementation Requirements

### XIRR Calculation Methodology

#### Cash Flow Treatment
- **Negative Cash Flows**: Money leaving investor's account (BUYS, DEPOSITS)
- **Positive Cash Flows**: Money entering investor's account (SELLS, WITHDRAWALS, DIVIDENDS)
- **Final Value**: Treated as positive cash flow on end date

#### Transaction Type Mapping
- **BUY**: Negative cash flow, increase holdings
- **SELL**: Positive cash flow, decrease holdings  
- **DEPOSIT**: Negative cash flow (additional capital)
- **WITHDRAWAL**: Positive cash flow (capital removal)
- **DIVIDEND**: Positive cash flow (income received)
- **REINVEST**: Increase holdings, no direct cash flow

### Dividend Reinvestment Complexity

#### Why Dividend Reinvestment is Critical
1. **Significant Impact**: Dividends represent ~2% annual return for S&P 500
2. **Benchmark Accuracy**: Without proper modeling, benchmark returns are artificially low
3. **Fair Comparison**: Ensures actual vs benchmark comparison uses same assumptions

#### Implementation Subtleties

##### Temporal Accuracy Requirements
- **Eligibility Rule**: Must own shares BEFORE dividend ex-date to receive dividend
- **Holdings Calculation**: Use `get_holdings_on_date()` to check pre-dividend holdings
- **Chronological Processing**: Process dividends in date order for proper compounding

##### Cash Flow Modeling for XIRR
**Critical Requirement**: Dividend reinvestment must create negative cash flow

**Incorrect Approach** (artificially inflates XIRR):
```
shares += dividend_amount / price  # No cash flow recorded
```

**Correct Approach** (accurate XIRR):
```
1. Record dividend receipt (positive cash flow)
2. Immediately record share purchase (negative cash flow via buy_shares())
```

##### Compounding Effects
- **Share Accumulation**: Reinvested dividends buy additional shares
- **Future Eligibility**: New shares from reinvestment earn future dividends
- **Cascade Impact**: Early calculation errors compound over time

### Data Requirements and Sources

#### Transaction Data
- **Format**: CSV files with standardized columns
- **Required Fields**: Trade Date, Type, Ticker, Amount USD, Quantity
- **Normalization**: Consistent transaction type mapping
- **Filtering**: Optional exclusion of cash equivalents/money market funds

#### Market Data (Nasdaq Data Link)
- **Price Data**: Daily adjusted closing prices (SHARADAR/SEP closeadj)
- **Dividend Data**: Dividend per share amounts (SHARADAR/SEP divamt)
- **Date Range**: From first transaction to analysis end date
- **Caching**: In-memory caching to reduce API calls

### Portfolio State Tracking

#### Holdings Management
- **Current Holdings**: Dictionary mapping ticker → shares owned
- **Holdings History**: Time-series record for dividend eligibility calculations
- **Transaction Processing**: Update holdings and history for each transaction

#### Cash Flow Recording
- **Comprehensive Tracking**: All money movements affect XIRR
- **Temporal Precision**: Date-stamped cash flows for accurate timing
- **Final Value Integration**: End value added as positive cash flow

## Implementation Plan

### Phase 1: Data Loading and Normalization
1. **CSV Processing**: Load and concatenate multiple transaction files
2. **Data Cleaning**: Normalize dates, amounts, and transaction types
3. **Type Mapping**: Standardize transaction type vocabulary
4. **Filtering**: Remove cash equivalents if specified

### Phase 2: Market Data Acquisition
1. **API Setup**: Configure Nasdaq Data Link authentication
2. **Ticker Identification**: Extract unique tickers from transactions
3. **Price Fetching**: Retrieve daily adjusted prices for all tickers
4. **Dividend Fetching**: Get dividend data for benchmark simulation
5. **Caching Strategy**: Implement efficient data caching

### Phase 3: Actual Portfolio Processing
1. **Transaction Processing**: Iterate through chronological transactions
2. **Holdings Tracking**: Maintain current and historical position records
3. **Cash Flow Recording**: Track all monetary movements
4. **Final Valuation**: Calculate end-date portfolio value

### Phase 4: Benchmark Simulation
1. **Transaction Mapping**: Map actual transactions to benchmark purchases/sales
2. **Dividend Processing**: Implement accurate reinvestment modeling
3. **Holdings Management**: Track benchmark portfolio state
4. **Cash Flow Alignment**: Ensure identical timing with actual portfolio

### Phase 5: Analysis and Reporting
1. **XIRR Calculation**: Compute annualized returns for both portfolios
2. **Metric Computation**: Calculate total returns and other performance indicators
3. **Comparison Analysis**: Determine outperformance/underperformance
4. **Result Presentation**: Format and display comprehensive analysis

## Accuracy Considerations

### Potential Error Sources
1. **Price Data Gaps**: Missing prices for specific dates
2. **Transaction Timing**: Ensuring correct date usage for calculations
3. **Dividend Eligibility**: Accurate determination of share ownership dates
4. **Cash Flow Modeling**: Proper treatment of reinvestment flows

### Validation Strategies
1. **Holdings Verification**: Cross-check calculated holdings against known positions
2. **Cash Flow Audit**: Verify all transactions create appropriate cash flows
3. **Benchmark Validation**: Compare results against known benchmark performance
4. **Edge Case Testing**: Handle unusual transaction patterns and data gaps

## Performance and Scalability
- **API Rate Limiting**: Implement appropriate delays for market data requests
- **Memory Management**: Efficient handling of large transaction datasets
- **Error Handling**: Graceful degradation when data is unavailable
- **User Feedback**: Progress indicators for long-running operations