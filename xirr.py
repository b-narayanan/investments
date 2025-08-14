#!/usr/bin/env python3
"""
Compute your money-weighted annualized return (XIRR) and compare it to a
shadow benchmark that invests in VOO on the exact same cash-flow dates,
with dividends reinvested for the benchmark.

Data sources:
- Your brokerage CSV (as shown in your 2024.csv example)
- Nasdaq Data Link Sharadar datasets:
    * SHARADAR/SEP  : Equity & ETF daily prices
    * SHARADAR/SFP  : Dividend per-share series for ETFs (fallback to SEP if needed)

Key behavior:
- Ignores Money Market transactions (e.g., VMFXX, QACDS) so parking cash doesn’t
  trigger benchmark trades.
- Treats external cash movements correctly (deposits as negative CF, withdrawals as positive CF).
- Reinvests all VOO dividends into additional VOO shares for the benchmark (total return).

Usage:
    export NASDAQ_DATA_LINK_API_KEY=YOUR_KEY
    python IRR_vs_VOO_with_Sharadar_reinvesting_dividends.py --csv /path/to/your.csv --benchmark VOO

Requires:
    pip install nasdaq-data-link pandas numpy numpy-financial python-dateutil
"""

import argparse
import os
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import numpy_financial as npf
import pandas as pd
import nasdaqdatalink

# ---------------------------- Configurable sets ----------------------------- #
# Cash-like tickers to ignore entirely (expand as needed)
CASH_EQUIVALENTS = {
    "VMFXX", "QACDS", "SWVXX", "SPAXX", "SNVXX", "FDRXX", "FGXX", "SPRXX", "SNAXX",
}

# Transaction-type normalization maps (case-insensitive matching applied)
BUY_TYPES = {"BUY"}
SELL_TYPES = {"SELL"}
# External money movement in
DEPOSIT_TYPES = {"BNK", "DEP", "DEPOSIT"}
# External money movement out
WITHDRAWAL_TYPES = {"WDL", "WITHDRAWAL"}
# Income events (cash paid to you by the position)
DIV_TYPES = {"DIV", "DIVIDEND", "CASH DIVIDEND", "CASHDIVIDEND"}
INTEREST_TYPES = {"INTEREST"}

SEP_DATASET = "SHARADAR/SEP"
SFP_DATASET = "SHARADAR/SFP"  # for ETF dividends per share

# ----------------------------- Helper functions ---------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XIRR vs VOO (Sharadar, dividends reinvested)")
    p.add_argument("--csv", required=True, help="Path to brokerage CSV (Chase-style)")
    p.add_argument("--benchmark", default="VOO", help="Benchmark ETF ticker (default: VOO)")
    p.add_argument("--end-date", default=datetime.today().strftime("%Y-%m-%d"),
                   help="Valuation end date (YYYY-MM-DD), default: today")
    p.add_argument("--ignore-cash-by-security-type", action="store_true",
                   help="Additionally drop rows where Security Type == 'Money Market' (recommended)")
    p.add_argument("--dump-cashflows", default=None,
                   help="Optional path to write out the real/shadow cashflows CSV")
    return p.parse_args()


def setup_api():
    api_key = os.getenv("NASDAQ_DATA_LINK_API_KEY")
    if not api_key:
        raise RuntimeError("Set NASDAQ_DATA_LINK_API_KEY in your environment.")
    nasdaqdatalink.ApiConfig.api_key = api_key


def normalize_types(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.upper()


def next_trading_close(series: pd.Series, date: pd.Timestamp) -> Tuple[pd.Timestamp, float]:
    """Return (actual_date, close_price) using the same or next available trading date."""
    if date in series.index:
        return date, float(series.loc[date])
    # next available trading date
    future = series.loc[series.index >= date]
    if future.empty:
        # fallback to last available
        return series.index[-1], float(series.iloc[-1])
    return future.index[0], float(future.iloc[0])


def load_sep_prices(tickers: List[str], start: str, end: str) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for t in sorted(set(tickers)):
        dfp = nasdaqdatalink.get(SEP_DATASET, ticker=t, start_date=start, end_date=end)
        if dfp is None or dfp.empty:
            raise ValueError(f"No SEP data for ticker {t}.")
        # Prefer 'close' if present; fallback sensibly
        dfp = dfp.copy()
        if "date" in dfp.columns:
            dfp.set_index("date", inplace=True)
        dfp.sort_index(inplace=True)
        col = None
        for candidate in ["close", "closeadj", "adj_close", "close_unadj", "closeunadj"]:
            if candidate in dfp.columns:
                col = candidate
                break
        if col is None:
            # If no obvious close column, try the first numeric column
            numeric_cols = dfp.select_dtypes(include=["float", "int"]).columns
            if len(numeric_cols) == 0:
                raise ValueError(f"SEP data for {t} missing close-like columns.")
            col = numeric_cols[0]
        out[t] = dfp[col].astype(float)
    return out


def load_dividends_per_share(ticker: str, start: str, end: str, fallback_prices: Optional[pd.DataFrame]=None) -> pd.Series:
    """Return a Series indexed by ex-date with dividend-per-share.
    We try SHARADAR/SFP first. If unavailable, we attempt to fall back to a 'dividends'
    column in SEP for the ticker. Returns empty Series if neither available.
    """
    try:
        df = nasdaqdatalink.get(SFP_DATASET, ticker=ticker, start_date=start, end_date=end)
        if df is not None and not df.empty:
            df = df.copy()
            if "date" in df.columns:
                df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            # Common column name in SFP is 'dividends'
            if "dividends" in df.columns:
                s = df["dividends"].astype(float)
                return s[s > 0.0]
    except Exception:
        pass

    # Fallback: if SEP had a dividends column
    try:
        if fallback_prices is not None and "dividends" in fallback_prices.columns:
            s = fallback_prices["dividends"].astype(float)
            s = s[s > 0.0]
            s.index.name = "date"
            return s
    except Exception:
        pass

    # Nothing found
    return pd.Series(dtype=float)


def compute_real_cashflows(df: pd.DataFrame, end_date: str, prices: Dict[str, pd.Series]) -> Tuple[List[Tuple[pd.Timestamp, float]], float]:
    """Build real cashflows list and compute final portfolio market value at end_date."""
    # Build cashflows from transactions
    cfs: List[Tuple[pd.Timestamp, float]] = []

    # We will value current positions at end
    # Compute net positions from non-cash-equivalent symbols
    positions = df.groupby("Ticker")["Quantity"].sum().reindex(prices.keys()).fillna(0.0)

    for _, row in df.iterrows():
        d = pd.to_datetime(row["Trade Date"])  # already parsed earlier, but safe
        typ = str(row["Type"]).upper()
        amt = float(row["Amount USD"])

        if typ in BUY_TYPES:
            # Buy: money leaves you (negative), Amount is typically negative already
            cfs.append((d, float(amt if amt < 0 else -abs(amt))))
        elif typ in SELL_TYPES:
            # Sell: money comes to you (positive)
            cfs.append((d, float(amt if amt > 0 else abs(amt))))
        elif typ in DIV_TYPES or typ in INTEREST_TYPES:
            # Dividends/interest: cash inflow
            cfs.append((d, float(abs(amt))))
        elif typ in DEPOSIT_TYPES:
            # External deposit: from your perspective, negative cash flow (you pay in)
            cfs.append((d, -abs(amt)))
        elif typ in WITHDRAWAL_TYPES:
            # External withdrawal: positive cash flow (you receive)
            cfs.append((d, abs(amt)))
        else:
            # Unknown/other types are ignored (or could be logged)
            pass

    # Final portfolio value: sum position * closing price on/after end_date
    end_dt = pd.to_datetime(end_date)
    final_value = 0.0
    for sym, qty in positions.items():
        if abs(qty) < 1e-10:
            continue
        series = prices[sym]
        _, px = next_trading_close(series, end_dt)
        final_value += float(qty) * float(px)

    cfs.append((end_dt, final_value))
    return cfs, final_value


def build_shadow_benchmark_cashflows(
    df: pd.DataFrame,
    bench_prices: pd.Series,
    bench_divs: pd.Series,
    end_date: str,
) -> Tuple[List[Tuple[pd.Timestamp, float]], float]:
    """Mirror your *dollar* buys/sells into benchmark shares; reinvest benchmark dividends.

    - Only your trades produce benchmark trades (deposits/withdrawals still appear in CFs but do not change shares).
    - Dividends of the benchmark are reinvested into more benchmark shares at the dividend date close.
    """
    shares = 0.0
    bench_cfs: List[Tuple[pd.Timestamp, float]] = []

    # Mirror your cashflows as-is (so XIRR timing matches), but only BUY/SELL affect shares
    for _, row in df.iterrows():
        d = pd.to_datetime(row["Trade Date"])
        typ = str(row["Type"]).upper()
        amt = float(row["Amount USD"])  # sign as in CSV
        # Push the cashflow (normalized) same as in real cashflows mapping
        if typ in BUY_TYPES:
            bench_cfs.append((d, float(amt if amt < 0 else -abs(amt))))
            # Update shares: buy with the same dollar amount at benchmark price
            date_used, px = next_trading_close(bench_prices, d)
            dollars = -bench_cfs[-1][1]  # positive dollars invested
            if px > 0:
                shares += dollars / px
        elif typ in SELL_TYPES:
            bench_cfs.append((d, float(amt if amt > 0 else abs(amt))))
            # Reduce shares by the dollar value sold / price
            date_used, px = next_trading_close(bench_prices, d)
            dollars = bench_cfs[-1][1]
            if px > 0:
                shares_sold = dollars / px
                shares -= shares_sold
                shares = max(shares, 0.0)
        elif typ in DIV_TYPES or typ in INTEREST_TYPES:
            bench_cfs.append((d, float(abs(amt))))
        elif typ in DEPOSIT_TYPES:
            bench_cfs.append((d, -abs(amt)))
        elif typ in WITHDRAWAL_TYPES:
            bench_cfs.append((d, abs(amt)))
        else:
            # Ignore others
            pass

    # Reinvest benchmark dividends (divs is per-share amount indexed by date)
    for exdt, div_ps in bench_divs.items():
        # buy extra shares using current holdings' dividend cash at ex-date close
        date_used, px = next_trading_close(bench_prices, pd.to_datetime(exdt))
        if px > 0:
            add_shares = shares * float(div_ps) / px
            shares += add_shares

    # Final value
    end_dt = pd.to_datetime(end_date)
    _, last_px = next_trading_close(bench_prices, end_dt)
    final_val = shares * last_px
    bench_cfs.append((end_dt, final_val))
    return bench_cfs, final_val


def xirr(cashflows: List[Tuple[pd.Timestamp, float]]) -> float:
    dates = [d for d, _ in cashflows]
    amts = [a for _, a in cashflows]
    return float(npf.xirr(amts, dates))


# --------------------------------- Main ------------------------------------ #

def main():
    args = parse_args()
    setup_api()

    # Load and normalize CSV
    raw = pd.read_csv(args.csv)

    # Expect columns like your example
    required = ["Trade Date", "Type", "Ticker", "Security Type", "Price USD", "Quantity", "Amount USD"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {missing}")

    df = raw.copy()
    df["Trade Date"] = pd.to_datetime(df["Trade Date"])  # parse dates
    df["Type"] = normalize_types(df["Type"])            # normalize types
    df["Ticker"] = df["Ticker"].fillna("").astype(str).str.upper().str.strip()
    df["Security Type"] = normalize_types(df["Security Type"])  # e.g., MONEY MARKET, STOCK

    # Filter cash equivalents by ticker
    df = df[~df["Ticker"].isin(CASH_EQUIVALENTS)].copy()
    # Optionally filter by Security Type flag
    if args.ignore_cash_by_security_type:
        df = df[df["Security Type"] != "MONEY MARKET"].copy()

    # Keep a working set for pricing (only symbols with actual positions/trades)
    traded_equities = (
        df.loc[df["Type"].isin(list(BUY_TYPES | SELL_TYPES)), "Ticker"].dropna().unique().tolist()
    )

    # Build list of price tickers = traded_equities + benchmark
    tickers = sorted(set([t for t in traded_equities if t] + [args.benchmark]))

    # Determine start/end
    start_date = df["Trade Date"].min().strftime("%Y-%m-%d")
    end_date = args.end_date

    # Load prices for all tickers
    prices = load_sep_prices(tickers, start=start_date, end=end_date)

    # Load benchmark dividends per share (try SFP; fallback to SEP dividends column if present)
    # For fallback, we need the full SEP DF, not just series. Re-query once for benchmark only.
    try:
        sep_full = nasdaqdatalink.get(SEP_DATASET, ticker=args.benchmark, start_date=start_date, end_date=end_date)
        if sep_full is not None and not sep_full.empty:
            sep_full = sep_full.copy()
            if "date" in sep_full.columns:
                sep_full.set_index("date", inplace=True)
            sep_full.sort_index(inplace=True)
        else:
            sep_full = None
    except Exception:
        sep_full = None

    bench_divs = load_dividends_per_share(args.benchmark, start=start_date, end=end_date, fallback_prices=sep_full)

    # REAL cashflows & final value
    # Use only rows that affect portfolio economics; Money Market already filtered out.
    # We'll pass the subset of df that includes all potential cashflow types (buys, sells, divs, deposits, withdrawals)
    real_cfs, real_final_value = compute_real_cashflows(df, end_date=end_date, prices={k: v for k, v in prices.items() if k != args.benchmark})

    # SHADOW benchmark cashflows & final value
    # Use same df (mirrors all CFs) but shares are only changed on BUY/SELL rows
    bench_cfs, bench_final_value = build_shadow_benchmark_cashflows(
        df=df,
        bench_prices=prices[args.benchmark],
        bench_divs=bench_divs,
        end_date=end_date,
    )

    # Compute XIRR for both
    real_irr = xirr(real_cfs)
    bench_irr = xirr(bench_cfs)

    # Optional dump
    if args.dump_cashflows:
        out = (
            pd.DataFrame({"date": [d for d, _ in real_cfs], "amount": [a for _, a in real_cfs]})
            .assign(portfolio="real")
        )
        out2 = (
            pd.DataFrame({"date": [d for d, _ in bench_cfs], "amount": [a for _, a in bench_cfs]})
            .assign(portfolio="benchmark")
        )
        pd.concat([out, out2], ignore_index=True).to_csv(args.dump_cashflows, index=False)

    # Report
    print("=== Money-weighted return (XIRR) Comparison ===")
    print(f"Period: {start_date} -> {end_date}")
    print(f"Benchmark: {args.benchmark}")
    print("")
    print(f"Your portfolio (XIRR): {real_irr:.2%}")
    print(f"Benchmark total-return (XIRR): {bench_irr:.2%}")
    print("")
    diff = real_irr - bench_irr
    print(f"Active outperformance (XIRR): {diff:+.2%}")


if __name__ == "__main__":
    main()
