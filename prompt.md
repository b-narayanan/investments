# Task : Total Returns Analysis
I’ve been investing for a few years in my brokerage account. I’d like to know what my annual rate of return over the history of investing is. I’d also like to know if I’d have done better by just buying the broad market.

# Details
I want to evaluate how replacing each of my stock purchases with the same dollar amount of a broad market benchmark, say, VOO, would have performed. The folder @transactions/ contains CSV files with transactions over the history of the account. For financial data, let's use Nasdaq Data Link -- I have access to the Sharadar equities bundle (and would like to get familiar with the API).

# Solution
I want to solve this problem with Python, the way a human working without AI would. What do I mean? I'd want to write small functions one at a time, test them in the REPL, and then piece them together in a larger script. For example, first loading up the data from disk, then fetching all necessary reference data from Nasdaq/Sharadar, then writing a function to simulate a simpler portfolio etc. Generate a markdown file, @notebook.py with a solution that contains Python snippets following this script.

# Subtleties
Reinvest all dividends.
Ignore my buying of any money market funds like VMFXX, since that's me holding money as cash essentially.
