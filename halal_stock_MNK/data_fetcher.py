import yfinance as yf

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)

    info = stock.info
    financials = stock.financials
    balance = stock.balance_sheet

    try:
        interest_income = financials.loc["Interest Income"].iloc[0]
    except:
        interest_income = 0

    try:
        total_income = financials.loc["Total Revenue"].iloc[0]
    except:
        total_income = 1  # กันหารศูนย์

    try:
        interest_debt = balance.loc["Total Debt"].iloc[0]
    except:
        interest_debt = 0

    try:
        total_assets = balance.loc["Total Assets"].iloc[0]
    except:
        total_assets = 1

    sector = info.get("sector", "Unknown")

    return {
        "interest_income": abs(interest_income),
        "total_income": abs(total_income),
        "interest_debt": abs(interest_debt),
        "total_assets": abs(total_assets),
        "sector": sector
    }
