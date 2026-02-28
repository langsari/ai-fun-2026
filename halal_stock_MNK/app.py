from flask import Flask, render_template, request
from data_fetcher import get_stock_data
from ml_predictor import ai_analyze

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        ticker = request.form["ticker"].upper()

        stock_data = get_stock_data(ticker)
        sector = stock_data["sector"]

        status, confidence, reasons = ai_analyze(stock_data, sector)

        result = {
            "ticker": ticker,
            "sector": sector,
            "status": status,
            "confidence": round(confidence * 100, 2),
            "reasons": reasons
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
