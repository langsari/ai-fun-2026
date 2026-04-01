from flask import Flask, render_template, request
from data_fetcher import get_stock_data
from ml_predictor import ai_analyze
<<<<<<< HEAD
import pandas as pd
=======

>>>>>>> ac81be44a1a42f92e104a3a72a4f72d908e48d58
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
<<<<<<< HEAD
    error = None
    
    df = pd.read_csv("data/stock_dataset.csv")
=======
>>>>>>> ac81be44a1a42f92e104a3a72a4f72d908e48d58

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

<<<<<<< HEAD
    return render_template(
    "index.html",
    result=result,
    error=error,
    dataset=df.to_dict(orient="records")
)
=======
    return render_template("index.html", result=result)
>>>>>>> ac81be44a1a42f92e104a3a72a4f72d908e48d58

if __name__ == "__main__":
    app.run(debug=True)
