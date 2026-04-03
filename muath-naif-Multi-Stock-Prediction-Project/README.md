# Multi-Stock Price Prediction

This project uses multiple machine learning models to predict future stock prices for four major companies. The models are trained on historical data from Yahoo Finance and evaluated using standard regression metrics.

---

## Stocks

- AAPL (Apple)
- NVDA (NVIDIA)
- MSFT (Microsoft)
- GOOGL (Google)

## Dataset

- Source: Yahoo Finance
- Period: 2015 – 2023
- Feature used: Closing price
- Prediction horizon: 1 year ahead

---

## Models Used

Six models were built and compared in this project:

**1. LSTM (Long Short-Term Memory)**
A recurrent neural network that is good at learning long-term patterns in sequential data. This is the original model used as the main baseline.

**2. SimpleRNN**
A basic recurrent neural network. Simpler than LSTM but can still detect patterns in time-series data.

**3. GRU (Gated Recurrent Unit)**
Similar to LSTM but with a simpler structure. It trains faster and usually achieves similar accuracy.

**4. Bidirectional LSTM**
An LSTM that reads the data in both directions (forward and backward), which helps capture more patterns and usually improves accuracy.

**5. CNN + LSTM**
Combines a Convolutional Neural Network with LSTM. The CNN extracts local patterns from the price sequence, then LSTM learns the time dependencies.

**6. Linear Regression**
A simple statistical model used as a baseline to compare how much the deep learning models improve over a basic approach.

---

## How It Works

1. Download historical closing prices from Yahoo Finance
2. Normalize the data using MinMaxScaler
3. Create sequences of 60 days as input to predict the next day
4. Split data into 80% training and 20% testing
5. Train each model and evaluate on the test set
6. Compare all models using MAE, RMSE, R2, and Accuracy

---

## Results on NVDA

| Model | MAE | RMSE | R2 | Accuracy % |
|---|---|---|---|---|
| Linear Regression | 0.7020 | 0.9492 | 0.9946 | 97.48 |
| GRU | 0.9965 | 1.3203 | 0.9896 | 96.42 |
| CNN + LSTM | 1.5830 | 2.1349 | 0.9728 | 94.32 |
| Bidirectional LSTM | 1.7224 | 2.3276 | 0.9677 | 93.82 |
| LSTM | 1.7722 | 2.3524 | 0.9670 | 93.64 |
| SimpleRNN | 2.6584 | 3.7040 | 0.9182 | 90.46 |

## Results on All Stocks

| Stock | Model | Accuracy % |
|---|---|---|
| AAPL | Linear Regression | 98.61 |
| GOOGL | Linear Regression | 98.27 |
| MSFT | Linear Regression | 98.51 |
| NVDA | Linear Regression | 97.48 |

---

## Evaluation Metrics

- **MAE** (Mean Absolute Error): average error between predicted and real price. Lower is better.
- **RMSE** (Root Mean Square Error): similar to MAE but penalizes larger errors more. Lower is better.
- **R2** (R-squared): measures how well the model fits the data. Closer to 1 is better.
- **Accuracy %**: custom formula used in this project.

```
Accuracy = 100 - (MAE / mean price * 100)
```

---

## Libraries Used

```
yfinance
pandas
numpy
matplotlib
scikit-learn
tensorflow / keras
```

---

## How to Run

Install the required libraries:

```
pip install yfinance tensorflow scikit-learn matplotlib pandas
```

Open the notebook and run all cells:

```
jupyter notebook Multi_Stock.ipynb
```

Or upload to Google Colab and click Runtime > Run all.

---

## Project Structure

```
Multi_Stock.ipynb         # main notebook with all 6 models
Multi_Stock__result.ipynb # notebook with outputs and results
stock_prediction.csv      # saved prediction results
README.md                 # project documentation
```
