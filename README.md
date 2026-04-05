
## **System Overview**
Your system is a hybrid engine that combines **"Rule-based Shariah Screening"** (fixed religious criteria) with **"AI Consensus"** (machine learning intelligence) to determine if a stock is Halal or Haram.

---

## **1. Phase 1: The Learning Process (Training Phase)**
Before the system can analyze a stock, the AI needs to go to "school" to understand what makes a stock Halal.
* **The Lesson:** The system reads historical data from `stock_dataset.csv`.
* **The Experts (`train_model.py`):** It trains **5 different AI models** (Random Forest, SVM, Logistic Regression, KNN, and Naive Bayes) so they can recognize financial patterns.
* **The Memory:** Once trained, these "brains" are saved in the `model/` folder as `.pkl` files for instant use.

---

## **2. Phase 2: User Input**
Everything starts at the web interface (`app.py`).
* The user enters a **Ticker Symbol** (e.g., `AAPL`, `TSLA`, or `PTT`).
* The system uses **Caching (`lru_cache`)**; if someone else already searched for that stock recently, it shows the result instantly without recalculating.

---

## **3. Phase 3: The Messenger (Data Fetching)**
The file `data_fetcher.py` acts as a messenger to the global stock market.
* **The Source:** It uses the `yfinance` library to pull real-time data from Yahoo Finance.
* **The Data:** It grabs the company’s **Sector**, **Interest Income**, **Total Debt**, **Total Assets**, and **Total Revenue**.
* **The Safety Net:** It includes "Error Handling" logic—if a specific number is missing, it prevents the system from crashing by assigning a default value.

---

## **4. Phase 4: The Iron Screen (Shariah Rules)**
Before the AI even looks at the stock, it must pass the "Rule-based" gatekeeper in `shariah_rules.py`.
* **Business Check:** If the company is involved in **Haram sectors** (Banks, Alcohol, Gambling, Pork, Tobacco), it is rejected immediately.
* **Financial Ratio Check:** The system calculates the "Gold Standard" ratios:
    * **Interest Income Ratio:** Must be less than **5%**.
    * **Debt Ratio:** Interest-bearing debt must be less than **30% (or 33%)** of Total Assets.
* **The Verdict:** If the stock fails these rules, it is marked **HARAM** immediately, and the process stops here to ensure religious integrity.

---

## **5. Phase 5: The Council of Jury (AI Consensus)**
If the stock passes the initial rules, it moves to the "AI Council" in `ml_predictor.py`.
* **Voting System:** All 5 AI models analyze the financial ratios simultaneously.
* **Majority Rule:** The system uses a **Consensus Logic (3 out of 5)**. If at least 3 models agree that the stock is Halal, it passes.
* **Confidence Level:** The system calculates a percentage score showing how sure the AI is about its decision.

---

## **6. Phase 6: Final Result (Presentation)**
Finally, `app.py` gathers all the data and displays it on the `index.html` dashboard.
* **Status:** Clear **HALAL** or **HARAM** badge.
* **Reasons:** A detailed breakdown of why it passed or failed (e.g., "Debt exceeds 30%").
* **Comparison Table:** A transparent table showing exactly how each of the 5 AI models voted.

---

### **System Workflow Summary**

| Step | Component | Primary Function |
| :--- | :--- | :--- |
| **1. Input** | `app.py` | Receives the stock ticker from the user. |
| **2. Fetching** | `data_fetcher.py` | Pulls live financial data from Yahoo Finance. |
| **3. Screening** | `shariah_rules.py` | Checks the 5% and 30% Shariah thresholds. |
| **4. AI Prediction** | `ml_predictor.py` | 5 AI models vote to confirm the status. |
| **5. Output** | `index.html` | Displays the final verdict, reasons, and confidence. |

