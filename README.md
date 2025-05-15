# Stock_price_prediction
AI-based stock price prediction using sentiment analysis
## Description  
This project predicts stock price movements using AI techniques by combining historical financial data and sentiment analysis from news headlines.  ":
It uses **Random Forest Classifier** for prediction and **VADER** for sentiment analysis, providing insights into whether a stock's price will go **Up** or **Down**.  

---

## Features  
- Predicts stock movements based on historical data and news sentiment.  
- Integrates Yahoo Finance for stock data and NewsAPI for news headlines.  
- Uses machine learning (Random Forest) for accurate predictions.  
- Displays prediction accuracy and recent news headlines.  
- Supports both **US and Indian stocks**.  

---

## Installation  

1. **Clone the repository:**  
   git clone https://github.com/Siddharth-mishra-123/Stock_price_prediction.git
   
   
   cd Stock_price_prediction
   
   
   Set up a virtual environment (optional):

python -m venv env

env\Scripts\activate  # On Windows

source env/bin/activate  # On Linux/Mac

Install the required packages:

pip install streamlit yfinance requests scikit-learn cachetools vaderSentiment
---
## Usage
Update the API Key:

Replace the api_key variable inside the stock_price_prediction.py file with your NewsAPI key (if you have). otherwise remain it as it is

Run the Streamlit app:
streamlit run stock_price_prediction.py

Enter a stock ticker:

US Stocks: AAPL, MSFT, etc.

Indian Stocks: RELIANCE.NS, TCS.NS, etc.
---
## Technologies Used

Python: Programming Language

Streamlit: Web Framework

Yahoo Finance API: Stock Data

NewsAPI: News Headlines

Scikit-learn: Machine Learning

VADER Sentiment Analysis: Text Sentiment Analysis

Cachetools: Caching API responses
---
## License
This project is licensed under the MIT License.
