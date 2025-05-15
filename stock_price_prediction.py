# Import necessary libraries
import numpy as np
import yfinance as yf
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from cachetools import TTLCache
import streamlit as st
from datetime import datetime, timedelta

# Set up a cache to minimize API calls
cache = TTLCache(maxsize=100, ttl=12 * 60 * 60)  # Cache for 12 hours

# Function to get news data and sentiment analysis
def get_news_sentiment(ticker):
    api_key = "a9795a9fad124c12a5e1d5552fd748d9"
    # Check cache first to minimize API calls
    if ticker in cache:
        return cache[ticker]
    company_name= ticker.split('.')[0]
    # Make the API call
    url = f"https://newsapi.org/v2/everything?q={company_name}&apiKey={api_key}&pageSize=5&language=en&sortBy=publishedAt"
    try:
        response = requests.get(url, timeout=10)
        news_data = response.json()

        # Check for rate limit error
        if news_data.get("status") == "error" and news_data.get("code") == "rateLimited":
            return ["Rate limit exceeded. Try again later."]

        # Process articles if available
        if news_data.get("status") == "ok" and news_data.get("articles"):
            headlines = [article['title'] for article in news_data['articles']]
            cache[ticker] = headlines  # Store in cache
            return headlines
        else:
            return ["No news articles found. Please check your data source."]
    
    except requests.RequestException as e:
        return [f"Error fetching news: {str(e)}"]

# Sentiment analysis function
def analyze_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    avg_sentiment = np.mean(sentiment_scores)
    return avg_sentiment

# Predict stock movement
def predict_stock_movement(data):
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

    # Features and labels
    X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = data['Target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report, model

# Streamlit app
st.title("Stock Price Prediction with Sentiment Analysis")
st.write("Note: For Indian stocks, use the format <SYMBOL>.NS (e.g., RELIANCE.NS for Reliance Industries)")


# User input
ticker = st.text_input("Enter the stock ticker (e.g., AAPL for US stocks, RELIANCE.NS for Indian stocks):", "RELIANCE.NS")


if ticker:
    # Fetch data
    try:
        data = yf.download(ticker, period='1y')
        st.write(f"Stock data for {ticker}")
        st.dataframe(data.tail())

        # Predict stock movement
        accuracy, report, model = predict_stock_movement(data)
        st.write(f"Prediction Accuracy: {accuracy:.2%}")
        st.text("Classification Report:")
        st.text(report)

        # News sentiment
        headlines = get_news_sentiment(ticker)
        if "Rate limit exceeded" not in headlines[0]:
            st.write("Latest News Headlines:")
            for headline in headlines:
                st.write(f"- {headline}")

            sentiment = analyze_sentiment(headlines)
            st.write(f"Average Sentiment Score: {sentiment:.2f}")

            # Combine sentiment with prediction
            sentiment_label = "Positive" if sentiment > 0 else "Negative"
            st.write(f"Overall Sentiment: {sentiment_label}")

            # Final stock movement prediction
            if sentiment > 0:
                st.write("Predicted Stock Movement: 📈 Up")
            else:
                st.write("Predicted Stock Movement: 📉 Down")
        else:
            st.write(headlines[0])

    except Exception as e:
        st.error(f"Error loading data for ticker {ticker}: {e}")

