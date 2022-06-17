import streamlit as st
import yfinance as yf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import datetime as dt
from datetime import timedelta
from dateutil.relativedelta import relativedelta

header = st.container()
dataset = st.container()
gold_plot = st.container()
expl_var = st.container()
gold_model = st.container()
gold_pred = st.container()
gold_model_result = st.container()
gold_prediction = st.container()


with header:
  st.title("Simple Gold Prediction Program using Linear Regression")
  st.text("In this project we are making a simple Machine Learning program to predict \ngold price using Linear Regression.")

with dataset:
  st.header("Dataset information used to create the model")
  st.write("We are using gold price data from 10 years ago (2012-2022) provided by Yahoo Finance with 'GLD' as the ticker object. Here we provide a table consisting the tail information of the dataset.")
  
  start = dt.datetime.now() - relativedelta(years=10)
  end = dt.datetime.now() - relativedelta(1)
  df = yf.download('GLD', start, end, auto_adjust = True)
  st.dataframe(df.tail(10))

with gold_plot:
  st.header("Previous Gold Closing Price")
  st.write("You can plot the closing gold price by choosing the starting date and ending date.")
  start_date = st.date_input('Start date', value=None, min_value=None, max_value=None, key=None)
  end_date = st.date_input('End date', value=None, min_value=None, max_value=None, key=None)
  
  if start_date < end_date:
    data_symbol = 'GLD'
    data = yf.Ticker(data_symbol)
    data_df = data.history(period='id', start=start_date, end=end_date)

    st.write("Gold closing price starting from %s until %s" % (start_date, end_date))
    st.line_chart(data_df.Close)

  else:
    st.error("Error: Start date must fall before end date.")

with gold_model:
  df = df[['Close']]
  df.dropna()

  # moving price average for the past 3 days 
  df['past_3'] = df['Close'].rolling(window=3).mean()

  # moving price average for the past 9 days 
  df['past_9'] = df['Close'].rolling(window=9).mean()

  # price of the Gold ETF for prediction
  df['next_day_price'] = df['Close'].shift(-1)

  df = df.dropna()

  # define dependent variable 
  X = df[['past_3', 'past_9']]

  # define independent variable
  y = df['next_day_price']

  # splitting the first 80% data for training and the remaining 20% for testing
  t = int(df.shape[0]*0.8)

  # train dataset
  X_train = X[0:t]
  y_train = y[0:t]

  # test dataset
  X_test = X[t:]
  y_test = y[t:]

  linear = LinearRegression().fit(X_train, y_train)

with gold_pred: 
  # Predicting the Gold ETF prices
  predicted_price = linear.predict(X_test)
  predicted_price = pd.DataFrame(predicted_price, index=y_test.index, columns=['price'])

  # Goodness of the Fit 
  r2_score = linear.score(X[t:], y[t:]) * 100
  r2 = float("{0:.2f}".format(r2_score))

  # Sharpe Ratio
  gold = pd.DataFrame()

  gold['price'] = df[t:]['Close']
  gold['predicted_price_next_day'] = predicted_price
  gold['actual_price_next_day'] = y_test

  # generate daily percentage change of gold price
  gold['gold_returns'] = gold['price'].pct_change().shift(-1)

  # signalling '1' if next day's price = buy
  gold['signal'] = np.where(gold.predicted_price_next_day.shift(1) < gold.predicted_price_next_day,1,0)

  # calculate strategy returns by multiplying daily percentage change and signal
  gold['strategy_returns'] = gold['gold_returns'] * gold.signal 

  sharpe = gold['strategy_returns'].mean()/(gold['strategy_returns'].std())*(252**0.5)

with expl_var:
  st.header("Explanatory Variables")
  st.write("Explanatory variable is the features we use to predict the gold price. \n")
  st.write("Here, we use moving price averages for the past 3 days and 9 days as the independent variables.\n")
  st.write("The predicted or dependent variable will be the next day price.")
  st.dataframe(df.tail())

with gold_model_result:
  st.markdown("---")

  st.header("Model Results")

  st.subheader("Linear Regression Coefficient and Intercept")
  st.write("3 days moving average (X1) Coefficient = %.2f" % linear.coef_[0])
  st.write("9 days moving average (X2) Coefficient = %.2f" % linear.coef_[1])
  st.write("Intercept (constant): %.2f" % linear.intercept_)
  st.subheader("Linear Regression Equation: ")
  st.latex("Y = (X1 * %.2f) + (X2 * %.2f) + %.2f" % (linear.coef_[0], linear.coef_[1], linear.intercept_))

  st.subheader("Predicted Gold Price")
  st.line_chart(predicted_price)

  st.write("**Model Accuracy Score: %.2f**" % linear.score(X_test, y_test))
  st.write("**Goodness of the Fit (R Square): %.2f**" % r2)
  st.write("**Sharpe ratio: %.2f**" % sharpe)


with gold_prediction:
  st.markdown("---")

  # Prediction
  p_start = dt.datetime.now() - relativedelta(years=10)
  curr_date = dt.datetime.now()

  # getting the data for prediction
  p_data = yf.download('GLD', p_start, curr_date, auto_adjust = True)

  # moving price average for the past 3 days 
  p_data['past_3'] = p_data['Close'].rolling(window=3).mean()

  # moving price average for the past 0 days 
  p_data['past_9'] = p_data['Close'].rolling(window=9).mean()

  p_data = p_data.dropna()

  # predicting the price 
  p_data['predicted_gold_price'] = linear.predict(p_data[['past_3', 'past_9']])
  p_data['signal'] = np.where(p_data.predicted_gold_price.shift(0) < p_data.predicted_gold_price, "Buy", "No Position")

  st.subheader('Click the button to predict gold price:')
  if st.button('Predict'):
    # print the forecast
    p_data.tail(1)[['signal', 'predicted_gold_price']]

  st.caption("Note: Data is taken from Yahoo Finance, which has opening time at 09:30 AM EST (UTC -4) and currency in USD.")

