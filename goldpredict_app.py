import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime as dt
from datetime import timedelta


# additional customization for the web (semacem CSS nya)
st.markdown(
  """
  <style>.main{
    background-color: #F8F7F3;
  }
  """,
  unsafe_allow_html = True
)


header = st.container()
dataset = st.container()
gold_plot = st.container()
gold_predict = st.container()
predictor = st.container()

@st.cache
def getData(filename):
  df = pd.read_csv(filename)
  return df

with header:
  df = getData("https://raw.githubusercontent.com/jmkho/ML_CSV/main/gold.csv")
  st.title("Simple Gold Prediction Program using Linear Regression")
  st.text("In this project we are making a simple program to predict \ngold price in the future using Linear Regression.")

with dataset:
  st.subheader("Dataset information used to create the model")
  st.write(df.head())

with gold_plot:
  st.subheader("Previous Gold Closing Price")
  start_date = st.date_input('Start date', value=None, min_value=None, max_value=None, key=None)
  end_date = st.date_input('End date', value=None, min_value=None, max_value=None, key=None)
  
  if start_date < end_date:
    
    data_symbol = 'GLD'
    data = yf.Ticker(data_symbol)
    data_df = data.history(period='id', start=start_date, end=end_date)

    st.text("Gold closing price starting from %s until %s" % (start_date, end_date))
    st.line_chart(data_df.Close)
  else:
    st.error("Error: Start date must fall before end date.")

with gold_predict:
  df_new = df
  df_new = df_new.dropna()

  # define explanatory variables 
  df_new['S_3'] = df_new['Close'].rolling(window=3).mean()
  df_new['S_9'] = df_new['Close'].rolling(window=9).mean()
  df_new['next_day_price'] = df_new['Close'].shift(-1)

  df_new = df_new.dropna()

  # independent variable
  X = df_new[['S_3', 'S_9']]
  y = df_new['next_day_price']

  t = .8
  t = int(t*len(df_new))

  # train dataset
  X_train = X[:t]
  y_train = y[:t]

  # test dataset
  X_test = X[t:]
  y_test = y[t:]

  linear = LinearRegression().fit(X_train, y_train)
  st.subheader("Linear Regression Model: ")
  st.text("Gold ETF Price (y) = \n%.2f * 3 Days Moving Average (x1) \ + %.2f * 9 Days Moving Average (x2) \ + %.2f (constant)" % (linear.coef_[0], linear.coef_[1], linear.intercept_))

with predictor:
  curr_date = dt.datetime.now() + timedelta(days=5)

  # getting the data for prediction
  p_data = yf.download('GLD', '2008-06-01', curr_date, auto_adjust = True)
  p_data['S_3'] = p_data['Close'].rolling(window=3).mean()
  p_data['S_9'] = p_data['Close'].rolling(window=9).mean()

  p_data = p_data.dropna()

  # predicting the price 
  p_data['predicted_gold_price'] = linear.predict(p_data[['S_3', 'S_9']])
  p_data['signal'] = np.where(p_data.predicted_gold_price.shift(1) < p_data.predicted_gold_price, "Buy", "No Position")

  # print the forecast
  p_data.tail(1)[['signal', 'predicted_gold_price']]


