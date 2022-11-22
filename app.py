import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st

st.title('Oil Price Prediction')

user_input = st.text_input('Enter Date (format: Y-M-d)', '2011-02-01')
df = pd.read_csv("crude-oil-price.csv")
df_origin = df

# Prediction with Linear Equation
df_origin['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
X_origin = df['date']
y_origin = df['price'].values.reshape(-1, 1)
model2 = LinearRegression().fit(X_origin.values.reshape(-1,1),y_origin)
predictions = model2.predict(X_origin.values.astype(float).reshape(-1, 1))

# Prediction with Past Price
df['date'] = pd.to_datetime(df['date'])
df['MA3'] = df['price'].shift(1).rolling(window=3).mean()
df['MA9']= df['price'].shift(1).rolling(window=9).mean()
df = df.dropna()
X = df[['MA3','MA9']]
y = df['price']

training = 0.7
t = int(training*len(df))

X_train = X[:t]
y_train = y[:t]
X_test = X[t:]
y_test = y[t:]

model1 = LinearRegression().fit(X_train,y_train)
predicted_price = model1.predict(X_test)
predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])
df['predictedPrice'] = predicted_price

df = df.drop(columns='MA3', axis=6).copy()
df = df.drop(columns='MA9', axis=6).copy()
df = df.dropna()

# User Input
input_format = pd.to_datetime(user_input, format='%Y-%m-%d')
user_pred = model2.predict(pd.DataFrame([input_format]).values.astype(float).reshape(-1, 1))

# Display value
st.subheader(f'Price of {user_input}')
st.write('Predict Price :', user_pred[0,0])

# Linear Regression
st.subheader('Prediction with Linear Equation')
fig3 = plt.figure(figsize=(12,6))    
plt.scatter(X_origin, y_origin, label='actual price', color='green')
plt.plot(X_origin, predictions, label='prediction', linewidth=3)
plt.xlabel('date')
plt.ylabel('price')
plt.legend()
st.pyplot(fig3)

# Plotting 
st.subheader('Prediction Present Price with Past Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(predicted_price)
plt.plot(y_test)
plt.legend(['Predicted Price','Actual Price'])
plt.ylabel("Crude Oil Prices:")
st.pyplot(fig2)

# Visualization
st.subheader('Oil Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df_origin.price)
st.pyplot(fig)

# Describing Data
st.subheader('Data table')
st.write(df_origin)