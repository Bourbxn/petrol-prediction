import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st

st.title('Oil Price Prediction')

user_input = st.text_input('Enter Date (format: Y-M-d)', '2011-02-01')
df = pd.read_csv("crude-oil-price.csv")
df_origin = df

# Prediction
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

model = LinearRegression().fit(X_train,y_train)
predicted_price = model.predict(X_test)
predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])
df['predictedPrice'] = predicted_price

df = df.drop(columns='MA3', axis=6).copy()
df = df.drop(columns='MA9', axis=6).copy()
df = df.dropna()

# User Input
df_index = df.index
index = df['date'] == user_input
result = df_index[index]

# Display value
st.subheader(f'Value of {user_input}')
try:
    st.write(df.loc[result.tolist()[0], ['date', 'price', 'predictedPrice']])
except:
    st.write("No data")

# Plotting
st.subheader('Prediction vs Actual Price')
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
st.write(df)