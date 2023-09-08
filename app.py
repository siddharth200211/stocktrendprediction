import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from keras.models import load_model

st.title('Stock trend prediction')

user_input=st.text_input('enter stock ticker','AAPL')
data=yf.download(user_input,start='2010-01-01',end='2022-12-30')

#describing data
st.subheader('data from 2010 - 2022')
st.write(data.describe())

#Visulization
st.subheader('Closing price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(data.Close, 'b')
st.pyplot(fig)

st.subheader('Closing price vs Time chart with 100ma')
ma100=data.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(data.Close, 'b')
plt.plot(ma100, 'r')
st.pyplot(fig)

st.subheader('Closing price vs Time chart with 100ma & 200ma')
ma200=data.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(data.Close, 'b')
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
st.pyplot(fig)

#spliting data into test and training
data_training=pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing=pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)


#load my model
model=load_model('keras_model.h5')


past_100_days=data_training.tail(100)
final_df = pd.concat([past_100_days, data_training], ignore_index=True)

input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])


x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)


scaler=scaler.scale_

scale_factor=1/scaler
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

#final plot
st.subheader('prediction vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label="Original price")
plt.plot(y_predicted,'r',label="Predicted price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)