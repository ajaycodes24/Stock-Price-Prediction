import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Fetch Wipro stock data for the last 5 years
wipro_data = yf.download('WIPRO.NS', start='2020-01-01', end='2024-12-24')

# Display the first few rows of the data
print(wipro_data.head())

# Use 'Close' price for prediction
wipro_data = wipro_data[['Close']]

# Normalize the stock prices
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(wipro_data)

# Create training and testing datasets (80% train, 20% test)
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[0:train_size], scaled_data[train_size:]

# Create features and labels for training
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Define time_step
time_step = 60

# Use last 60 days of data to predict the next day's price
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, predictions))  # Evaluate model

# Predict for the next day
predicted_price = model.predict(X_test[-1].reshape(1, -1))
predicted_price = predicted_price.reshape(-1, 1)  # Reshape to 2D array (n_samples, n_features)
predicted_price = scaler.inverse_transform(predicted_price)
print(f"Predicted stock price for the next day: ₹{predicted_price[0][0]}")

# Plotting the historical prices and predictions
plt.plot(wipro_data.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='Actual Price')
plt.plot(wipro_data.index[-len(predictions):], scaler.inverse_transform(predictions.reshape(-1, 1)), color='red', label='Predicted Price')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price (₹)')
plt.title('Wipro Stock Price Prediction')
plt.show()
