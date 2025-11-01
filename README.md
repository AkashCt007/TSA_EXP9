# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 1/11/2025

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("IMDB Top 250 Movies (1).csv")

# Prepare data
data = data[['year', 'rating']].copy()
data['year'] = pd.to_numeric(data['year'], errors='coerce')
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
data.dropna(inplace=True)

# Group by year and average the ratings
data = data.groupby('year')['rating'].mean().reset_index()
data.columns = ['Year', 'Rating']

# Convert 'Year' column to datetime format
data['Year'] = pd.to_datetime(data['Year'], format='%Y')

# Set 'Year' column as index
data.set_index('Year', inplace=True)

# ARIMA model
def arima_model(data, target_variable, order):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=len(test_data))

    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data', color='blue')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data', color='green')
    plt.plot(test_data.index, forecast, label='Forecasted Data', color='red')
    plt.xlabel('Year')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.grid()
    plt.show()

    print("Root Mean Squared Error (RMSE):", rmse)

# Run ARIMA model on IMDb ratings
arima_model(data, 'Rating', order=(2,1,2))

```

### OUTPUT:
<img width="1343" height="665" alt="image" src="https://github.com/user-attachments/assets/9e7cbc80-3e1c-4a59-beb5-fa1a43ce5c02" />



### RESULT:
Thus the program run successfully based on the ARIMA model using python.
