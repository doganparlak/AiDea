from abc import ABC, abstractmethod
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from itertools import product
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import base64
import io

# Suppress all warnings from statsmodels
warnings.filterwarnings("ignore", category=Warning, module='statsmodels')
plt.switch_backend('Agg') 

def create_model(model_type, data, symbol_name):
    if model_type == 'AR':
        return AR_model(data, symbol_name)
    
class Model(ABC):
    def __init__(self, data, open, high, low, volume, symbol_name):
        self.data = data
        self.open = open
        self.volume = volume
        self.high = high
        self.low = low
        self.symbol_name = symbol_name

    @abstractmethod
    def train(self):
        """
        Abstract method to train the model.
        """
        pass
    
    @abstractmethod
    def forecast(self, forecast_days):
        """
        Abstract method to make predictions using the trained model.
        """
        pass


class AR_model(Model):

    def prepare_data(self, data):
        data = data.dropna()
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        if not data.index.freq:
            # Attempt to infer the frequency
            inferred_freq = pd.infer_freq(data.index)
            if inferred_freq:
                data.index.freq = inferred_freq
            else:
                # Handle the case where frequency cannot be inferred
                # For example, you might decide to use a default frequency or handle this as an exception
                print("Unable to infer frequency for the datetime index.")

        data.index.length = len(data)
        return data
    
    def __init__(self, data, symbol_name):
        data = self.prepare_data(data)
        super().__init__(data = data['Close'], open = data['Open'], high = data['High'], low = data['Low'], volume = data['Volume'], symbol_name = symbol_name)
        self.trained_model = None
        self.model_type = 'Auto Regressive'
        self.stationary = False

    def check_stationarity(self, series, alpha=0.05):
        series = series.dropna()
        result = adfuller(series)
        p_value = result[1]
        self.stationary = p_value < alpha
        return self.stationary 

    def log_transform(self, series):
        return np.log(series).dropna()
    
    def train(self):
            # Check stationarity and apply log transformation if needed
            data = self.data
            if not self.check_stationarity(self.data):
                print("Series is not stationary. Applying log transformation...")
                data = self.log_transform(self.data)
                    
            # Define parameter grid for tuning
            trends = ['n', 'c', 't', 'ct']
            min_lag = 1
            max_lag = int(np.sqrt(len(data))) if len(data) >= 20 else len(data) // 2  # Ensure a practical upper bound for small datasets
            lags_range = range(min_lag, max_lag + 1) 

            best_r2 = -float('inf') 
            best_params = 'n', 1

            # Perform grid search with cross-validation on the training set
            # Choose the best params based on R2 score
            n_splits = 3
            tscv = TimeSeriesSplit(n_splits=n_splits)  # Time series cross-validation
            warnings.filterwarnings("ignore")
            for trend, lags in product(trends, lags_range):
                r2_sum = 0
                for train_index, val_index in tscv.split(data):
                    train_split, val_split = data.iloc[train_index], data.iloc[val_index]
                    try:
                        model = AutoReg(train_split.values, lags=lags, trend=trend).fit()
                        predictions = model.predict(start=len(train_split), end=len(train_split) + len(val_split) - 1)
                        r2 = r2_score(val_split, predictions)
                        r2_sum += r2
                    except Exception as e:
                        continue
                
                # Average R2 score across folds
                avg_r2 = r2_sum / n_splits
                
                # Update best parameters if better R2 found
                if avg_r2 > best_r2:
                    best_r2 = avg_r2
                    best_params = (trend, lags)

            best_trend, best_lags = best_params
        
            print(f"Best R2 score: {best_r2:.4f}")
            print(f"Best parameters: trend={best_trend}, lags={best_lags}")
        
            # Fit the best model on the entire dataset 
            try:
                self.trained_model = AutoReg(data, lags=best_lags, trend=best_trend).fit()
                print(f'Model training successful')
            except Exception as e:
                print(f'Model training failed with the error message: {e}')
            
    def forecast(self, forecast_days):
        #Forecast next forecast_period days
        start = len(self.data)
        end = start + forecast_days - 1
        forecast_prices = self.trained_model.predict(start=start, end=end)
       # Reverse log transformation if applied
        if not self.stationary:
            forecast_prices = np.exp(forecast_prices)

        # Plot the data
        # Create date range for forecasted data
        forecast_dates = pd.date_range(start=self.data.index[-2] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        # Create figure and axis
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Create candlestick data
        candlestick_data = pd.DataFrame({
            'Date': self.data.index,
            'Open': self.open,
            'Close': self.data,
            'High': self.high,
            'Low': self.low
        })
        # Plot the candlestick data with decreased transparency
        for idx, row in candlestick_data.iterrows():
            date_num = mdates.date2num(row['Date'])
            if row['Close'] >= row['Open']:
                color = 'green'
                lower = row['Open']
                height = row['Close'] - row['Open']
            else:
                color = 'red'
                lower = row['Close']
                height = row['Open'] - row['Close']
            
            # Draw high and low lines (wicks) outside the rectangle
            ax1.vlines(date_num, row['Low'], lower, color=color, alpha=0.5, linewidth=0.5)
            ax1.vlines(date_num, lower + height, row['High'], color=color, alpha=0.5, linewidth=0.5)
            
            # Draw the rectangle (candlestick body)
            ax1.add_patch(mpatches.Rectangle((date_num - 0.5, lower), 1, height, edgecolor=color, facecolor=color, alpha=1, linewidth=1))
        # Plot the price data
        ax1.plot(self.data.index, self.data, label='Historical Data', color='gray', linewidth=1, alpha=0.6)
        ax1.plot(forecast_dates, forecast_prices, label='Forecasted Prices', color='black', linewidth=1, linestyle = 'dashdot')
        ax1.set_title(f'Model: {self.model_type} \n Symbol: {self.symbol_name}', weight = 'bold', fontsize = 16)
        ax1.set_ylabel('Price', weight = 'bold', fontsize = 15)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha = 0.3)

        # Plot the volume data
        volume_colors = np.where(self.data.diff() >= 0, 'green', 'red')
        ax2.bar(self.data.index, self.volume, color=volume_colors, alpha=0.6)
        ax2.set_ylabel('Volume', weight = 'bold', fontsize = 15)
        ax2.set_xlabel('Time', weight = 'bold', fontsize = 15)
        ax2.grid(True, alpha = 0.3)
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        #plt.show()
        plt.close(fig)  # Close the plot to free up resources

        return plot_data
    


