from abc import ABC, abstractmethod
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model
from statsmodels.tsa.statespace.structural import UnobservedComponents
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from itertools import product
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import base64
import io
import optuna

# Suppress all warnings from statsmodels
warnings.filterwarnings("ignore", category=Warning, module='statsmodels')
plt.switch_backend('Agg') 

def create_model(model_type, data, symbol_name):
    if model_type == 'AR':
        return AR_model(data, symbol_name)
    elif model_type == 'ARIMA':
        return ARIMA_model(data, symbol_name)
    elif model_type == 'SARIMA':
        return SARIMA_model(data, symbol_name)
    elif model_type == 'HWES':
        return HWES_model(data, symbol_name)
    elif model_type == 'ARCH':
        return ARCH_model(data, symbol_name)
    elif model_type == 'UCM':
        return UCM_model(data, symbol_name)
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
    def __init__(self, data, symbol_name):
        super().__init__(data = data['Close'], open = data['Open'], high = data['High'], low = data['Low'], volume = data['Volume'], symbol_name = symbol_name)
        self.trained_model = None
        self.model_type = 'Autoregressive'
        self.stationary = False
        self.show_backtest = True

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

            best_mse = float('inf')
            best_params = 'n', 1
            self.last_val_predictions = None  # To store the last validation split predictions
            self.last_val_index = None  # To store the index of the last validation split

            # Perform grid search with cross-validation on the training set
            # Choose the best params based on MSE score
            n_splits = 2
            tscv = TimeSeriesSplit(n_splits=n_splits)  # Time series cross-validation
            warnings.filterwarnings("ignore")
            for trend, lags in product(trends, lags_range):
                mse_sum = 0
                val_predictions = None
                val_index = None
                for train_index, val_index in tscv.split(data):
                    train_split, val_split = data.iloc[train_index], data.iloc[val_index]
                    try:
                        model = AutoReg(train_split.values, lags=lags, trend=trend).fit()
                        predictions = model.predict(start=len(train_split), end=len(train_split) + len(val_split) - 1)
                        mse = mean_squared_error(val_split, predictions)
                        mse_sum += mse
                        # Store the predictions and index of the last validation split
                        if val_index[0] == len(data) - len(val_split):
                           val_predictions = predictions
                           val_index = val_index
                    except Exception as e:
                        continue
                
                # Average mse score across folds
                avg_mse = mse_sum / n_splits
                
                # Update best parameters if better mse found
                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_params = (trend, lags)
                    self.last_val_predictions = val_predictions
                    self.last_val_index = val_index

            best_trend, best_lags = best_params
        
            print(f"Best MSE score: {best_mse:.4f}")
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
            if self.last_val_predictions is not None and self.last_val_index is not None:
               self.last_val_predictions = np.exp(self.last_val_predictions)

        # Plot the data
        # Create date range for forecasted data
        forecast_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
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
        ax1.plot(forecast_dates, forecast_prices, label='Forecasted Prices', color='black', linewidth=1.5, linestyle = '-')
        ax1.set_title(f'Model: {self.model_type} \n Symbol: {self.symbol_name}', weight = 'bold', fontsize = 16)
        ax1.set_ylabel('Price', weight = 'bold', fontsize = 15)
       
        ax1.grid(True, alpha = 0.3)
        # Plot the last validation split predictions if available
        if self.show_backtest:
            if self.last_val_predictions is not None and self.last_val_index is not None:
                ax1.plot(self.data.index[self.last_val_index], self.last_val_predictions, label='Backtest Predictions', color='dimgray', linewidth=1.5, linestyle='-')
        ax1.legend(loc='upper left')
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
    
class ARIMA_model(Model):    
    def __init__(self, data, symbol_name):
        super().__init__(data = data['Close'], open = data['Open'], high = data['High'], low = data['Low'], volume = data['Volume'], symbol_name = symbol_name)
        self.trained_model = None
        self.model_type = 'Autoregressive Integrated Moving Average'
        self.stationary = False
        self.show_backtest = True
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
            p_values = range(0, 3)
            d_values = range(0, 2)
            q_values = range(0, 3)
            trends = ['c', 't', 'ct', [0, 0, 1, 0], [0, 0, 0, 1]]
            
            best_mse = float('inf') 
            best_params = 't', 0, 0, 0
            self.last_val_predictions = None  # To store the last validation split predictions
            self.last_val_index = None  # To store the index of the last validation split

            # Perform grid search with cross-validation on the training set
            # Choose the best params based on MSE score
            n_splits = 2
            tscv = TimeSeriesSplit(n_splits=n_splits)  # Time series cross-validation
            warnings.filterwarnings("ignore")
            for d in d_values:
                if d == 1:
                    trends = ['t', [0,0,1,0], [0,0,0,1]]
                elif d == 2:
                    trends = [[0,0,1,0], [0,0,0,1]]
                elif d == 0:
                    trends = ['c', 't', 'ct', [0, 0, 1, 0], [0, 0, 0, 1]]
                for trend in trends:
                    for p in p_values:   
                        for q in q_values:
                            mse_sum = 0
                            val_predictions = None
                            val_index = None
                            for train_index, val_index in tscv.split(data):
                                train_split, val_split = data.iloc[train_index], data.iloc[val_index]
                                try:
                                    model = ARIMA(train_split, order=(p,d,q), trend = trend).fit()
                                    predictions = model.predict(start=len(train_split), end=len(train_split) + len(val_split) - 1)
                                    mse = mean_squared_error(val_split, predictions)
                                    mse_sum += mse
                                    # Store the predictions and index of the last validation split
                                    if val_index[0] == len(data) - len(val_split):
                                        val_predictions = predictions
                                        val_index = val_index
                                except Exception as e:
                                    continue

                            avg_mse = mse_sum / n_splits
                            if avg_mse < best_mse:
                                best_mse = avg_mse
                                best_params = (trend, p, d, q)
                                self.last_val_predictions = val_predictions
                                self.last_val_index = val_index

            best_trend, best_p, best_d, best_q = best_params
            print(f"Best MSE score: {best_mse:.4f}")
            print(f"Best parameters: trend={best_trend}, p={best_p}, d={best_d}, q={best_q}")
        
            # Fit the best model on the entire dataset 
            try:
                self.trained_model = ARIMA(data, order=(best_p,best_d,best_q), trend=best_trend).fit()
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
            if self.last_val_predictions is not None and self.last_val_index is not None:
                self.last_val_predictions = np.exp(self.last_val_predictions)

        # Plot the data
        # Create date range for forecasted data
        forecast_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
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
        ax1.plot(forecast_dates, forecast_prices, label='Forecasted Prices', color='black', linewidth=1.5, linestyle = '-')
        ax1.set_title(f'Model: {self.model_type} \n Symbol: {self.symbol_name}', weight = 'bold', fontsize = 16)
        ax1.set_ylabel('Price', weight = 'bold', fontsize = 15)
        ax1.grid(True, alpha = 0.3)

        # Plot the last validation split predictions if available
        if self.show_backtest:
            if self.last_val_predictions is not None and self.last_val_index is not None:
                ax1.plot(self.data.index[self.last_val_index], self.last_val_predictions, label='Backtest Predictions', color='dimgray', linewidth=1.5, linestyle='-')

        ax1.legend(loc='upper left')
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
    
class SARIMA_model(Model):    
    def __init__(self, data, symbol_name):
        super().__init__(data = data['Close'], open = data['Open'], high = data['High'], low = data['Low'], volume = data['Volume'], symbol_name = symbol_name)
        self.trained_model = None
        self.model_type = 'Seasonal Autoregressive Integrated Moving Average'
        self.stationary = False
        self.show_backtest = True

    def check_stationarity(self, series, alpha=0.05):
        series = series.dropna()
        result = adfuller(series)
        p_value = result[1]
        self.stationary = p_value < alpha
        return self.stationary 

    def log_transform(self, series):
        return np.log(series).dropna()
    
    def objective(self, trial):
        p = trial.suggest_int('p', 0, 3)
        d = trial.suggest_int('d', 0, 2)
        q = trial.suggest_int('q', 0, 3)
        P = trial.suggest_int('P', 0, 2)
        D = trial.suggest_int('D', 0, 1)
        Q = trial.suggest_int('Q', 0, 2)
        s = trial.suggest_categorical('s', [7, 12, 30, 52])  # Example seasonal periods
        trend = trial.suggest_categorical('trend', ['c', 't', 'ct'])

        mse_sum = 0
        n_splits = 2
        best_val_predictions = None
        best_val_index = None
        tscv = TimeSeriesSplit(n_splits=n_splits)
        # Retrieve the current best_mse from user attributes
        best_mse = trial.user_attrs.get('best_mse', float('inf'))

        for train_index, val_index in tscv.split(self.data):
            train_split, val_split = self.data.iloc[train_index], self.data.iloc[val_index]
            try:
                model = SARIMAX(train_split, order=(p, d, q), seasonal_order=(P, D, Q, s), trend=trend).fit(disp = False)
                predictions = model.predict(start=len(train_split), end=len(train_split) + len(val_split) - 1)
                mse = mean_squared_error(val_split, predictions)
                mse_sum += mse
                # Store the predictions and index for the last validation split
                if len(val_index) > 0 and val_index[0] == len(self.data) - len(val_split):
                    best_val_predictions = predictions
                    best_val_index = val_index
            except Exception as e:
                return float('inf')  # Return a very low value if an error occurs

        avg_mse = mse_sum / n_splits
        if avg_mse<best_mse:
            # Store the best predictions and index within the trial object for later retrieval
            trial.set_user_attr("best_mse", avg_mse)
            trial.set_user_attr("best_val_predictions", best_val_predictions)
            trial.set_user_attr("best_val_index", best_val_index)

        return avg_mse
    
    def train(self):
            warnings.filterwarnings("ignore")
            # Check stationarity and apply log transformation if needed
            if not self.check_stationarity(self.data):
                print("Series is not stationary. Applying log transformation...")
                self.data = self.log_transform(self.data)
                    
             # Create an Optuna study
            study = optuna.create_study(direction='minimize')
            # Define an initial best_mse as infinity
            initial_best_mse = float('inf')
            # Define the objective function with an initial best_mse
            def objective_with_initial_best_mse(trial):
                trial.set_user_attr("best_mse", initial_best_mse)
                return self.objective(trial)

            study.optimize(objective_with_initial_best_mse, n_trials=30)  # Number of trials can be adjusted


            best_params = study.best_params
            best_mse = study.best_value  # Access custom attributes returned from objective function

            # Retrieve the best trial
            best_trial = study.best_trial

            # Store the best validation predictions and index
            self.last_val_predictions = best_trial.user_attrs["best_val_predictions"]
            self.last_val_index = best_trial.user_attrs["best_val_index"]

            
            print(f"Best MSE score: {best_mse:.4f}")
            print(f"Best parameters: {best_params}")

            # Fit the best model on the entire dataset
            try:
                self.trained_model = SARIMAX(self.data, order=(best_params['p'], best_params['d'], best_params['q']),
                                            seasonal_order=(best_params['P'], best_params['D'], best_params['Q'], best_params['s']),
                                            trend=best_params['trend']).fit(disp = False)
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
            self.data = np.exp(self.data)
            forecast_prices = np.exp(forecast_prices)
            if self.last_val_predictions is not None and self.last_val_index is not None:
                self.last_val_predictions = np.exp(self.last_val_predictions)

        # Plot the data
        # Create date range for forecasted data
        forecast_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
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
        ax1.plot(forecast_dates, forecast_prices, label='Forecasted Prices', color='black', linewidth=1.5, linestyle = '-')
        ax1.set_title(f'Model: {self.model_type} \n Symbol: {self.symbol_name}', weight = 'bold', fontsize = 16)
        ax1.set_ylabel('Price', weight = 'bold', fontsize = 15)
        ax1.grid(True, alpha = 0.3)

        # Plot the last validation split predictions if available
        if self.show_backtest:
            if self.last_val_predictions is not None and self.last_val_index is not None:
                ax1.plot(self.data.index[self.last_val_index], self.last_val_predictions, label='Backtest Predictions', color='dimgray', linewidth=1.5, linestyle='-')

        ax1.legend(loc='upper left')
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

class HWES_model(Model):    
    def __init__(self, data, symbol_name):
        super().__init__(data = data['Close'], open = data['Open'], high = data['High'], low = data['Low'], volume = data['Volume'], symbol_name = symbol_name)
        self.trained_model = None
        self.model_type = 'Holt-Winters Exponential Smoothing'
        self.stationary = False
        self.show_backtest = True

    def check_stationarity(self, series, alpha=0.05):
        series = series.dropna()
        result = adfuller(series)
        p_value = result[1]
        self.stationary = p_value < alpha
        return self.stationary 

    def log_transform(self, series):
        return np.log(series).dropna()
    
    def objective(self, trial):

        # Define range of parameters
        trend = trial.suggest_categorical('trend', ['add', 'mul', 'additive', 'multiplicative'])
        seasonal = trial.suggest_categorical('seasonal', ['add', 'mul', 'additive', 'multiplicative'])
        seasonal_periods = trial.suggest_categorical('seasonal_periods', [7, 12, 30, 52])  # Example seasonal periods
        initialization_method = trial.suggest_categorical('initialization_method', [None, 'estimated', 'heuristic', 'legacy-heuristic'])
        use_boxcox = trial.suggest_categorical('use_boxcox', [True, False])
        mse_sum = 0
        n_splits = 2
        best_val_predictions = None
        best_val_index = None
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
         # Retrieve the current best_mse from user attributes
        best_mse = trial.user_attrs.get('best_mse', float('inf'))
        for train_index, val_index in tscv.split(self.data):
            train_split, val_split = self.data.iloc[train_index], self.data.iloc[val_index]
            try:
                model = ExponentialSmoothing(train_split, trend = trend, seasonal = seasonal, 
                                             seasonal_periods = seasonal_periods,
                                             initialization_method = initialization_method,
                                             use_boxcox = use_boxcox).fit()
                predictions = model.predict(start=len(train_split), end=len(train_split) + len(val_split) - 1)
                mse = mean_squared_error(val_split, predictions)
                mse_sum += mse
                # Store the predictions and index for the last validation split
                if len(val_index) > 0 and val_index[0] == len(self.data) - len(val_split):
                    best_val_predictions = predictions
                    best_val_index = val_index
            except Exception as e:
                return float('inf')  # Return a very low value if an error occurs

        avg_mse = mse_sum / n_splits
        # Store the best predictions and index within the trial object for later retrieval
        if avg_mse < best_mse:
            trial.set_user_attr('best_mse', avg_mse)
            trial.set_user_attr("best_val_predictions", best_val_predictions)
            trial.set_user_attr("best_val_index", best_val_index)
        return avg_mse
    
    def train(self):
        warnings.filterwarnings("ignore")
        # Check stationarity and apply log transformation if needed
        if not self.check_stationarity(self.data):
            print("Series is not stationary. Applying log transformation...")
            self.data = self.log_transform(self.data)
                
        # Create an Optuna study
        optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress most of the output
        study = optuna.create_study(direction='minimize')
        # Define an initial best_mse as infinity
        initial_best_mse = float('inf')
        # Define the objective function with an initial best_mse
        def objective_with_initial_best_mse(trial):
            trial.set_user_attr("best_mse", initial_best_mse)
            return self.objective(trial)

        study.optimize(objective_with_initial_best_mse, n_trials=75)  # Number of trials can be adjusted

        best_params = study.best_params
        best_mse = study.best_value  # Access custom attributes returned from objective function

        # Retrieve the best trial
        best_trial = study.best_trial

        # Store the best validation predictions and index
        self.last_val_predictions = best_trial.user_attrs["best_val_predictions"]
        self.last_val_index = best_trial.user_attrs["best_val_index"]

        
        print(f"Best MSE score: {best_mse:.4f}")
        print(f"Best parameters: {best_params}")

        # Fit the best model on the entire dataset
        try:
            self.trained_model = ExponentialSmoothing(self.data, trend = best_params['trend'], seasonal = best_params['seasonal'],
                                                        seasonal_periods = best_params['seasonal_periods'], 
                                                        initialization_method = best_params['initialization_method'],
                                                        use_boxcox = best_params['use_boxcox']).fit()
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
            self.data = np.exp(self.data)
            forecast_prices = np.exp(forecast_prices)
            if self.last_val_predictions is not None and self.last_val_index is not None:
                self.last_val_predictions = np.exp(self.last_val_predictions)

        # Plot the data
        # Create date range for forecasted data
        forecast_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
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
        ax1.plot(forecast_dates, forecast_prices, label='Forecasted Prices', color='black', linewidth=1.5, linestyle = '-')
        ax1.set_title(f'Model: {self.model_type} \n Symbol: {self.symbol_name}', weight = 'bold', fontsize = 16)
        ax1.set_ylabel('Price', weight = 'bold', fontsize = 15)
        ax1.grid(True, alpha = 0.3)

        # Plot the last validation split predictions if available
        if self.show_backtest:
            if self.last_val_predictions is not None and self.last_val_index is not None:
                ax1.plot(self.data.index[self.last_val_index], self.last_val_predictions, label='Backtest Predictions', color='dimgray', linewidth=1.5, linestyle='-')

        ax1.legend(loc='upper left')
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
    
class ARCH_model(Model):    
    def __init__(self, data, symbol_name):
        super().__init__(data = data['Close'], open = data['Open'], high = data['High'], low = data['Low'], volume = data['Volume'], symbol_name = symbol_name)
        self.trained_model = None
        self.model_type = 'Generalized AutoRegressive Conditional Heteroskedasticity'
        self.stationary = False
        self.show_backtest = True

    def check_stationarity(self, series, alpha=0.05):
        series = series.dropna()
        result = adfuller(series)
        p_value = result[1]
        self.stationary = p_value < alpha
        return self.stationary 

    def log_transform(self, series):
        return np.log(series).dropna()
    
    def objective(self, trial):

        # Define range of parameters
        rescale = trial.suggest_categorical('rescale', [True, False])
        min_lag = 1
        max_lag = int(np.sqrt(len(self.data))) if len(self.data) >= 20 else len(self.data) // 2  # Ensure a practical upper bound for small datasets
        lags = trial.suggest_int('lags', min_lag, max_lag)
        p = trial.suggest_int('p', 1, 5)
        q = trial.suggest_int('q', 1, 5)
        vol = trial.suggest_categorical('vol', ['GARCH', 'ARCH', 'EGARCH', 'FIGARCH', 
                                                'APARCH', 'HARCH'])
        mean = trial.suggest_categorical('mean', ['LS', 'AR', 
                                                  'ARX', 'HAR', 'HARX'])
        mse_sum = 0
        n_splits = 2
        best_val_predictions = None
        best_val_index = None
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Retrieve the current best_mse from user attributes
        best_mse = trial.user_attrs.get('best_mse', float('inf'))


        for train_index, val_index in tscv.split(self.data):
            train_split, val_split = self.data.iloc[train_index], self.data.iloc[val_index]
            try:
                model = arch_model(train_split, p = p, q = q, mean = mean, lags = lags, 
                                   rescale = rescale, vol = vol).fit(disp = 'off')
                predictions = model.forecast(horizon=len(val_split)).mean.values[-1, :]
                mse = mean_squared_error(val_split, predictions)
                mse_sum += mse
                # Store the predictions and index for the last validation split
                if len(val_index) > 0 and val_index[0] == len(self.data) - len(val_split):
                    best_val_predictions = predictions
                    best_val_index = val_index
            except Exception as e:
                return float('inf')  # Return a very low value if an error occurs

        avg_mse = mse_sum / n_splits
        if avg_mse<best_mse:
            # Store the best predictions and index within the trial object for later retrieval
            trial.set_user_attr("best_mse", avg_mse)
            trial.set_user_attr("best_val_predictions", best_val_predictions)
            trial.set_user_attr("best_val_index", best_val_index)

        return avg_mse
    
    def train(self):
            warnings.filterwarnings("ignore")
            # Check stationarity and apply log transformation if needed
            if not self.check_stationarity(self.data):
                print("Series is not stationary. Applying log transformation...")
                self.data = self.log_transform(self.data)
                    
            # Create an Optuna study
            optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress most of the output
            study = optuna.create_study(direction='minimize')

            # Define an initial best_mse as infinity
            initial_best_mse = float('inf')
            # Define the objective function with an initial best_mse
            def objective_with_initial_best_mse(trial):
                trial.set_user_attr("best_mse", initial_best_mse)
                return self.objective(trial)

            study.optimize(objective_with_initial_best_mse, n_trials=75)  # Number of trials can be adjusted

            best_params = study.best_params
            best_mse = study.best_value  # Access custom attributes returned from objective function

            # Retrieve the best trial
            best_trial = study.best_trial

            # Store the best validation predictions and index
            self.last_val_predictions = best_trial.user_attrs["best_val_predictions"]
            self.last_val_index = best_trial.user_attrs["best_val_index"]

            
            print(f"Best MSE score: {best_mse:.4f}")
            print(f"Best parameters: {best_params}")

            # Fit the best model on the entire dataset
            try:
                self.trained_model = arch_model(self.data, p = best_params['p'], q = best_params['q'], 
                                                mean = best_params['mean'],lags =  best_params['lags'],
                                                rescale =  best_params['rescale'], vol = best_params['vol']).fit(disp = 'off')
                print(f'Model training successful')
            except Exception as e:
                print(f'Model training failed with the error message: {e}')
            
    def forecast(self, forecast_days):
        #Forecast next forecast_period days
        start = len(self.data)
        end = start + forecast_days - 1
        forecast_prices = self.trained_model.forecast(horizon=forecast_days).mean.values[-1, :]
       # Reverse log transformation if applied
        if not self.stationary:
            self.data = np.exp(self.data)
            forecast_prices = np.exp(forecast_prices)
            if self.last_val_predictions is not None and self.last_val_index is not None:
                self.last_val_predictions = np.exp(self.last_val_predictions)

        # Plot the data
        # Create date range for forecasted data
        forecast_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
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
        ax1.plot(forecast_dates, forecast_prices, label='Forecasted Prices', color='black', linewidth=1.5, linestyle = '-')
        ax1.set_title(f'Model: {self.model_type} \n Symbol: {self.symbol_name}', weight = 'bold', fontsize = 16)
        ax1.set_ylabel('Price', weight = 'bold', fontsize = 15)
        ax1.grid(True, alpha = 0.3)

        # Plot the last validation split predictions if available
        if self.show_backtest:
            if self.last_val_predictions is not None and self.last_val_index is not None:
                ax1.plot(self.data.index[self.last_val_index], self.last_val_predictions, label='Backtest Predictions', color='dimgray', linewidth=1.5, linestyle='-')

        ax1.legend(loc='upper left')
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
    
class UCM_model(Model):    
    def __init__(self, data, symbol_name):
        super().__init__(data = data['Close'], open = data['Open'], high = data['High'], low = data['Low'], volume = data['Volume'], symbol_name = symbol_name)
        self.trained_model = None
        self.model_type = 'Unobserved Components Model'
        self.stationary = False
        self.show_backtest = True

    def check_stationarity(self, series, alpha=0.05):
        series = series.dropna()
        result = adfuller(series)
        p_value = result[1]
        self.stationary = p_value < alpha
        return self.stationary 

    def log_transform(self, series):
        return np.log(series).dropna()
    
    def objective(self, trial):

        # Define range of parameters
        level = trial.suggest_categorical('level', [True, False])
        trend = trial.suggest_categorical('trend', [True, False])
        seasonal = trial.suggest_categorical('seasonal', [7, 12, 30, 52]) 
        cycle = trial.suggest_categorical('cycle', [True, False])  
        damped_trend = trial.suggest_categorical('damped_trend', [True, False]) 
        irregular = trial.suggest_categorical('irregular', [True, False])  
        autoregressive = trial.suggest_int('autoregressive', 1, 10)

        mse_sum = 0
        n_splits = 2
        best_val_predictions = None
        best_val_index = None
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Retrieve the current best_mse from user attributes
        best_mse = trial.user_attrs.get('best_mse', float('inf'))


        for train_index, val_index in tscv.split(self.data):
            train_split, val_split = self.data.iloc[train_index], self.data.iloc[val_index]
            try:
                model = UnobservedComponents(train_split, level = level, trend = trend, seasonal = seasonal, cycle = cycle, 
                                             damped_trend = damped_trend, irregular = irregular, autoregressive = autoregressive).fit(disp = False)
                predictions = model.get_forecast(steps=len(val_split)).predicted_mean
                mse = mean_squared_error(val_split, predictions)
                mse_sum += mse
                # Store the predictions and index for the last validation split
                if len(val_index) > 0 and val_index[0] == len(self.data) - len(val_split):
                    best_val_predictions = predictions
                    best_val_index = val_index
            except Exception as e:
                return float('inf')  # Return a very low value if an error occurs

        avg_mse = mse_sum / n_splits
        if avg_mse<best_mse:
            # Store the best predictions and index within the trial object for later retrieval
            trial.set_user_attr("best_mse", avg_mse)
            trial.set_user_attr("best_val_predictions", best_val_predictions)
            trial.set_user_attr("best_val_index", best_val_index)

        return avg_mse
    
    def train(self):
            warnings.filterwarnings("ignore")
            # Check stationarity and apply log transformation if needed
            if not self.check_stationarity(self.data):
                print("Series is not stationary. Applying log transformation...")
                self.data = self.log_transform(self.data)
                    
            # Create an Optuna study
            study = optuna.create_study(direction='minimize')

            # Define an initial best_mse as infinity
            initial_best_mse = float('inf')
            # Define the objective function with an initial best_mse
            def objective_with_initial_best_mse(trial):
                trial.set_user_attr("best_mse", initial_best_mse)
                return self.objective(trial)

            study.optimize(objective_with_initial_best_mse, n_trials=30)  # Number of trials can be adjusted

            best_params = study.best_params
            best_mse = study.best_value  # Access custom attributes returned from objective function

            # Retrieve the best trial
            best_trial = study.best_trial

            # Store the best validation predictions and index
            self.last_val_predictions = best_trial.user_attrs["best_val_predictions"]
            self.last_val_index = best_trial.user_attrs["best_val_index"]

            
            print(f"Best MSE score: {best_mse:.4f}")
            print(f"Best parameters: {best_params}")

            # Fit the best model on the entire dataset
            try:
                self.trained_model =  UnobservedComponents(self.data, level = best_params['level'], trend = best_params['trend'], seasonal = best_params['seasonal'], 
                                                           cycle = best_params['cycle'], damped_trend = best_params['damped_trend'], irregular = best_params['irregular'],
                                                           autoregressive = best_params['autoregressive']).fit(disp = False)
                print(f'Model training successful')
            except Exception as e:
                print(f'Model training failed with the error message: {e}')
            
    def forecast(self, forecast_days):
        #Forecast next forecast_period days
        start = len(self.data)
        end = start + forecast_days - 1
        forecast_prices = self.trained_model.get_forecast(steps=forecast_days).predicted_mean
       # Reverse log transformation if applied
        if not self.stationary:
            self.data = np.exp(self.data)
            forecast_prices = np.exp(forecast_prices)
            if self.last_val_predictions is not None and self.last_val_index is not None:
                self.last_val_predictions = np.exp(self.last_val_predictions)

        # Plot the data
        # Create date range for forecasted data
        forecast_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
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
        ax1.plot(forecast_dates, forecast_prices, label='Forecasted Prices', color='black', linewidth=1.5, linestyle = '-')
        ax1.set_title(f'Model: {self.model_type} \n Symbol: {self.symbol_name}', weight = 'bold', fontsize = 16)
        ax1.set_ylabel('Price', weight = 'bold', fontsize = 15)
        ax1.grid(True, alpha = 0.3)

        # Plot the last validation split predictions if available
        if self.show_backtest:
            if self.last_val_predictions is not None and self.last_val_index is not None:
                ax1.plot(self.data.index[self.last_val_index], self.last_val_predictions, label='Backtest Predictions', color='dimgray', linewidth=1.5, linestyle='-')

        ax1.legend(loc='upper left')
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