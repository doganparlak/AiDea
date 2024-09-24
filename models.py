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
import re
import optuna
from openai import OpenAI
import yaml


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
    elif model_type == 'AI':
        return AI_model(data, symbol_name)

# Read the configuration file at the top of the script
def read_settings(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load the settings file
settings = read_settings('settings.yaml')
# Set your API key
client = OpenAI(
    # This is the default and can be omitted
    api_key= settings['OPENAI-API-KEY']['key'],
)
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
        self.show_backtest = False

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
                #print("Series is not stationary. Applying log transformation...")
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
        
            #print(f"Best MSE score: {best_mse:.4f}")
            #print(f"Best parameters: trend={best_trend}, lags={best_lags}")
        
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

        forecast_prices.iloc[0] = self.data.iloc[-1]
        # Plot the data
        # Create date range for forecasted data
        forecast_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')

        # Convert data to JSON in order to send to the frontend
        result = {
            'historical_data': {
                'dates': self.data.index.strftime('%Y-%m-%d').tolist(),
                'close': self.data.tolist(),
                'open': self.open.tolist(),
                'high': self.high.tolist(),
                'low': self.low.tolist(),
                'volume': self.volume.tolist(),
            },
            'forecasted_data': {
                'dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
                'forecast_prices': forecast_prices.tolist(),
            }
        }
        
        return result
    
class ARIMA_model(Model):    
    def __init__(self, data, symbol_name):
        super().__init__(data = data['Close'], open = data['Open'], high = data['High'], low = data['Low'], volume = data['Volume'], symbol_name = symbol_name)
        self.trained_model = None
        self.model_type = 'Autoregressive Integrated Moving Average'
        self.stationary = False
        self.show_backtest = False
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
                #print("Series is not stationary. Applying log transformation...")
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
            #print(f"Best MSE score: {best_mse:.4f}")
            #print(f"Best parameters: trend={best_trend}, p={best_p}, d={best_d}, q={best_q}")
        
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

        forecast_prices.iloc[0] = self.data.iloc[-1]
        # Plot the data
        # Create date range for forecasted data
        forecast_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')

        # Convert data to JSON in order to send to the frontend
        result = {
            'historical_data': {
                'dates': self.data.index.strftime('%Y-%m-%d').tolist(),
                'close': self.data.tolist(),
                'open': self.open.tolist(),
                'high': self.high.tolist(),
                'low': self.low.tolist(),
                'volume': self.volume.tolist(),
            },
            'forecasted_data': {
                'dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
                'forecast_prices': forecast_prices.tolist(),
            }
        }
        
        return result
    
class SARIMA_model(Model):    
    def __init__(self, data, symbol_name):
        super().__init__(data = data['Close'], open = data['Open'], high = data['High'], low = data['Low'], volume = data['Volume'], symbol_name = symbol_name)
        self.trained_model = None
        self.model_type = 'Seasonal Autoregressive Integrated Moving Average'
        self.stationary = False
        self.show_backtest = False

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
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            # Check stationarity and apply log transformation if needed
            if not self.check_stationarity(self.data):
                #print("Series is not stationary. Applying log transformation...")
                self.data = self.log_transform(self.data)
                    
             # Create an Optuna study
            study = optuna.create_study(direction='minimize')
            # Define an initial best_mse as infinity
            initial_best_mse = float('inf')
            # Define the objective function with an initial best_mse
            def objective_with_initial_best_mse(trial):
                trial.set_user_attr("best_mse", initial_best_mse)
                return self.objective(trial)

            study.optimize(objective_with_initial_best_mse, n_trials=20)  # Number of trials can be adjusted


            best_params = study.best_params
            best_mse = study.best_value  # Access custom attributes returned from objective function

            # Retrieve the best trial
            best_trial = study.best_trial

            # Store the best validation predictions and index
            self.last_val_predictions = best_trial.user_attrs["best_val_predictions"]
            self.last_val_index = best_trial.user_attrs["best_val_index"]

            
            #print(f"Best MSE score: {best_mse:.4f}")
            #print(f"Best parameters: {best_params}")

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

        forecast_prices.iloc[0] = self.data.iloc[-1]
        # Plot the data
        # Create date range for forecasted data
        forecast_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        # Convert data to JSON in order to send to the frontend
        result = {
            'historical_data': {
                'dates': self.data.index.strftime('%Y-%m-%d').tolist(),
                'close': self.data.tolist(),
                'open': self.open.tolist(),
                'high': self.high.tolist(),
                'low': self.low.tolist(),
                'volume': self.volume.tolist(),
            },
            'forecasted_data': {
                'dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
                'forecast_prices': forecast_prices.tolist(),
            }
        }
        
        return result

class HWES_model(Model):    
    def __init__(self, data, symbol_name):
        super().__init__(data = data['Close'], open = data['Open'], high = data['High'], low = data['Low'], volume = data['Volume'], symbol_name = symbol_name)
        self.trained_model = None
        self.model_type = 'Holt-Winters Exponential Smoothing'
        self.stationary = False
        self.show_backtest = False

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
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # Check stationarity and apply log transformation if needed
        if not self.check_stationarity(self.data):
            #print("Series is not stationary. Applying log transformation...")
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

        
        #print(f"Best MSE score: {best_mse:.4f}")
        #print(f"Best parameters: {best_params}")

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

        forecast_prices.iloc[0] = self.data.iloc[-1]
        # Plot the data
        # Create date range for forecasted data
        forecast_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        # Convert data to JSON in order to send to the frontend
        result = {
            'historical_data': {
                'dates': self.data.index.strftime('%Y-%m-%d').tolist(),
                'close': self.data.tolist(),
                'open': self.open.tolist(),
                'high': self.high.tolist(),
                'low': self.low.tolist(),
                'volume': self.volume.tolist(),
            },
            'forecasted_data': {
                'dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
                'forecast_prices': forecast_prices.tolist(),
            }
        }
        
        return result
    
class ARCH_model(Model):    
    def __init__(self, data, symbol_name):
        super().__init__(data = data['Close'], open = data['Open'], high = data['High'], low = data['Low'], volume = data['Volume'], symbol_name = symbol_name)
        self.trained_model = None
        self.model_type = 'Autoregressive Conditional Heteroskedasticity'
        self.stationary = False
        self.show_backtest = False

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
        p = trial.suggest_int('p', 2, 6)
        q = trial.suggest_int('q', 2, 6)
        vol = trial.suggest_categorical('vol', ['GARCH', 'ARCH', 'EGARCH', 'FIGARCH', 
                                                'APARCH', 'HARCH'])
        mean = trial.suggest_categorical('mean', ['LS', 'ARX', 'HAR', 'HARX'])
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
                                   rescale = rescale, vol = vol).fit(disp = 'off', options={'maxiter': 200})
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
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            # Check stationarity and apply log transformation if needed
            if not self.check_stationarity(self.data):
                #print("Series is not stationary. Applying log transformation...")
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

            
            #print(f"Best MSE score: {best_mse:.4f}")
            #print(f"Best parameters: {best_params}")

            # Fit the best model on the entire dataset
            try:
                self.trained_model = arch_model(self.data, p = best_params['p'], q = best_params['q'], 
                                                mean = best_params['mean'],lags =  best_params['lags'],
                                                rescale =  best_params['rescale'], vol = best_params['vol']).fit(disp = 'off', options={'maxiter': 200})
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

        forecast_prices[0] = self.data.iloc[-1]
        # Plot the data
        # Create date range for forecasted data
        forecast_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        # Convert data to JSON in order to send to the frontend
        result = {
            'historical_data': {
                'dates': self.data.index.strftime('%Y-%m-%d').tolist(),
                'close': self.data.tolist(),
                'open': self.open.tolist(),
                'high': self.high.tolist(),
                'low': self.low.tolist(),
                'volume': self.volume.tolist(),
            },
            'forecasted_data': {
                'dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
                'forecast_prices': forecast_prices.tolist(),
            }
        }
        
        return result
    
class UCM_model(Model):    
    def __init__(self, data, symbol_name):
        super().__init__(data = data['Close'], open = data['Open'], high = data['High'], low = data['Low'], volume = data['Volume'], symbol_name = symbol_name)
        self.trained_model = None
        self.model_type = 'Unobserved Components Model'
        self.stationary = False
        self.show_backtest = False

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
        damped_cycle = trial.suggest_categorical('damped_cycle', [True, False]) 
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
                                             damped_cycle = damped_cycle, irregular = irregular, autoregressive = autoregressive).fit(disp = False)
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
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            # Check stationarity and apply log transformation if needed
            if not self.check_stationarity(self.data):
                #print("Series is not stationary. Applying log transformation...")
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

            
            #print(f"Best MSE score: {best_mse:.4f}")
            #print(f"Best parameters: {best_params}")

            # Fit the best model on the entire dataset
            try:
                self.trained_model =  UnobservedComponents(self.data, level = best_params['level'], trend = best_params['trend'], seasonal = best_params['seasonal'], 
                                                           cycle = best_params['cycle'], damped_cycle = best_params['damped_cycle'], irregular = best_params['irregular'],
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

        forecast_prices.iloc[0] = self.data.iloc[-1]
        # Plot the data
        # Create date range for forecasted data
        forecast_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        # Convert data to JSON in order to send to the frontend
        result = {
            'historical_data': {
                'dates': self.data.index.strftime('%Y-%m-%d').tolist(),
                'close': self.data.tolist(),
                'open': self.open.tolist(),
                'high': self.high.tolist(),
                'low': self.low.tolist(),
                'volume': self.volume.tolist(),
            },
            'forecasted_data': {
                'dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
                'forecast_prices': forecast_prices.tolist(),
            }
        }
        
        return result
    
class AI_model(Model):
    def __init__(self, data, symbol_name):
        super().__init__(data = data['Close'], open = data['Open'], high = data['High'], low = data['Low'], volume = data['Volume'], symbol_name = symbol_name)
        self.model_type = 'Artificial Intelligence Model'
        self.data_prompt = ''
        self.data_len = 30
    def train(self):
        # Price data formatted as a string for the prompt, using the last 30 observations
        pass
    def adjust_forecast_prices(self, forecast_prices):
        adjusted_prices = forecast_prices[:]
        for idx in range(len(adjusted_prices) - 1):
            current_price = adjusted_prices[idx]
            next_price = adjusted_prices[idx + 1]
            if next_price > current_price * 1.10:
                adjusted_prices[idx + 1] = current_price * 1.10
            elif next_price < current_price * 0.90:
                adjusted_prices[idx + 1] = current_price * 0.90
        
        # Ensure the forecasted prices remain consistent
        for idx in range(1, len(adjusted_prices)):
            if adjusted_prices[idx] > adjusted_prices[idx - 1] * 1.10:
                adjusted_prices[idx] = adjusted_prices[idx - 1] * 1.10
            elif adjusted_prices[idx] < adjusted_prices[idx - 1] * 0.90:
                adjusted_prices[idx] = adjusted_prices[idx - 1] * 0.90
        
        adjusted_prices[0] = self.data[-1]
        return adjusted_prices
    
    def forecast(self, forecast_days):
        # Function to generate forecast for a chunk
        def generate_forecast(chunk):
            prompt = (
                f"Given the historical price data: {chunk}, "
                f"forecast the next {forecast_days} days of prices. "
                f"Provide a list of predicted prices for the next {forecast_days} days, "
                f"formatted as: price1, price2, price3, ..., price{forecast_days}. "
                f"Ensure realistic fluctuations with moderate changes between consecutive days. "
                f"Give more weight to the most recent observations in your forecasting to better reflect recent market behavior. "
                f"Make sure the first forecasted price is very close to the last price in the historical data to maintain continuity."
            )
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="gpt-3.5-turbo", 
                max_tokens=2048,  # Limit the number of tokens in the response
                temperature=0.2,  # Control the creativity of the model
            )
            decoded_output = response.choices[0].message.content
            extracted_prices = re.findall(r"[-+]?\d*\.\d+|\d+", decoded_output)
            return [float(price) for price in extracted_prices]
        
        # Chunking the data and generating forecasts based on the last 30 observations
        forecast_prices = []
        chunk = ', '.join(str(val) for val in self.data[-self.data_len:])  # Only the last 30 values
        chunk_forecast = generate_forecast(chunk)
        forecast_prices.extend(chunk_forecast)
        # Limit forecast to the desired number of days
        forecast_prices = forecast_prices[1:forecast_days+1]
        # Apply the adjustment logic
        forecast_prices = self.adjust_forecast_prices(forecast_prices)
        # Create date range for forecasted data
        forecast_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=len(forecast_prices), freq='D')    
        # Convert data to JSON in order to send to the frontend
        result = {
            'historical_data': {
                'dates': self.data.index.strftime('%Y-%m-%d').tolist(),
                'close': self.data.tolist(),
                'open': self.open.tolist(),
                'high': self.high.tolist(),
                'low': self.low.tolist(),
                'volume': self.volume.tolist(),
            },
            'forecasted_data': {
                'dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
                'forecast_prices': forecast_prices,
            }
        }
        
        return result