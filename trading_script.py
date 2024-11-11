# %%
import backtrader as bt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import math
import backtrader.analyzers as btanalyzers
import multiprocessing


# %%
API_KEY = "uwQtl3txGt5BLbecq7ZbIu0ZbuitCGjc"

# %% [markdown]
# # Loading and Preprocessing Historical Price Data
# 
# This section of the notebook focuses on **loading** and **preprocessing** historical price data for two Exchange-Traded Funds (ETFs): **GLD** (representing gold) and **SLV** (representing silver). The data is sourced from CSV files and prepared for subsequent analysis or modeling.
# 
# ---
# 
# ## **1. Loading Data from CSV Files**
# 
# **Concept Overview:**
# 
# Loading data from CSV (Comma-Separated Values) files is a fundamental step in data analysis. CSV files are widely used for storing tabular data due to their simplicity and compatibility with various tools and programming languages.
# 
# **Purpose:**
# 
# The primary goal of this code is to:
# 
# 1. **Load CSV Files:** Import historical price data for GLD and SLV from their respective CSV files into pandas DataFrames.
# 2. **Convert 'Date' Column to Datetime:** Ensure that the 'Date' columns in both DataFrames are in a datetime format suitable for time series analysis.
# 3. **Set 'Date' as Index:** Assign the 'Date' column as the index of each DataFrame to facilitate efficient data manipulation and analysis based on dates.
# 
# **Benefits:**
# 
# - **Structured Data Loading:** Efficiently imports data into pandas DataFrames, enabling powerful data manipulation and analysis capabilities.
# - **Time Series Readiness:** Converting the 'Date' column to datetime and setting it as the index prepares the data for time series operations such as resampling, rolling statistics, and plotting.
# - **Data Consistency:** Ensures that both datasets (GLD and SLV) are processed uniformly, maintaining consistency in analysis.
# 
# 
# 

# %%
# Load the CSV file into a DataFrame
gld_data = pd.read_csv('data/GLD-Prices.csv')
gld_data['Date'] = pd.to_datetime(gld_data['Date'])  # Ensure the 'Date' column is datetime
gld_data.set_index('Date', inplace=True)  # Set 'Date' as index

# Load the CSV file into a DataFrame
slv_data = pd.read_csv('data/SLV-Prices.csv')
slv_data['Date'] = pd.to_datetime(slv_data['Date'])  # Ensure the 'Date' column is datetime
slv_data.set_index('Date', inplace=True)  # Set 'Date' as index
# %% [markdown]
# # Customized Pandas Data Feed for Backtrader
# 
# This section introduces a **customized data feed** tailored for the Backtrader framework, enabling seamless integration of pandas DataFrames containing historical price data. The customization ensures that the data aligns perfectly with Backtrader's expectations, facilitating accurate backtesting and strategy development.
# 
# ---
# 
# ## **1. Defining the Customized Pandas Data Feed Class**
# 
# ### **a. Overview**
# 
# Backtrader, a versatile Python library for backtesting trading strategies, relies on data feeds to ingest historical market data. While Backtrader provides several built-in data feeds, customizing a data feed allows for greater flexibility and compatibility with diverse data sources, especially when working with pandas DataFrames.
# 
# ### **b. Code Implementation**

# %%

class PandasData(bt.feeds.PandasData):
    """
    Customized Pandas Data Feed for Backtrader.
    """
    params = (
        ('datetime', None),  # Use the DataFrame index as datetime
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', -1),  # No open interest data
    )


# Create data feeds for GLD and SLV
data_gld = PandasData(dataname=gld_data)
data_slv = PandasData(dataname=slv_data)
# data_soyb = PandasData(dataname=soyb_data)
# data_corn = PandasData(dataname=corn_data)
# data_bal = PandasData(dataname=bal_data)

# %% [markdown]
# # SARIMAX Trading Strategy for Backtrader
# 
# This section introduces the **SARIMAXStrategy**, a sophisticated trading strategy implemented using the **Backtrader** framework. The strategy leverages the **Seasonal Autoregressive Integrated Moving Average with eXogenous regressors (SARIMAX)** model to forecast future prices and make informed trading decisions based on these predictions.
# 
# ---
#     
# ## **1. Strategy Overview**
# 
# ### **Concept Overview**
# 
# The **SARIMAXStrategy** is designed to predict future price movements of a financial instrument (e.g., stocks, ETFs) using the SARIMAX model. SARIMAX is an extension of the ARIMA model that incorporates seasonality and exogenous variables, making it suitable for modeling time series data with seasonal patterns and external influences.
# 
# ### **Purpose**
# 
# The primary objectives of the SARIMAXStrategy are:
# 
# 1. **Forecasting:**  
#    Utilize the SARIMAX model to predict future closing prices based on historical data.
# 
# 2. **Automated Trading:**  
#    Execute buy and sell orders automatically based on the forecasted price movements, incorporating risk management through stop-loss and take-profit mechanisms.
# 
# 3. **Optimization:**  
#    Fine-tune model parameters and trading thresholds to enhance strategy performance through Backtrader's optimization capabilities.
# 
# ### **Benefits**
# 
# - **Data-Driven Decisions:**  
#   Relies on statistical models to make informed trading decisions, reducing emotional biases.
# 
# - **Seasonality Handling:**  
#   Effectively captures and leverages seasonal patterns in price data for more accurate forecasts.
# 
# - **Risk Management:**  
#   Incorporates stop-loss and take-profit parameters to manage potential losses and lock in profits.
# 
# - **Flexibility:**  
#   Allows for parameter optimization to adapt the strategy to different market conditions and instruments.
# 
# ---
#     
# ## **2. Strategy Parameters**
# 
# The **SARIMAXStrategy** class inherits from Backtrader's `bt.Strategy` and defines several parameters that control its behavior:# SARIMAX Trading Strategy for Backtrader
# 
# This section introduces the **SARIMAXStrategy**, a sophisticated trading strategy implemented using the **Backtrader** framework. The strategy leverages the **Seasonal Autoregressive Integrated Moving Average with eXogenous regressors (SARIMAX)** model to forecast future prices and make informed trading decisions based on these predictions.
# 
# ---
#     
# ## **1. Strategy Overview**
# 
# ### **Concept Overview**
# 
# The **SARIMAXStrategy** is designed to predict future price movements of a financial instrument (e.g., stocks, ETFs) using the SARIMAX model. SARIMAX is an extension of the ARIMA model that incorporates seasonality and exogenous variables, making it suitable for modeling time series data with seasonal patterns and external influences.
# 
# ### **Purpose**
# 
# The primary objectives of the SARIMAXStrategy are:
# 
# 1. **Forecasting:**  
#    Utilize the SARIMAX model to predict future closing prices based on historical data.
# 
# 2. **Automated Trading:**  
#    Execute buy and sell orders automatically based on the forecasted price movements, incorporating risk management through stop-loss and take-profit mechanisms.
# 
# 3. **Optimization:**  
#    Fine-tune model parameters and trading thresholds to enhance strategy performance through Backtrader's optimization capabilities.
# 
# ### **Benefits**
# 
# - **Data-Driven Decisions:**  
#   Relies on statistical models to make informed trading decisions, reducing emotional biases.
# 
# - **Seasonality Handling:**  
#   Effectively captures and leverages seasonal patterns in price data for more accurate forecasts.
# 
# - **Risk Management:**  
#   Incorporates stop-loss and take-profit parameters to manage potential losses and lock in profits.
# 
# - **Flexibility:**  
#   Allows for parameter optimization to adapt the strategy to different market conditions and instruments.
# 
# ---
#     
# ## **2. Strategy Parameters**
# 
# The **SARIMAXStrategy** class inherits from Backtrader's `bt.Strategy` and defines several parameters that control its behavior:
# 
# ```python
# 
# ('forecast_period', 1),          # Number of days to forecast ahead
# ('model_order', (1, 1, 1)),      # SARIMAX (p,d,q) order
# ('seasonal_order', (1, 1, 1, 7)),  # SARIMAX (P,D,Q,s) seasonal order
# ('threshold', 0.05),              # Threshold for making trades
# ('verbose', False),              # Enable detailed logging
# ('min_data_points', 30),         # Minimum data points required to fit the model
# ('stop_loss', 0.05),              # 5% stop loss
# ('take_profit', 0.1),             # 10% take profit
# ('fit_interval', 5)               # Interval in which to recompute and fit the model
#     
# 

# trading_script.py

import backtrader as bt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging
import math
import backtrader.analyzers as btanalyzers
import multiprocessing

# Define the FinalValue Analyzer
class FinalValue(bt.Analyzer):
    """
    Custom Analyzer to capture the final portfolio value.
    """
    def __init__(self):
        self.final_value = 0.0

    def stop(self):
        """
        Called at the end of the strategy run.
        """
        self.final_value = self.strategy.broker.getvalue()
        print(f"[FinalValue Analyzer] Final Portfolio Value: {self.final_value}")

    def get_analysis(self):
        """
        Returns the analysis results.
        """
        return {'final_value': self.final_value}

# Define the SARIMAXStrategy
class SARIMAXStrategy(bt.Strategy):
    params = (
        ('forecast_period', 1),          # Number of days to forecast ahead
        ('model_order', (1, 1, 1)),      # SARIMAX (p,d,q) order
        ('seasonal_order', (1, 1, 1, 7)),# SARIMAX (P,D,Q,s) seasonal order
        ('threshold', 0.05),             # Threshold for making trades
        ('verbose', False),              # Enable detailed logging
        ('min_data_points', 30),         # Minimum data points required to fit the model
        ('stop_loss', 0.05),             # 5% stop loss
        ('take_profit', 0.25),           # 25% take profit
        ('fit_interval', 5),             # Interval to recompute and fit the model
        ('sizing_method', 'fixed'),      # 'fixed', 'percentage', or 'confidence'
        ('fixed_size', 1000),            # Fixed amount to invest per trade
        ('max_percentage', 0.1),         # Maximum portfolio percentage per trade
        ('confidence_multiplier', 2),    # Multiplier for confidence-based sizing
        ('max_positions', 5),            # Maximum number of buy orders allowed
        ('max_total_size', 10000),       # Maximum total capital allocated to the asset
        ('kelly_b', 1.0),                # Net odds for Kelly Criterion
        ('kelly_p', 0.0),                # Estimated win probability for Kelly Criterion
        ('kelly_multiplier', 0.5),       # Fraction of Kelly to use (e.g., 0.5 for half-Kelly)
    )

    def __init__(self):
        # Reference to the close prices
        self.dataclose = self.datas[0].close

        # Initialize variables to track orders
        self.order = None

        # Model variables
        self.model = None
        self.model_fit = None
        self.counter = 0
        self.share_price = math.inf
        self.curr_share_predicted_price = -math.inf

    def determine_size(self, current_price, predicted_price, confidence=1.0):
        """
        Determine the size of the position based on the sizing method.

        Args:
            current_price (float): Current market price.
            predicted_price (float): Predicted future price.
            confidence (float): Confidence level of the prediction (0 to 1).

        Returns:
            int: Number of shares to buy.
        """
        if self.params.sizing_method == 'fixed':
            # Option 1: Fixed amount per trade
            size = self.params.fixed_size / current_price
        elif self.params.sizing_method == 'percentage':
            # Option 2: Fixed percentage of portfolio
            size = (self.broker.getcash() * self.params.max_percentage) / current_price
        elif self.params.sizing_method == 'confidence':
            # Option 3: Confidence-based sizing
            size = (self.broker.getvalue() * (self.params.max_percentage * confidence)) / current_price
        elif self.params.sizing_method == 'kelly':
            p = self.params.kelly_p
            b = self.params.kelly_b
            kelly_fraction = self.calculate_kelly_fraction(p, b)
            # Optionally, scale Kelly fraction (e.g., half-Kelly to reduce risk)
            kelly_fraction *= self.params.kelly_multiplier  # e.g., 0.5 for half-Kelly
            # Determine position size
            size = (self.broker.getvalue() * kelly_fraction) / current_price
        else:
            # Default to fixed size if unknown method
            size = self.params.fixed_size / current_price

        # Enforce maximum total size
        if self.params.max_total_size:
            current_total_size = self.position.size * current_price
            available_size = self.params.max_total_size - current_total_size
            if available_size <= 0:
                return 0
            size = min(size, available_size / current_price)

        # Enforce minimum size
        size = max(size, 0)

        return int(size)
    def calculate_kelly_fraction(self, p, b):
        """
        Calculate the Kelly fraction based on win probability and win/loss ratio.
        
        Args:
            p (float): Probability of a winning trade.
            b (float): Win/Loss ratio.
        
        Returns:
            float: Kelly fraction (between 0 and 1).
        """
        if b == 0:
            return 0.0
        kelly = (b * p - (1 - p)) / b
        kelly = max(0.0, kelly)  # Ensure non-negative
        return kelly
    def next(self):
        """
        Called for each new data point. Make predictions and execute trades.
        """
        # Less computation by fitting model periodically instead of at every data point
        self.counter += 1
        if self.counter < self.params.fit_interval:
            return
        self.counter = 0
        if self.order:
            # Pending order execution
            return

        # Ensure we have enough data to fit the model
        if len(self.dataclose) < self.params.min_data_points:
            return

        # Extract historical close prices
        historical_data = pd.Series(list(self.dataclose.get(size=self.params.min_data_points)))
        historical_data = historical_data[::-1] 

        # Convert to numeric and drop NaNs
        historical_data = pd.to_numeric(historical_data, errors='coerce').dropna()

        try:
            # Initialize and fit the SARIMAX model
            self.model = SARIMAX(
                historical_data,
                order=self.params.model_order,
                seasonal_order=self.params.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model_fit = self.model.fit(disp=False)

            if self.params.verbose:
                print(self.model_fit.summary())

            # Make a forecast
            forecast = self.model_fit.forecast(steps=self.params.forecast_period)
            predicted_price = forecast.iloc[-1]

            # Current price
            current_price = self.dataclose[0]

            if self.params.verbose:
                print(f"Predicted Price: {predicted_price}, Current Price: {current_price}")

            # Calculate confidence (as an example, using the inverse of the standard error)
            try:
                # Get forecast standard error
                forecast_result = self.model_fit.get_forecast(steps=self.params.forecast_period)
                stderr = forecast_result.se_mean
                confidence = 1 / (1 + stderr.iloc[-1])  # Normalize confidence between 0 and 1
                confidence = max(0.0, min(confidence, 1.0))  # Ensure within [0,1]
            except:
                confidence = 1.0  # Default confidence if unable to calculate

            # SELL logic
            if self.position:
                if self.curr_share_predicted_price > predicted_price:
                    if self.params.verbose:
                        print("Price decreased according to forecast, selling position.")
                    self.order = self.sell(size=self.position.size)  # Sell entire position
                    self.share_price = math.inf
                    self.curr_share_predicted_price = -math.inf
                    if self.params.verbose:
                        print(f"SELL EXECUTED at {current_price}")
                elif predicted_price < current_price * (1 - self.params.stop_loss):
                    if self.params.verbose:
                        print("Hit stop loss, selling position.")
                    self.order = self.sell(size=self.position.size)  # Sell entire position
                    self.share_price = math.inf
                    self.curr_share_predicted_price = -math.inf
                    if self.params.verbose:
                        print(f"SELL EXECUTED at {current_price}")

            # BUY logic (allows multiple buys)
            if predicted_price > current_price:
                size = self.determine_size(current_price, predicted_price, confidence)
                if size > 0:
                    # Check if adding this size exceeds maximum allowed
                    current_total_size = self.position.size * current_price
                    if self.params.max_total_size and (current_total_size + size * current_price) > self.params.max_total_size:
                        if self.params.verbose:
                            print("Maximum total size reached. Skipping additional buy.")
                    elif self.params.max_positions and self.position.size >= self.params.max_total_size / current_price:
                        if self.params.verbose:
                            print("Maximum number of positions reached. Skipping additional buy.")
                    else:
                        if self.params.verbose:
                            print(f"Placing BUY order for size: {size}")
                        self.order = self.buy(size=size)
                        self.share_price = current_price
                        self.curr_share_predicted_price = predicted_price
                        if self.params.verbose:
                            print(f"BUY EXECUTED at {current_price}, Size: {size}")

        except Exception as e:
            logging.error(f"Error during forecasting and trading: {e}")

    def notify_order(self, order):
        """
        Notification for order status changes.
        """
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted, nothing to do
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                if self.params.verbose:
                    print(f"BUY ORDER COMPLETED at {self.buyprice}")
            elif order.issell():
                if self.params.verbose:
                    print(f"SELL ORDER COMPLETED at {order.executed.price}")

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logging.warning(f"Order {order.Status[order.status]}")

        # Reset orders
        self.order = None

    def notify_trade(self, trade):
        """
        Notification for trade status changes.
        """
        if not trade.isclosed:
            return

        if self.params.verbose:
            print(f"OPERATION PROFIT, GROSS {trade.pnl}, NET {trade.pnlcomm}")
# %% [markdown]
# # Custom Analyzer: FinalValue for Backtrader
# 
# This section introduces the **FinalValue Analyzer**, a custom analyzer tailored for the **Backtrader** framework. The analyzer is designed to **capture the final portfolio value** at the end of a backtesting run, providing valuable insights into the overall performance of a trading strategy.
# 
# ---
#     
# ## **1. Analyzer Overview**
# 
# ### **Concept Overview**
# 
# In Backtrader, **Analyzers** are specialized components that compute and return various metrics and statistics about the performance of trading strategies. They operate alongside strategies to provide comprehensive performance evaluations without interfering with the strategy's execution flow.
# 
# The **FinalValue Analyzer** is a custom analyzer that specifically focuses on recording the **final value** of the portfolio once the backtest concludes. This metric is crucial for assessing the profitability and effectiveness of a trading strategy.
# 
# ### **Purpose**
# 
# The primary objectives of the **FinalValue Analyzer** are:
# 
# 1. **Capture Final Portfolio Value:**  
#    Record the total value of the portfolio at the end of the backtest, providing a clear indicator of the strategy's success.
# 
# 2. **Facilitate Performance Comparison:**  
#    Enable easy comparison of different strategies or parameter configurations based on their final portfolio values.
# 
# 3. **Enhance Reporting:**  
#    Integrate seamlessly with Backtrader's reporting mechanisms, allowing for automated extraction and presentation of the final portfolio value alongside other performance metrics.
# 
# ### **Benefits**
# 
# - **Simplicity:**  
#   Provides a straightforward mechanism to obtain the final portfolio value without the need for complex computations or additional data handling.
# 
# - **Reusability:**  
#   Can be easily integrated into any Backtrader backtesting setup, making it a versatile tool for various trading strategies.
# 
# - **Automation:**  
#   Automatically captures and stores the final portfolio value, reducing manual effort and potential errors in performance assessment.
# 
# - **Integration with Other Analyzers:**  
#   Complements other analyzers (e.g., Sharpe Ratio, DrawDown) to offer a holistic view of strategy performance.
# 
# ---
# 

# %%
import backtrader as bt

class FinalValue(bt.Analyzer):
    """
    Custom Analyzer to capture the final portfolio value.
    """
    def __init__(self):
        self.final_value = 0.0

    def stop(self):
        """
        Called at the end of the strategy run.
        """
        self.final_value = self.strategy.broker.getvalue()

    def get_analysis(self):
        """
        Returns the analysis results.
        """
        return {'final_value': self.final_value}


# %% [markdown]
# # Backtrader Backtesting Setup with SARIMAXStrategy and Custom Analyzers
# 
# This section outlines the setup and execution of a **Backtrader** backtesting environment using the **SARIMAXStrategy**. The configuration includes initializing the Cerebro engine, adding data feeds, setting up strategy parameters for optimization, incorporating analyzers for performance evaluation, configuring broker settings, executing the backtest, and extracting the results for analysis.
# 
# ---
#     
# ## **1. Overview**
# 
# The provided code performs the following key functions:
# 
# 1. **Initialize Cerebro Engine:** Sets up Backtrader's core engine for running backtests.
# 2. **Add Data Feeds:** Imports historical price data for financial instruments (e.g., GLD and SLV).
# 3. **Add and Configure Strategy with Optimization:** Implements the `SARIMAXStrategy` with multiple parameter configurations for optimization.
# 4. **Add Analyzers:** Integrates performance analyzers to evaluate strategy effectiveness.
# 5. **Configure Broker Settings:** Sets initial capital and commission structures.
# 6. **Run the Backtest:** Executes the backtest using specified CPU resources.
# 7. **Extract and Display Results:** Retrieves and prints performance metrics from each optimized strategy run.
# 8. **Optional Plotting:** Provides an option to visualize the backtest results. *Side Note: Will not work inside jupyter notebooks.*
# 
# ---

def run_backtest():
    # Initialize Cerebro engine
    cerebro = bt.Cerebro(optreturn=False)  # Set optreturn=False to get a list of results

    # Add data feeds to Cerebro
    # Example: Assuming you've loaded your data into Pandas DataFrames named data_bal, etc.
    def load_data(file_path):
        """
        Load and preprocess CSV data.
        """
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        return bt.feeds.PandasData(dataname=data)

    # Load all required data
    data_bal = load_data('BAL-Prices.csv')
    # data_slv = load_data('SLV-Prices.csv')  # Uncomment if needed
    # Add more data feeds as required

    cerebro.adddata(data_gld, name='GLD')
    # cerebro.adddata(data_slv, name='SLV')  # Uncomment if needed

    # Add the SARIMAX strategy to Cerebro with optimization parameters
    cerebro.optstrategy(
        SARIMAXStrategy,
        forecast_period=[1],                # Use list even if single value
        model_order=[(2, 1, 1)],  # Example ranges for p,d,q
        seasonal_order=[(1, 1, 1, 7)],  # Example ranges for P,D,Q,s
        threshold=[0.01],             # Example thresholds
        verbose=[False],                    # Set to True for detailed logging if needed
        stop_loss=[0.10],             # 5% and 10% stop loss
        take_profit=[0.25],           # 10% and 25% take profit
        fit_interval=[5],                   # Fixed fit interval
        sizing_method=['confidence'],  # Example sizing methods 'fixed', 'percentage', 'confidence', 'kelly'
        fixed_size=[1000],                  # Fixed amount to invest per trade
        max_percentage=[0.1],               # Max portfolio percentage per trade
        confidence_multiplier=[2],          # Multiplier for confidence-based sizing
        max_positions=[5],                  # Maximum number of buy orders allowed
        max_total_size=[10000],             # Maximum total capital allocated to the asset
        kelly_b=[1.0],                       # Net odds (1:1)
        kelly_p=[0.6],                       # Example win probability
        kelly_multiplier=[0.5],                     # Example win probability
    )

    # Add Analyzers
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(FinalValue, _name='final_value')  # Add the custom analyzer

    # Set initial capital
    cerebro.broker.setcash(100000.0)
    

    # Set commission - 0.1% per trade
    cerebro.broker.setcommission(commission=0.001)

    # Print starting portfolio value
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Run the backtest with optimization
    num_cpus = multiprocessing.cpu_count()

    results = cerebro.run(maxcpus=num_cpus)

    # Since we used optstrategy, results is a list of lists
    print(f"Total Strategies Run: {len(results)}")

    # Initialize variables to track the best strategy
    best_value = -math.inf
    best_params = None
    best_sharpe = None
    best_drawdown = None

    # Iterate over all strategy instances and extract analyzer data
    for strat_list in results:
        for strat in strat_list:
            final_analyzer = strat.analyzers.getbyname('final_value')
            if final_analyzer:
                final = final_analyzer.get_analysis()
            else:
                print("FinalValue analyzer not found for this strategy instance.")
                final = None

            sharpe_analyzer = strat.analyzers.getbyname('sharpe_ratio')
            if sharpe_analyzer:
                sharpe = sharpe_analyzer.get_analysis()
            else:
                print("SharpeRatio analyzer not found for this strategy instance.")
                sharpe = None

            drawdown_analyzer = strat.analyzers.getbyname('drawdown')
            if drawdown_analyzer:
                drawdown = drawdown_analyzer.get_analysis()
            else:
                print("DrawDown analyzer not found for this strategy instance.")
                drawdown = None

            if final:
                portfolio_value = final.get('final_value')
                # Check if this is the best portfolio value so far
                if portfolio_value > best_value:
                    best_value = portfolio_value
                    best_params = strat.params
                    best_sharpe = sharpe
                    best_drawdown = drawdown

    # Check if a best strategy was found
    if best_params:
        # Print the best strategy parameters and performance
        print("\n===== Best Strategy Parameters =====")
        print(f"Final Portfolio Value: {best_value:.2f}")
        print(f"Model Order (p, d, q): {best_params.model_order}")
        print(f"Seasonal Order (P, D, Q, s): {best_params.seasonal_order}")
        print(f"Threshold: {best_params.threshold}")
        print(f"Stop Loss: {best_params.stop_loss}")
        print(f"Take Profit: {best_params.take_profit}")
        print(f"Sizing Method: {best_params.sizing_method}")
        if best_params.sizing_method == 'fixed':
            print(f"Fixed Size: {best_params.fixed_size}")
        elif best_params.sizing_method in ['percentage', 'confidence']:
            print(f"Max Percentage: {best_params.max_percentage}")
            if best_params.sizing_method == 'confidence':
                print(f"Confidence Multiplier: {best_params.confidence_multiplier}")
        print("\n===== Performance Metrics =====")
        print(f"Sharpe Ratio: {best_sharpe.get('sharperatio', 'N/A')}")
        print(f"Max Drawdown: {drawdown.max.drawdown}%")
    else:
        print("No valid strategies found.")

    # Optional: Plot the results for the best strategy
    # Note: Plotting optimized strategies can be complex. Consider re-running Cerebro with the best parameters.
    
    cerebro = bt.Cerebro()
    cerebro.adddata(data_gld, name='GLD')
    cerebro.addstrategy(
        SARIMAXStrategy,
        forecast_period=best_params.forecast_period,
        model_order=best_params.model_order,
        seasonal_order=best_params.seasonal_order,
        threshold=best_params.threshold,
        verbose=best_params.verbose,
        stop_loss=best_params.stop_loss,
        take_profit=best_params.take_profit,
        fit_interval=best_params.fit_interval,
        sizing_method=best_params.sizing_method,
        fixed_size=best_params.fixed_size,
        max_percentage=best_params.max_percentage,
        confidence_multiplier=best_params.confidence_multiplier,
        max_positions=best_params.max_positions,
        max_total_size=best_params.max_total_size,
    )
    # Add analyzers again if needed
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(FinalValue, _name='final_value')
    # Set broker settings
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    # Run and plot
    cerebro.run()
    cerebro.plot(style='candlestick')
    print("\n===== Best Strategy Parameters =====")
    print(f"Final Portfolio Value: {best_value:.2f}")
    print(f"Model Order (p, d, q): {best_params.model_order}")
    print(f"Seasonal Order (P, D, Q, s): {best_params.seasonal_order}")
    print(f"Threshold: {best_params.threshold}")
    print(f"Stop Loss: {best_params.stop_loss}")
    print(f"Take Profit: {best_params.take_profit}")
    print(f"Sizing Method: {best_params.sizing_method}")
    if best_params.sizing_method == 'fixed':
        print(f"Fixed Size: {best_params.fixed_size}")
    elif best_params.sizing_method in ['percentage', 'confidence']:
        print(f"Max Percentage: {best_params.max_percentage}")
        if best_params.sizing_method == 'confidence':
            print(f"Confidence Multiplier: {best_params.confidence_multiplier}")
    print("\n===== Performance Metrics =====")
    print(f"Sharpe Ratio: {best_sharpe.get('sharperatio', 'N/A')}")
    print(f"Max Drawdown: {drawdown.max.drawdown}%")


if __name__ == "__main__":
    run_backtest()
