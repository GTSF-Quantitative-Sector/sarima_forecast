import backtrader as bt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import math
import logging

import math
import logging
import pmdarima as pm
from pmdarima import auto_arima
import backtrader as bt
import pandas as pd


class AutoArimaStrategy(bt.Strategy):
    params = (
        ('forecast_period', 1),
        ('seasonal', True),
        ('m', 7),  # Weekly seasonality
        ('threshold', 0.05),
        ('verbose', False),
        ('min_data_points', 30),
        ('stop_loss', 0.05),
        ('take_profit', 0.1),
        ('fit_interval', 5)
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.model = None
        self.counter = 0
        self.share_price = math.inf

    def next(self):
        self.counter += 1
        if self.counter < self.params.fit_interval:
            return
        self.counter = 0

        if self.order:
            return

        if len(self.dataclose) < self.params.min_data_points:
            return

        historical_data = pd.Series(
            list(self.dataclose.get(size=self.params.min_data_points)))
        historical_data = historical_data[::-1]
        historical_data = pd.to_numeric(
            historical_data, errors='coerce').dropna()
        historical_data.index = pd.date_range(
            end=pd.Timestamp.now(), periods=len(historical_data), freq='D')

        try:
            self.model = auto_arima(
                historical_data,
                seasonal=self.params.seasonal,
                m=self.params.m,
                trace=self.params.verbose,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )

            if self.params.verbose:
                print(self.model.summary())

            forecast = self.model.predict(
                n_periods=self.params.forecast_period)
            predicted_price = forecast[-1]
            current_price = self.dataclose[0]

            if self.params.verbose:
                print(
                    f"Predicted Price: {predicted_price}, Current Price: {current_price}")

            if predicted_price > current_price * (1 + self.params.threshold):
                if not self.position:
                    self.order = self.buy()
                    self.share_price = current_price
                    if self.params.verbose:
                        print(f"BUY EXECUTED at {current_price}")
            elif current_price >= self.share_price * (1 + self.params.take_profit) or current_price <= self.share_price * (1 - self.params.stop_loss):
                if self.position:
                    self.order = self.sell()
                    self.share_price = math.inf
                    if self.params.verbose:
                        print(f"SELL EXECUTED at {current_price}")

        except Exception as e:
            logging.error(f"Error during forecasting and trading: {e}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
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

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logging.warning(f"Order {order.Status[order.status]}")
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        if self.params.verbose:
            print(f"OPERATION PROFIT, GROSS {trade.pnl}, NET {trade.pnlcomm}")


class SARIMAXStrategy(bt.Strategy):
    params = (
        ('forecast_period', 1),
        ('model_order', (1, 1, 1)),
        ('seasonal_order', (1, 1, 1, 7)),
        ('threshold', 0.05),
        ('verbose', False),
        ('min_data_points', 30),
        ('stop_loss', 0.05),
        ('take_profit', 0.1),
        ('fit_interval', 5),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.model = None
        self.model_fit = None
        self.counter = 0
        self.share_price = math.inf

    def next(self):
        self.counter += 1
        if self.counter < self.params.fit_interval:
            return
        self.counter = 0

        if self.order or len(self.dataclose) < self.params.min_data_points:
            return

        historical_data = pd.Series(
            list(self.dataclose.get(size=self.params.min_data_points)))[::-1]
        historical_data = pd.to_numeric(
            historical_data, errors='coerce').dropna()

        try:
            self.model = SARIMAX(
                historical_data,
                order=self.params.model_order,
                seasonal_order=self.params.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model_fit = self.model.fit(disp=False)

            forecast = self.model_fit.forecast(
                steps=self.params.forecast_period)
            predicted_price = forecast.iloc[-1]
            current_price = self.dataclose[0]

            if predicted_price > current_price * (1 + self.params.threshold):
                if not self.position:
                    self.order = self.buy()
                    self.share_price = current_price
            elif current_price >= self.share_price * (1 + self.params.take_profit) or current_price <= self.share_price * (1 - self.params.stop_loss):
                if self.position:
                    self.order = self.sell()
                    self.share_price = math.inf

        except Exception as e:
            logging.error(f"Error during forecasting and trading: {e}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
