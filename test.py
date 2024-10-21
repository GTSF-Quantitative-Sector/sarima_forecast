import backtrader as bt
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime
import backtrader.feeds as btfeeds
from polygon.rest import RESTClient


class SARIMAXStrategy(bt.Strategy):
    params = (
        ('order', (2, 1, 1)),
        ('seasonal_order', (1, 1, 1, 7)),
        ('forecast_period', 1),
        ('model_fit_frequency', 10),  # Fit the model every 10 bars
        # Minimum percentage difference to trigger a trade
        ('trade_threshold', 0.05),
        ('stop_loss', 0.05),          # Stop loss percentage
        ('take_profit', 0.1),        # Take profit percentage
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.history = []
        self.model = None
        self.predicted = None
        self.bar_counter = 0
        self.order = None  # To keep track of pending orders
        self.entry_price = 0  # Price at which the position was entered

    def next(self):
        # Append the current close price to history
        self.history.append(self.dataclose[0])

        # Increment the bar counter
        self.bar_counter += 1

        # Check if there are any pending orders
        if self.order:
            return  # If an order is pending, do nothing

        # Check if we have enough data to train the model
        if len(self.history) < 50:
            # Not enough data to train the model yet
            return

        # Fit the model at specified frequency to save computation
        if self.bar_counter % self.params.model_fit_frequency == 0:
            try:
                # Define the SARIMAX model
                self.model = SARIMAX(
                    self.history, order=self.params.order, seasonal_order=self.params.seasonal_order)

                # Fit the model
                self.model_fit = self.model.fit(disp=False)

                # Forecast the next period
                self.predicted = self.model_fit.forecast(
                    steps=self.params.forecast_period)[0]

                # Debugging: Print the forecasted value
                print(
                    f'Predicted next close: {self.predicted:.2f}, Actual close: {self.dataclose[0]:.2f}')

            except Exception as e:
                print(f"Model fitting failed: {e}")
                self.predicted = None

        # Make trading decisions based on the prediction and trade threshold
        if self.predicted:
            # Calculate the percentage difference between predicted and current price
            pct_diff = (
                (self.predicted - self.dataclose[0]) / self.dataclose[0]) * 100

            # Check if the difference meets the trade threshold
            if abs(pct_diff) >= self.params.trade_threshold:
                if pct_diff > 0:
                    # If forecasted price is higher than current price, consider buying
                    if not self.position:
                        self.buy()
                elif pct_diff < 0:
                    # If forecasted price is lower than current price, consider selling
                    if self.position:
                        self.sell()

        # Implement Stop Loss and Take Profit
        if self.position:
            # Calculate the current percentage change from the entry price
            current_price = self.dataclose[0]
            pct_change = ((current_price - self.entry_price) /
                          self.entry_price) * 100

            # Check for Take Profit
            if pct_change >= self.params.take_profit:
                print(
                    f'Take Profit triggered at {current_price:.2f} (Change: {pct_change:.2f}%)')
                self.sell()

            # Check for Stop Loss
            elif pct_change <= -self.params.stop_loss:
                print(
                    f'Stop Loss triggered at {current_price:.2f} (Change: {pct_change:.2f}%)')
                self.sell()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Order is pending
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price  # Set entry price on buy
                print(f'BUY EXECUTED at {order.executed.price:.2f}')
            elif order.issell():
                print(f'SELL EXECUTED at {order.executed.price:.2f}')
            self.bar_counter = 0  # Reset bar counter after order execution

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print('Order Canceled/Margin/Rejected')

        # Reset orders
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        print(f'Trade PnL: Gross {trade.pnl:.2f}, Net {trade.pnlcomm:.2f}')


def fetch_polygon_data(symbol, start, end, api_key):
    client = RESTClient(api_key)
    # Fetch aggregate bars (e.g., daily)
    aggs = client.get_aggs(symbol, 1, "day", start.strftime(
        '%Y-%m-%d'), end.strftime('%Y-%m-%d'))

    # Convert to pandas DataFrame
    data = []
    for agg in aggs:
        data.append({
            'datetime': datetime.datetime.fromtimestamp(agg.timestamp / 1000),
            'open': agg.open,
            'high': agg.high,
            'low': agg.low,
            'close': agg.close,
            'volume': agg.volume
        })
    df = pd.DataFrame(data)
    df.set_index('datetime', inplace=True)
    return df


def run_backtest():
    # Initialize Cerebro engine
    cerebro = bt.Cerebro()

    # Add the SARIMAX strategy
    cerebro.addstrategy(SARIMAXStrategy)

    # Fetch GLD data
    api_key = 'uwQtl3txGt5BLbecq7ZbIu0ZbuitCGjc'
    symbol = 'GLD'
    start_date = datetime.datetime(2010, 1, 1)
    end_date = datetime.datetime(2024, 10, 21)

    df = fetch_polygon_data(symbol, start_date, end_date, api_key)

    if df.empty:
        print("No data fetched from Polygon.")
        return

    # Create a Backtrader data feed
    data = bt.feeds.PandasData(dataname=df)

    # Add the data feed to Cerebro
    cerebro.adddata(data)

    # Set initial cash
    cerebro.broker.setcash(100000.0)

    # Set the commission
    cerebro.broker.setcommission(commission=0.001)

    # Print starting cash
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')

    # Run the backtest
    cerebro.run()

    # Print final cash
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

    # Plot the results
    cerebro.plot()


if __name__ == '__main__':
    run_backtest()
