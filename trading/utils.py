import pandas as pd


def load_data(filename):
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def configure_cerebro(cerebro, strategy_class, data, initial_cash=100000.0):
    cerebro.adddata(data)
    cerebro.addstrategy(strategy_class)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)
