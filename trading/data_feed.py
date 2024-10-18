import backtrader as bt


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
