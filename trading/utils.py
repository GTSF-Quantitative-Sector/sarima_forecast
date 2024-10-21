import pandas as pd
import logging
from polygon.rest import RESTClient


def load_data(filename):
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def get_asset_data(ticker: str, start_date: str, end_date: str, API_KEY: str):
    """
    Fetches historical price data for a given asset (e.g., GLD, SLV) between the specified start and end dates.

    Args:
        ticker (str): The ticker symbol (e.g., 'GLD', 'SLV').
        start_date (str): Start date in the format 'YYYY-MM-DD'.
        end_date (str): End date in the format 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: DataFrame containing the historical price data for the asset.
    """
    try:
        logging.info(f"Initializing RESTClient for {ticker}...")
        rest_client = RESTClient(API_KEY)

        logging.info(
            f"Fetching aggregate bars data for '{ticker}' from {start_date} to {end_date}...")
        # Fetching daily aggregate data
        response = rest_client.get_aggs(
            ticker, 1, 'day', start_date, end_date, limit=50000
        )

        if not response:
            logging.error(
                f"Received empty response from RESTClient for {ticker}.")
            return None

        logging.info(f"Converting response to DataFrame for {ticker}...")
        data = [
            {
                't': item.timestamp,
                'o': item.open,
                'h': item.high,
                'l': item.low,
                'c': item.close,
                'v': item.volume
            }
            for item in response
        ]
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('Date', inplace=True)
        df.drop(columns=['t'], inplace=True)

        # Rename columns to match desired format
        df.rename(columns={
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume'
        }, inplace=True)

        # Convert numeric columns to proper type
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_columns] = df[numeric_columns].apply(
            pd.to_numeric, errors='coerce')

        # Save DataFrame to CSV
        csv_filename = f'{ticker}-Prices.csv'
        df.to_csv(csv_filename, index=True)
        logging.info(f"{ticker} prices saved to '{csv_filename}'.")

    except Exception as e:
        logging.error(f"An error occurred in get_asset_data: {e}")
        return None

    return df


def configure_cerebro(cerebro, strategy_class, data, initial_cash=100000.0):
    cerebro.adddata(data)
    cerebro.addstrategy(strategy_class)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)
