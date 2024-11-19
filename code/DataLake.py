# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:21:36 2024

@author: 14211
"""

import os
import pandas as pd
import quandl
import datetime
import requests
from scipy.stats import zscore
from typing import List, Dict, Optional
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas_market_calendars import get_calendar
import yfinance as yf

# https://data.nasdaq.com/databases/ZFB#usage

api_key = 'CPByrDeaJtVhVwzhzYaD' # Quandl API
# Ensure you have your Quandl API key set as an environment variable or replace 'YOUR_API_KEY' with your actual key
quandl.ApiConfig.api_key = os.getenv('QUANDL_API_KEY', api_key)

#%%
class DataLake:
    """
    DataLake is used for persistent storage of raw and processed datasets.
    """
    def __init__(self, storage_path='data_lake/equity'):
        self.storage_path = storage_path
        self.raw_data = {}
        self.processed_data = {}
        os.makedirs(self.storage_path, exist_ok=True)
        
    def store_data(self, file_path: str, data: pd.DataFrame, processed: bool = False):
        """
        Stores data in either raw or processed storage and persists it to disk.
        
        Parameters:
            file_path (str): Name of the dataset.
            data (DataFrame): Data to be stored (Pandas DataFrame).
            processed (bool): Flag indicating whether the data is processed.
        """
        # Save to disk as parquet
        file_path = f"{file_path}.parquet"
        data.to_parquet(file_path, index=False)
        #print(f"Dataset '{dataset_name}' has been stored at '{file_path}'.")
        
    def retrieve_data(self, ticker: str, data_type: str, frequency: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Retrieves data for a specific ticker, data type (e.g., close, open, high, low, volume, news), and frequency.

        Parameters:
            ticker (str): Name of the stock ticker (e.g., 'TSLA').
            data_type (str): Type of data to retrieve (e.g., 'close', 'open', 'high', 'low', 'volume', 'news').
            frequency (str): Frequency of the data (e.g., '1day', ignored for 'news').
            start_date (str, optional): Start date for filtering (format: 'YYYY-MM-DD').
            end_date (str, optional): End date for filtering (format: 'YYYY-MM-DD').

        Returns:
            DataFrame: The requested dataset, filtered by the date range, or None if not found.
        """
        # Define the base path for equity data
        base_path = os.path.join(self.storage_path,'features', ticker)

        if not frequency:
            # News data path
            file_name = f"{data_type}.parquet"
            file_path = os.path.join(base_path,file_name)
        else:
            # Market data path
            file_name = f"{data_type}.{frequency}.parquet"
            file_path = os.path.join(base_path, file_name)

        # Check if the file exists
        if os.path.exists(file_path):
            try:
                # Load the data from the Parquet file
                data = pd.read_parquet(file_path)
                print(f"Data for ticker '{ticker}', data type '{data_type}' loaded from '{file_path}'.")

                # Filter data by the date range if specified
                if start_date or end_date:
                    data['date'] = pd.to_datetime(data['date'])  # Ensure 'date' column is in datetime format
                    if start_date:
                        start_date = pd.to_datetime(start_date)
                        data = data[data['date'] >= start_date]
                    if end_date:
                        end_date = pd.to_datetime(end_date)
                        data = data[data['date'] <= end_date]
                    print(f"Data filtered from {start_date} to {end_date}. Remaining records: {len(data)}.")

                return data
            except Exception as e:
                print(f"Failed to load or process data for ticker '{ticker}', data type '{data_type}': {e}")
                return None
        else:
            print(f"Data for ticker '{ticker}', data type '{data_type}' not found at '{file_path}'.")
            return None

    
class NewsDataManager(DataLake):
    def __init__(self, data_lake: DataLake, api_key: str):
        """
        Initialize NewsDataManager with DataLake and API key.
        """
        super().__init__()
        self.data_lake = data_lake
        self.api_key = api_key  
        self.base_url = "https://newsapi.org/v2/everything"

    def fetch_news(self, query: str, from_date: str, to_date: str, language: str = "en") -> Optional[pd.DataFrame]:
        """
        Fetch news data for a given query (e.g., Tesla) within a date range.
        """
        params = {
            "q": query,
            "from": from_date,
            "to": to_date,
            "language": language,
            "apiKey": self.api_key,
            "pageSize": 100
        }
        try:
            print(f"Fetching news for query '{query}' from {from_date} to {to_date}...")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            news_data = response.json().get("articles", [])
            if not news_data:
                print(f"No news articles found for query '{query}'.")
                return None
            print(f"Fetched {len(news_data)} news articles.")
            df = pd.DataFrame(news_data)
            return df
        except Exception as e:
            print(f"Failed to fetch news: {e}")
            return None

    def process_news(self, company_instrument_tuples: List[tuple], start_date: str, end_date: str):
        """
        Fetch and store news data for a specific query in a directory structure.
        Renames 'publishedAt' to 'date' and converts its data type to datetime64[ns].
        """
        for query, ticker in company_instrument_tuples:
            # Fetch news data
            df = self.fetch_news(query, start_date, end_date)
            root_file_path = os.path.join(self.data_lake.storage_path, 'features')
            os.makedirs(root_file_path, exist_ok=True)
            
            if df is not None and not df.empty:
                # Rename 'publishedAt' to 'date' and convert to datetime
                if 'publishedAt' in df.columns:
                    df = df.rename(columns={'publishedAt': 'date'})
                    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')
                else:
                    print("'publishedAt' column not found in the data.")

                # Create a directory for the ticker
                ticker_folder = os.path.join(root_file_path, ticker)
                os.makedirs(ticker_folder, exist_ok=True)

                # Save the data as news.parquet
                file_path = os.path.join(ticker_folder, "news.parquet")
                df.to_parquet(file_path, index=False)
                print(f"News data for '{query}' stored at '{file_path}'.")
            else:
                print(f"No news data to store for '{query}'.")


class QuandlDataManager(DataLake):
    """
    QuandlDataManager handles data extraction from Quandl, cleaning, and storage into the Data Lake.
    """
    def __init__(self, data_lake: DataLake, api_key: Optional[str] = None):
        super().__init__()
        
        self.data_lake = data_lake
        self.api_key = api_key or quandl.ApiConfig.api_key
    #     #self.setup_trading_days(start_date, end_date)
        
    
    # def setup_trading_days(self, start_date: str, end_date: str) -> (pd.DatetimeIndex, int):
    #     nyse = get_calendar('XNYS')
    #     trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)
    #     return trading_days, len(trading_days)
    
    def fetch_quandl_table(self, table_name: str, params: Optional[Dict] = None) -> Optional[pd.DataFrame]:
        """
        Fetches table data using the Quandl API.
        
        Parameters:
            table_name (str): Name of the Quandl table, e.g., 'ZACKS/FC'.
            params (dict): Additional query parameters, such as filters.
        
        Returns:
            DataFrame: The retrieved data.
        """
        try:
            print(f"Fetching table {table_name} ...")
            df = quandl.get_table(table_name, **(params or {}), paginate=True)
            print(f"Table {table_name} fetched successfully, records: {len(df)}")
            return df
        except Exception as e:
            print(f"Failed to fetch table {table_name}: {e}")
            return None
    
    def clean_data(self, df: pd.DataFrame, numerical_cols: List[str], z_score: bool = False) -> pd.DataFrame:
        """
        Cleans data, including handling missing values and applying z-score normalization.
        
        Parameters:
            df (DataFrame): Data to be cleaned.
            numerical_cols (list): Numerical columns for z-score normalization.
            z_score (bool): Whether to apply z-score normalization.
        
        Returns:
            DataFrame: Cleaned data.
        """
        # Handle missing values: Drop rows with missing values
        #import pdb;pdb.set_trace()
        df_clean = df.dropna()
        print(f"Records after removing missing values: {len(df_clean)}")
        
        # Apply z-score normalization
        if z_score and numerical_cols:
            df_clean[numerical_cols] = df_clean[numerical_cols].apply(zscore)
            print("Z-score normalization applied to numerical columns.")
        
        return df_clean

    def split_list(self, lst: List[str], n: int) -> List[List[str]]:
        """
        Splits a list into sublists of specified size.
        
        Parameters:
            lst (List[str]): List to be split.
            n (int): Maximum size of each sublist.
        
        Returns:
            List[List[str]]: Splitted sublists.
        """
        return [lst[i:i + n] for i in range(0, len(lst), n)]
    
    def process_quandl_table(
        self,
        instruments: List[str],
        table_name: str,
        start_date: str,
        end_date: str,
        keep_cols: List[str],
        date_col: str = 'per_end_date',
        sort_by: List[str] = None,
        additional_filters: Dict = None,
        process_zscore: bool = False
    ) -> None:
        """
        Process Quandl tables for a specific list of instruments and store data per ticker.

        Parameters:
            instruments (List[str]): List of stock tickers to process.
            table_name (str): Name of the Quandl table.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            keep_cols (List[str]): Columns to retain.
            date_col (str): Name of the date column.
            sort_by (List[str]): Columns for sorting.
            additional_filters (Dict): Additional filtering conditions.
            process_zscore (bool): Whether to apply z-score normalization.

        Returns:
            None
        """
        print(f"Processing Quandl table '{table_name}' for {len(instruments)} instruments...")
        root_file_path = os.path.join(self.data_lake.storage_path, 'features')
        os.makedirs(root_file_path, exist_ok=True)

        for ticker in instruments:
            print(f"Fetching data for ticker '{ticker}'...")
            params = {
                'ticker': ticker,
                f'{date_col}.gte': start_date,
                f'{date_col}.lte': end_date
            }
            if additional_filters:
                params.update(additional_filters)

            try:
                df = self.fetch_quandl_table(table_name, params=params)
                if df is None or df.empty:
                    print(f"No data fetched for ticker '{ticker}' in table '{table_name}'.")
                    continue

                # Convert date column
                df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d')
                df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
                if sort_by:
                    df = df.sort_values(by=sort_by)
                else:
                    df = df.sort_values(by=[date_col])
                
                if date_col != 'date':
                    df = df.rename(columns={date_col: 'date'})  

                # Define the path for storing ticker data
                ticker_folder = os.path.join(root_file_path, ticker)
                os.makedirs(ticker_folder, exist_ok=True)

                # Save ticker data
                for column in keep_cols:
                    column_data = df[['date', column]].reset_index(drop=True)
                    dataset_name = f"{root_file_path}/{ticker}/{column}"
                    self.store_data(dataset_name, column_data, processed=True)

            except Exception as e:
                print(f"Failed to fetch or process data for ticker '{ticker}': {e}")

        print(f"Finished processing and storing data for table '{table_name}'.")

            
    def process_quotemedia_prices(
        self,
        instruments: list,
        start_date: str,
        end_date: str,
        batch_size: int = 100
    ) -> None:
        """
        Processes the QUOTEMEDIA/PRICES table. Stores data for each stock ticker separately.
        """
        print("Starting to process the QUOTEMEDIA/PRICES table...")
        #print(self.data_lake.storage_path)
        root_file_path = os.path.join(self.data_lake.storage_path, 'features')
        os.makedirs(root_file_path, exist_ok=True)

        # Split ticker list into batches
        ticker_batches = self.split_list(list(instruments), batch_size)
        print(f"Split {len(instruments)} tickers into {len(ticker_batches)} batches, each with up to {batch_size} tickers.")

        for idx, batch in enumerate(ticker_batches):
            print(f"Processing ticker batch {idx + 1}/{len(ticker_batches)}, containing {len(batch)} tickers.")
            try:
                df_batch = quandl.get_table(
                    'QUOTEMEDIA/PRICES',
                    ticker=batch,
                    date={'gte': start_date, 'lte': end_date},
                    paginate=True
                )
                if df_batch is None or df_batch.empty:
                    print(f"Batch {idx + 1} contains no data.")
                    continue

                # Convert and filter by date
                df_batch['date'] = pd.to_datetime(df_batch['date'], format='%Y-%m-%d', errors='coerce')
                df_batch = df_batch[(df_batch['date'] >= start_date) & (df_batch['date'] <= end_date)]
                df_batch = df_batch.sort_values(by='date', ascending=True)
                #import pdb;pdb.set_trace()

                # Process data for each ticker
                for ticker in batch:
                    ticker_data = df_batch[df_batch['ticker'] == ticker]
                    if ticker_data.empty:
                        continue

                    # Save ticker data
                    ticker_folder = os.path.join(root_file_path, ticker)
                    os.makedirs(ticker_folder, exist_ok=True)

                    for column in ['open', 'high', 'low', 'close', 'volume']:
                        column_data = ticker_data[['date', column]].reset_index(drop=True)
                        dataset_name = f"{root_file_path}/{ticker}/{column}.1day"
                        self.store_data(dataset_name, column_data, processed=True)

            except Exception as e:
                print(f"Failed to fetch batch {idx + 1}: {e}")

        print("Finished processing the QUOTEMEDIA/PRICES table.")


    def process_yfinance_intraday_prices(
        self,
        instruments: list,
        start_date: str,
        end_date: str,
        interval: str = "30m"  # 30-minute interval
    ) -> None:
        """
        Processes intraday price data using yfinance. Stores data for each stock ticker separately.

        Parameters:
            instruments (list): List of stock tickers to process (e.g., ['AAPL', 'MSFT']).
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            interval (str): Data interval (default: '30m').
        """
        print(f"Starting to process intraday prices with interval '{interval}'...")
        root_file_path = os.path.join(self.data_lake.storage_path, 'features')
        os.makedirs(root_file_path, exist_ok=True)

        for ticker in instruments:
            print(f"Fetching intraday data for ticker '{ticker}'...")
            try:
                # Download data using yfinance
                df = yf.download(
                    tickers=ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False
                )

                if df.empty:
                    print(f"No data found for ticker '{ticker}' with interval '{interval}'.")
                    continue

                # Reset index to make 'Datetime' a column
                df.reset_index(inplace=True)
                df.rename(columns={"Datetime": "date"}, inplace=True)
                df["date"] = pd.to_datetime(df["date"])

                # Save the data for each column
                ticker_folder = os.path.join(root_file_path, ticker)
                os.makedirs(ticker_folder, exist_ok=True)

                for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if column not in df.columns:
                        print(f"Column '{column}' not found for ticker '{ticker}'. Skipping...")
                        continue

                    column_data = df[['date', column]].rename(columns={column: column.lower()}).reset_index(drop=True)
                    dataset_name = f"{ticker_folder}/{column.lower()}.{interval}"
                    self.store_data(dataset_name, column_data, processed=True)
                    print(f"Data for '{column}' of ticker '{ticker}' saved at '{dataset_name}.parquet'.")

            except Exception as e:
                print(f"Failed to fetch data for ticker '{ticker}': {e}")

        print(f"Finished processing intraday prices with interval '{interval}'.")


    def generate_trading_calendar(self) -> None:
        """
        Generates and stores the trading calendar based on the Data Lake data.
        """
        root_file_path = os.path.join(self.data_lake.storage_path, 'features')
        all_trading_days = set()

        for ticker_folder in os.listdir(root_file_path):
            ticker_path = os.path.join(root_file_path, ticker_folder)
            if os.path.isdir(ticker_path):
                close_data_path = os.path.join(ticker_path, 'close.1day.parquet')
                if os.path.exists(close_data_path):
                    ticker_data = pd.read_parquet(close_data_path)
                    all_trading_days.update(ticker_data['date'].unique())

        all_trading_days = sorted(all_trading_days)
        trading_calendar_path = "trading_calendar"
        self.store_data(trading_calendar_path, pd.DataFrame({'date': all_trading_days}), processed=True)
        print(f"Trading calendar generated and stored with {len(all_trading_days)} unique dates.")


    def generate_instrument_list(self, tickers: List[str]) -> None:
        """
        Generates and stores the instrument list with start and end dates for each ticker.
        """
        instrument_data = []

        root_file_path = os.path.join(self.data_lake.storage_path, 'features')
        for ticker in tickers:
            ticker_folder = os.path.join(root_file_path, ticker)
            if os.path.isdir(ticker_folder):
                close_data_path = os.path.join(ticker_folder, 'close.1day.parquet')
                if os.path.exists(close_data_path):
                    ticker_data = pd.read_parquet(close_data_path)
                    start_date = ticker_data['date'].min()
                    end_date = ticker_data['date'].max()
                    instrument_data.append({'ticker': ticker, 'start_date': start_date, 'end_date': end_date})
                else:
                    print(f"Close data not found for ticker '{ticker}'. Skipping...")
            else:
                print(f"No data folder found for ticker '{ticker}'. Skipping...")

        if instrument_data:
            instrument_list_df = pd.DataFrame(instrument_data)
            instrument_list_path = "instruments"
            self.store_data(instrument_list_path, instrument_list_df, processed=True)
            print(f"Instrument list generated and stored with {len(instrument_data)} tickers, including start and end dates.")
        else:
            print("No valid instrument data found.")


#%%
if __name__ == "__main__":
    instruments = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'TSLA']

    data_lake = DataLake(storage_path='./data_lake/equity')
    manager = QuandlDataManager(data_lake=data_lake, api_key=api_key)
    
    # Get volume and price data
    start_date = '2024-01-01'
    end_date = '2024-11-17'
    manager.process_quotemedia_prices(instruments,start_date, end_date)
    
    start_date = '2024-09-20'
    end_date = '2024-11-17'
    manager.process_yfinance_intraday_prices(instruments,start_date, end_date, interval='30m')
    
    # Define table configurations
    table_configs = [
        {
            'table_name': 'ZACKS/FC',
            'start_date': '2020-01-01',
            'end_date': '2024-01-01',
            'keep_cols': ['eps_diluted_net', 'basic_net_eps', 'tot_lterm_debt', 'net_lterm_debt', 'net_curr_debt', 'tot_revnu'],
            'date_col': 'per_end_date',
            'sort_by': ['per_end_date']
        },
        {
            'table_name': 'ZACKS/FR',
            'start_date': '2020-01-01',
            'end_date': '2024-01-01',
            'keep_cols': ['ret_invst', 'tot_debt_tot_equity', 'free_cash_flow_per_share'],
            'date_col': 'per_end_date',
            'sort_by': ['per_end_date']
        },
        {
            'table_name': 'ZACKS/MKTV',
            'start_date': '2020-01-01',
            'end_date': '2024-01-01',
            'keep_cols': ['mkt_val'],
            'date_col': 'per_end_date',
            'sort_by': ['per_end_date']
        },
        {
            'table_name': 'ZACKS/SHRS',
            'start_date': '2020-01-01',
            'end_date': '2024-01-01',
            'keep_cols': ['shares_out'],
            'date_col': 'per_end_date',
            'sort_by': ['per_end_date']
        }
    ]

    # Process each table configuration
    for config in table_configs:
        manager.process_quandl_table(
            instruments=instruments,
            table_name=config['table_name'],
            start_date=config['start_date'],
            end_date=config['end_date'],
            keep_cols=config['keep_cols'],
            date_col=config['date_col'],
            sort_by=config['sort_by'],
            process_zscore=False
        )
        
    start_date = '2024-10-17'
    end_date = '2024-11-17'
    news_manager = NewsDataManager(data_lake=data_lake, api_key='2e0583fa971345548a0517941b752945')
    company_instrument_tuples = [("Apple","AAPL"),("Amazon","AMZN"),("Alphabet","GOOGL"),("MSFT","Microsoft"),("Tesla","TSLA")]
    news_manager.process_news(company_instrument_tuples, start_date=start_date, end_date=end_date)
