import pandas as pd
import importlib
from DataCatalog import DataCatalog
import DataLake  
from typing import Optional, List, Dict, Callable
importlib.reload(DataLake)
from DataLake import DataLake
import re


class DataWorkbench:
    """
    The Data Workbench provides a workspace for transforming, processing, 
    and structuring data for analysis. It integrates with DataCatalog for 
    dataset management and retrieval.
    """

    def __init__(self, data_catalog, data_lake):
        """
        Initialize the DataWorkbench with a DataCatalog and a DataLake.

        Parameters:
            data_catalog (DataCatalog): Instance of the DataCatalog.
            data_lake (DataLake): Instance of the DataLake.
        """
        self.data_catalog = data_catalog
        self.data_lake = data_lake
        self.datasets = {}  # Temporary storage for datasets loaded into the workbench

    def retrieve_data(self, ticker: str, dataset_name: str, frequency: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Retrieves a dataset from the DataLake based on category and dataset name.

        Parameters:
            category_name (str): The name of the category in the DataCatalog.
            dataset_name (str): The name of the dataset to retrieve.

        Returns:
            pd.DataFrame: The requested dataset as a DataFrame.
        """

        data = self.data_lake.retrieve_data(ticker,dataset_name,frequency,start_date,end_date)
        if data is not None:
            self.datasets[dataset_name] = data
            return data
        else:
            print(f"Dataset '{dataset_name}' not found in category '{category_name}'.")
            return None
        
    def retrieve_data_by_dataset(self, dataset_name: str, instruments: List[str], frequency: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Retrieves and combines data for a given dataset across multiple instruments.

        Parameters:
            dataset_name (str): The name of the dataset to retrieve (e.g., 'close', 'volume').
            instruments (List[str]): List of stock tickers to retrieve data for.
            frequency (Optional[str]): Data frequency (e.g., '1day', '1hour').
            start_date (Optional[str]): Start date for the data retrieval (in 'YYYY-MM-DD' format).
            end_date (Optional[str]): End date for the data retrieval (in 'YYYY-MM-DD' format).

        Returns:
            pd.DataFrame: Combined dataset for the given instruments.
        """
        combined_data = []

        for ticker in instruments:
            print(f"Retrieving data for ticker '{ticker}' and dataset '{dataset_name}'...")
            data = self.data_lake.retrieve_data(ticker, dataset_name, frequency, start_date, end_date)

            if data is not None:
                # Add a ticker column to identify the source of the data
                data["ticker"] = ticker
                combined_data.append(data)
            else:
                print(f"No data found for ticker '{ticker}' and dataset '{dataset_name}'.")

        if combined_data:
            # Concatenate all data into a single DataFrame
            combined_df = pd.concat(combined_data, ignore_index=True)
            print(f"Combined data contains {len(combined_df)} records across {len(instruments)} instruments.")
            return combined_df
        else:
            print(f"No data retrieved for dataset '{dataset_name}' across the provided instruments.")
            return None
        
    def retrieve_data_by_category(self, category_name: str, ticker: str, frequency: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Retrieves and combines all datasets under a specific category for a given ticker.

        Parameters:
            category_name (str): The name of the category in the DataCatalog.
            ticker (str): The ticker symbol for which data is retrieved.
            frequency (Optional[str]): Data frequency (e.g., '1day', '1hour').
            start_date (Optional[str]): Start date for the data retrieval (in 'YYYY-MM-DD' format).
            end_date (Optional[str]): End date for the data retrieval (in 'YYYY-MM-DD' format).

        Returns:
            pd.DataFrame: Combined dataset containing all fields for the specified ticker and category.
        """
        if category_name not in self.data_catalog.categories:
            print(f"Category '{category_name}' not found in the DataCatalog.")
            return None

        datasets = self.data_catalog.list_datasets(category_name)
        if not datasets:
            print(f"No datasets found in category '{category_name}'.")
            return None

        combined_data = []

        for dataset in datasets:
            dataset_name = dataset["name"]
            print(f"Retrieving dataset '{dataset_name}' for ticker '{ticker}'...")
            if frequency is not None and frequency in dataset_name:
                dataset_name = re.sub(rf"\.{frequency}$", "", dataset_name)
            data = self.data_lake.retrieve_data(ticker, dataset_name, frequency, start_date, end_date)

            if data is not None:
                # Add a dataset name column to identify the source of the data
                data["dataset"] = dataset_name
                combined_data.append(data)
            else:
                print(f"No data found for dataset '{dataset_name}' under ticker '{ticker}'.")

        if combined_data:
            # Merge all datasets on their common columns (e.g., 'date')
            combined_df = pd.concat(combined_data, axis=1, join="inner").loc[:, ~pd.concat(combined_data, axis=1, join="inner").columns.duplicated()]
            print(f"Combined data contains {len(combined_df)} records for ticker '{ticker}' in category '{category_name}'.")
            return combined_df
        else:
            print(f"No data retrieved for category '{category_name}' under ticker '{ticker}'.")
            return None

    def clean_data(self, name: str):
        """
        Clean the dataset by handling missing values and correcting data types.

        Parameters:
            name (str): The name of the dataset to clean.
        """
        if name in self.datasets:
            data = self.datasets[name]
            # Handle missing values (e.g., fill with 0 or interpolate)
            data.fillna(0, inplace=True)

            # Convert date columns to datetime format if present
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            elif 'per_end_date' in data.columns:
                data['per_end_date'] = pd.to_datetime(data['per_end_date'])

            self.datasets[name] = data
            print(f"Dataset '{name}' cleaned successfully.")
        else:
            print(f"Dataset '{name}' not found in the workbench.")

    def transform_data(self, dataset: pd.DataFrame, transformation_func: Callable) -> pd.DataFrame:
        """
        Transforms data using a specified transformation function.

        Parameters:
            dataset (pd.DataFrame): The dataset to transform.
            transformation_func (Callable): A function to transform the dataset.

        Returns:
            pd.DataFrame: Transformed dataset.
        """
        if dataset is None or dataset.empty:
            raise ValueError("Dataset is empty or None. Cannot transform data.")
        return transformation_func(dataset)

    def store_transformed_data(self, dataset_name: str, data: pd.DataFrame, processed: bool = True):
        """
        Stores transformed data back to the DataLake.

        Parameters:
            dataset_name (str): The name of the dataset to store.
            data (pd.DataFrame): The transformed dataset.
            processed (bool): Whether the data is processed.
        """
        if data is None or data.empty:
            raise ValueError("Data to store is empty or None.")
        self.data_lake.store_data(dataset_name, data, processed)
        print(f"Transformed data stored for dataset '{dataset_name}'.")

    def apply_transformation_and_store(
        self, category_name: str, dataset_name: str, transformation_func: Callable
    ):
        """
        Applies a transformation to a dataset and stores the result.

        Parameters:
            category_name (str): The category of the dataset.
            dataset_name (str): The name of the dataset.
            transformation_func (Callable): The transformation function to apply.
        """
        print(f"Retrieving dataset '{dataset_name}' from category '{category_name}'...")
        data = self.retrieve_data(category_name, dataset_name)

        if data is not None:
            print(f"Applying transformation to dataset '{dataset_name}'...")
            transformed_data = self.transform_data(data, transformation_func)
            self.store_transformed_data(dataset_name=f"{dataset_name}_transformed", data=transformed_data)
        else:
            print(f"Dataset '{dataset_name}' not found or is empty. No transformation applied.")
            
    def filter_data(
        self, 
        dataset_name: str, 
        filter_func: Callable[[pd.DataFrame], pd.DataFrame], 
        inplace: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Filters a dataset using a specified filtering function.

        Parameters:
            dataset_name (str): The name of the dataset to filter.
            filter_func (Callable): A function that takes a DataFrame and returns a filtered DataFrame.
            inplace (bool): Whether to modify the dataset in the workbench directly.

        Returns:
            pd.DataFrame: The filtered dataset if inplace is False, otherwise None.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found in the workbench.")

        data = self.datasets[dataset_name]
        
        if data.empty:
            print(f"Dataset '{dataset_name}' is empty. No filtering applied.")
            return None

        # Apply the filter function
        filtered_data = filter_func(data)

        if inplace:
            self.datasets[dataset_name] = filtered_data
            print(f"Dataset '{dataset_name}' filtered and updated in place.")
            return None
        else:
            print(f"Dataset '{dataset_name}' filtered. Returning a new DataFrame.")
            return filtered_data

    def filter_by_date_range(
        self, dataset_name: str, start_date: str, end_date: str, date_column: str = "date", inplace: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Filters a dataset by a specific date range.

        Parameters:
            dataset_name (str): The name of the dataset to filter.
            start_date (str): The start date of the filter range (inclusive).
            end_date (str): The end date of the filter range (inclusive).
            date_column (str): The name of the column containing date values.
            inplace (bool): Whether to modify the dataset in the workbench directly.

        Returns:
            pd.DataFrame: The filtered dataset if inplace is False, otherwise None.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found in the workbench.")

        data = self.datasets[dataset_name]

        if date_column not in data.columns:
            raise ValueError(f"Column '{date_column}' not found in dataset '{dataset_name}'.")

        # Convert the date column to datetime for filtering
        data[date_column] = pd.to_datetime(data[date_column])

        # Define the filtering function
        def filter_func(df):
            return df[(df[date_column] >= pd.Timestamp(start_date)) & (df[date_column] <= pd.Timestamp(end_date))]

        # Use the general filtering method
        return self.filter_data(dataset_name, filter_func, inplace)

    def filter_by_column_value(
        self, dataset_name: str, column: str, value: str, inplace: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Filters a dataset by a specific column value.

        Parameters:
            dataset_name (str): The name of the dataset to filter.
            column (str): The name of the column to filter on.
            value (str): The value to filter by.
            inplace (bool): Whether to modify the dataset in the workbench directly.

        Returns:
            pd.DataFrame: The filtered dataset if inplace is False, otherwise None.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found in the workbench.")

        data = self.datasets[dataset_name]

        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in dataset '{dataset_name}'.")

        # Define the filtering function
        def filter_func(df):
            return df[df[column] == value]

        # Use the general filtering method
        return self.filter_data(dataset_name, filter_func, inplace)
            
            

if __name__ == "__main__":
        
    # Initialize the DataCatalog and DataLake
    data_catalog = DataCatalog(storage_path="./data_lake/equity")
    data_lake = DataLake(storage_path="./data_lake/equity")

    # Initialize the DataWorkbench
    data_workbench = DataWorkbench(data_catalog, data_lake)

    # Retrieve and clean a dataset
    category_name = "FC"
    dataset_name = "eps_diluted_net"
    data = data_workbench.retrieve_data('AAPL',dataset_name)
    if data is not None:
        data_workbench.clean_data(dataset_name)
        cleaned_data = data_workbench.datasets[dataset_name]

        # Store cleaned data back to the DataLake
        # data_workbench.store_transformed_data(f"{dataset_name}_cleaned", cleaned_data)

    # Retrieve and combine data by dataset
    combined_data = data_workbench.retrieve_data_by_dataset(
        dataset_name="close",
        instruments=["AAPL", "MSFT", "TSLA"],
        frequency="1day",
        start_date="2024-01-01",
        end_date="2024-11-17"
    )

    # Retrieve and combine data by category
    combined_data = data_workbench.retrieve_data_by_category(
        category_name= "FC",
        ticker="AAPL",
        start_date="2020-01-01",
        end_date="2024-11-17"
    )

