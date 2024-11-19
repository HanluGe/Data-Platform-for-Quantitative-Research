import os
import json
from typing import List, Dict, Optional
import pandas as pd

class DataCategory:
    """
    Represents a category in the DataCatalog. Each category contains datasets.
    """
    def __init__(self, name: str, datasets: Optional[List[Dict]] = None):
        self.name = name
        self.datasets = datasets if datasets else []

    def add_datasets(self, datasets: List[Dict]):
        """
        Adds multiple datasets to the category.
        """
        for dataset in datasets:
            if not any(d["name"] == dataset["name"] for d in self.datasets):
                self.datasets.append(dataset)

    def list_datasets(self) -> List[Dict]:
        """
        Lists all datasets in the category.
        """
        return self.datasets

    def search_by_keyword(self, keyword: str) -> List[Dict]:
        """
        Searches for datasets in the category by keyword.
        """
        return [
            dataset for dataset in self.datasets
            if keyword.lower() in dataset["name"].lower() or keyword.lower() in dataset["description"].lower()
        ]
        
    def search_by_metadata(self, metadata_key: str, metadata_value: str) -> List[Dict]:
        """
        Searches for datasets in the category by a specific metadata key and value.

        Parameters:
            metadata_key (str): The metadata key to search by.
            metadata_value (str): The metadata value to match.

        Returns:
            List[Dict]: A list of datasets matching the metadata criteria.
        """

        return [
            dataset for dataset in self.datasets
            if metadata_key in dataset['metadata'] and str(dataset['metadata'][metadata_key]).lower() == str(metadata_value).lower()
        ]

    def to_dict(self) -> Dict:
        """
        Converts the category to a dictionary for saving.
        """
        return {"name": self.name, "datasets": self.datasets}

    @staticmethod
    def from_dict(data: Dict) -> 'DataCategory':
        """
        Creates a DataCategory instance from a dictionary.
        """
        return DataCategory(name=data["name"], datasets=data["datasets"])


class DataCatalog:
    """
    A centralized catalog for managing datasets, organized by categories.
    """
    def __init__(self, storage_path: str):
        self.categories = {}
        self.storage_path = os.path.join(storage_path, "categories")
        os.makedirs(self.storage_path, exist_ok=True)

        # Load catalog from categories.json if it exists
        self.catalog_file = os.path.join(self.storage_path, "categories.json")
        if os.path.exists(self.catalog_file):
            self.load_catalog()
        else:
            print("No existing catalog found. Initialized an empty catalog.")

    def add_category(self, name: str):
        """
        Adds a new category to the catalog.
        """
        if name not in self.categories:
            self.categories[name] = DataCategory(name)

    def add_datasets(self, category_name: str, datasets: List[Dict]):
        """
        Adds multiple datasets to a specific category.
        """
        if category_name not in self.categories:
            raise ValueError(f"Category '{category_name}' does not exist. Please add it first.")
        self.categories[category_name].add_datasets(datasets)

    def list_categories(self) -> List[str]:
        """
        Lists all category names in the catalog.
        """
        return list(self.categories.keys())

    def list_datasets(self, category_name: str) -> List[Dict]:
        """
        Lists all datasets in a specific category.
        """
        if category_name in self.categories:
            return self.categories[category_name].list_datasets()
        raise ValueError(f"Category '{category_name}' not found.")

    def search_by_keyword(self, keyword: str) -> List[Dict]:
        """
        Searches for datasets across all categories by keyword.
        """
        results = []
        for category in self.categories.values():
            results.extend(category.search_by_keyword(keyword))
        return results
    
    def search_by_category(self, category_name: str) -> List[Dict]:
        """
        Searches for datasets within a specific category by keyword.

        Parameters:
            category_name (str): The name of the category to search in.
            keyword (str): The keyword to search for in dataset names or descriptions.

        Returns:
            List[Dict]: A list of matching datasets within the category.
        """
        if category_name not in self.categories:
            raise ValueError(f"Category '{category_name}' not found.")

        category = self.categories[category_name]
        return category.list_datasets()
    
    def search_by_metadata(self, metadata_key: str, metadata_value: str) -> List[Dict]:
        """
        Searches for datasets across all categories by metadata key and value.

        Parameters:
            metadata_key (str): The metadata key to search by.
            metadata_value (str): The metadata value to match.

        Returns:
            List[Dict]: A list of datasets matching the metadata criteria.
        """
        results = []
        for category in self.categories.values():
            results.extend(category.search_by_metadata(metadata_key, metadata_value))
        return results

    def retrieve_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Retrieve a dataset by its name from the storage path.

        Parameters:
            dataset_name (str): Name of the dataset to retrieve.

        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
        """
        dataset_path = os.path.join(self.storage_path, f"{dataset_name}.parquet")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found at path '{dataset_path}'.")

        print(f"Loading dataset from '{dataset_path}'...")
        return pd.read_parquet(dataset_path)
        
    
    def retrieve_datasets(self, key: str, value: str) -> List[pd.DataFrame]:
        """
        Retrieve datasets that match a specific metadata key-value pair.

        Parameters:
            key (str): Metadata key to filter datasets.
            value (str): Metadata value to match.

        Returns:
            List[pd.DataFrame]: A list of pandas DataFrames containing the retrieved datasets.
        """
        # Search for datasets matching the metadata
        matching_datasets = self.search_by_metadata(key, value)
        
        # List to store retrieved datasets
        retrieved_dataframes = []

        # Loop through the matching datasets and retrieve them
        for dataset in matching_datasets:
            dataset_name = dataset["name"]  # Extract dataset name
            try:
                # Retrieve the dataset using its name
                retrieved_df = self.retrieve_data(dataset_name)
                retrieved_dataframes.append(retrieved_df)
                print(f"Retrieved dataset '{dataset_name}' successfully.")
            except FileNotFoundError as e:
                # Handle missing datasets gracefully
                print(f"Error retrieving dataset '{dataset_name}': {e}")

        return retrieved_dataframes

    def save_catalog(self):
        """
        Saves the entire catalog to the categories.json file.
        """
        catalog_data = {name: category.to_dict() for name, category in self.categories.items()}
        with open(self.catalog_file, "w") as f:
            json.dump(catalog_data, f, indent=4)
        print(f"Catalog saved to '{self.catalog_file}'.")

    def load_catalog(self):
        """
        Loads the catalog from the categories.json file.
        """
        if os.path.exists(self.catalog_file):
            with open(self.catalog_file, "r") as f:
                catalog_data = json.load(f)
            self.categories = {name: DataCategory.from_dict(data) for name, data in catalog_data.items()}
            print(f"Catalog loaded from '{self.catalog_file}'.")
        else:
            print(f"Catalog file '{self.catalog_file}' does not exist.")
            
    def clear_catalog(self):
        """
        Clears the entire catalog, both in memory and on disk.
        """
        self.categories.clear()  # Clear in-memory catalog
        if os.path.exists(self.catalog_file):
            os.remove(self.catalog_file)  # Remove catalog file from disk
            print(f"Catalog file '{self.catalog_file}' has been deleted.")
        else:
            print("No catalog file exists to clear.")
        print("Catalog has been cleared.")



if __name__ == "__main__":

    storage_path = "./data_lake/equity"
    data_catalog = DataCatalog(storage_path=storage_path)

    data_catalog.clear_catalog()
    
    data_catalog.add_category("PV_LF")
    datasets_pv = [
        {"name": "close.1day", "description": "Daily closing price for the asset.", "metadata": {"source": "Quandl", "frequency": "1 day", "type": "Price Volume"}, "processed": True},
        {"name": "high.1day", "description": "Daily high price for the asset.", "metadata": {"source": "Quandl", "frequency": "1 day", "type": "Price Volume"}, "processed": True},
        {"name": "open.1day", "description": "Daily opening price for the asset.", "metadata": {"source": "Quandl", "frequency": "1 day", "type": "Price Volume"}, "processed": True},
        {"name": "low.1day", "description": "Daily low price for the asset.", "metadata": {"source": "Quandl", "frequency": "1 day", "type": "Price Volume"}, "processed": True},
        {"name": "volume.1day", "description": "Daily trading volume for the asset.", "metadata": {"source": "Quandl", "frequency": "1 day", "type": "Price Volume"}, "processed": True}
    ]
    data_catalog.add_datasets("PV_LF", datasets_pv)
    
    data_catalog.add_category("PV_HF")
    datasets_pv2 = [
        {"name": "close.30m", "description": "Intraday closing price for the asset.", "metadata": {"source": "Quandl", "frequency": "30 min", "type": "Price Volume"}, "processed": True},
        {"name": "high.30m", "description": "Intraday high price for the asset.", "metadata": {"source": "Quandl", "frequency": "30 min", "type": "Price Volume"}, "processed": True},
        {"name": "open.30m", "description": "Intraday opening price for the asset.", "metadata": {"source": "Quandl", "frequency": "30 min", "type": "Price Volume"}, "processed": True},
        {"name": "low.30m", "description": "Intraday low price for the asset.", "metadata": {"source": "Quandl", "frequency": "1 day", "30 min": "Price Volume"}, "processed": True},
        {"name": "volume.30m", "description": "Intraday trading volume for the asset.", "metadata": {"source": "Quandl", "frequency": "30 min", "type": "Price Volume"}, "processed": True}
    ]
    data_catalog.add_datasets("PV_HF", datasets_pv2)

    data_catalog.add_category("FC")
    datasets_fc = [
        {"name": "eps_diluted_net", "description": "Diluted net EPS from financial data.", "metadata": {"source": "Quandl", "table_name": "ZACKS/FC"}, "processed": True},
        {"name": "basic_net_eps", "description": "Basic net EPS from financial data.", "metadata": {"source": "Quandl", "table_name": "ZACKS/FC"}, "processed": True},
        {"name": "tot_lterm_debt", "description": "Total long-term debt from financial data.", "metadata": {"source": "Quandl", "table_name": "ZACKS/FC"}, "processed": True},
        {"name": "net_lterm_debt", "description": "Net long-term debt from financial data.", "metadata": {"source": "Quandl", "table_name": "ZACKS/FC"}, "processed": True},
        {"name": "net_curr_debt", "description": "Net current debt from financial data.", "metadata": {"source": "Quandl", "table_name": "ZACKS/FC"}, "processed": True},
        {"name": "tot_revnu", "description": "Total revenue from financial data.", "metadata": {"source": "Quandl", "table_name": "ZACKS/FC"}, "processed": True}
    ]
    data_catalog.add_datasets("FC", datasets_fc)

    # Add the "FR" category and datasets
    data_catalog.add_category("FR")
    datasets_fr = [
        {"name": "ret_invst", "description": "Return on investment from financial data.", "metadata": {"source": "Quandl", "table_name": "ZACKS/FR"}, "processed": True},
        {"name": "tot_debt_tot_equity", "description": "Total debt to total equity ratio.", "metadata": {"source": "Quandl", "table_name": "ZACKS/FR"}, "processed": True},
        {"name": "free_cash_flow_per_share", "description": "Free cash flow per share.", "metadata": {"source": "Quandl", "table_name": "ZACKS/FR"}, "processed": True}
    ]
    data_catalog.add_datasets("FR", datasets_fr)

    # Add the "MKTV" category and datasets
    data_catalog.add_category("MKTV")
    datasets_mktv = [
        {"name": "mkt_val", "description": "Market value from financial data.", "metadata": {"source": "Quandl", "table_name": "ZACKS/MKTV"}, "processed": True}
    ]
    data_catalog.add_datasets("MKTV", datasets_mktv)

    # Add the "SHRS" category and datasets
    data_catalog.add_category("SHRS")
    datasets_shrs = [
        {"name": "shares_out", "description": "Shares outstanding from financial data.", "metadata": {"source": "Quandl", "table_name": "ZACKS/SHRS"}, "processed": True}
    ]
    data_catalog.add_datasets("SHRS", datasets_shrs)

    data_catalog.save_catalog()

    loaded_catalog = DataCatalog(storage_path=storage_path)
    loaded_catalog.load_catalog()

    print("Categories:", loaded_catalog.list_categories())
    print("Datasets in Financials:", loaded_catalog.list_datasets("FC"))
    
    # search and retrieval based on keywords
    print("Search Results:", loaded_catalog.search_by_keyword("debt"))
    
    # search and retrieval based on category
    print("Search Results:", loaded_catalog.search_by_category("FC"))

    # search and retrieval based on metadata
    Quandl_datasets = data_catalog.search_by_metadata("source", "Quandl")
    print("Datasets from Quandl:", Quandl_datasets)
    
    

