# Quant Data Platform

A modular data platform designed to streamline the workflow for quantitative researchers — from data discovery to research-ready data pipelines.

## Objective

This platform empowers quant researchers to:
- Discover and explore datasets efficiently  
- Access and transform raw or processed data with minimal friction  
- Standardize data access through structured models  
- Perform event-driven analyses using intraday and news data

---

## Platform Components

### Data Lake

The Data Lake serves as the central storage system for both raw and processed financial datasets. It supports multi-source ingestion, including fundamental data, intraday pricing, and financial news.

**Key Features:**
- Supports multiple data sources:
  - **Quandl**: financial reports (e.g., earnings, debt ratios)
  - **YFinance**: intraday price and volume data
  - **NewsAPI**: company-level news headlines
- Stores both **raw and transformed** data in Parquet format
- Organized **by ticker symbol** and **dataset name**
- Access via flexible query interface using `ticker`, `dataset_name`, and time range
- Supports batch ingestion and custom table configurations

**Storage Path Example:**
/data_lake/equity/
├── AAPL/
│ ├── intraday.parquet
│ ├── ZACKS_FC.parquet
├── TSLA/
│ ├── news.parquet

**Example Usage (from main pipeline):**
```python
data_lake = DataLake(storage_path='./data_lake/equity')
manager = QuandlDataManager(data_lake=data_lake, api_key=api_key)

# Download and store intraday pricing data
manager.process_yfinance_intraday_prices(
    instruments=['AAPL', 'AMZN', 'GOOGL'],
    start_date='2024-09-20',
    end_date='2024-11-17',
    interval='30m'
)

# Download and store financial statement data (e.g. ZACKS/FC)
manager.process_quandl_table(
    instruments=['AAPL', 'TSLA'],
    table_name='ZACKS/FC',
    start_date='2020-01-01',
    end_date='2024-01-01',
    keep_cols=['eps_diluted_net', 'tot_revnu'],
    date_col='per_end_date',
    sort_by=['per_end_date']
)

### Data Catalog

The **Data Catalog** provides a structured, metadata-rich inventory of all datasets stored in the Data Lake. It is organized by **user-defined categories** (e.g., intraday pricing, fundamentals, news) and supports efficient **search**, **indexing**, and **retrieval** for quantitative workflows.

#### Key Features

- **Hierarchical Categorization**: Datasets are grouped into named categories (e.g., `"PV_LF"`, `"FC"`).
- **Rich Metadata**: Each dataset includes metadata fields like `source`, `frequency`, `table_name`, and `type`, enabling precise filtering.
- **Flexible Search Capabilities**:
  - **By Keyword**: Search dataset `name` or `description`
  - **By Metadata**: Search by metadata fields (e.g., `source=Quandl`)
- **Persistence**:
  - Catalog is saved to and loaded from `categories.json`
  - Easy to clear, reload, or modify
- **Data Retrieval Support**:
  - Retrieve by dataset name
  - Retrieve all datasets matching metadata filters

---

#### Sample Dataset Metadata Schema

{
  "name": "close.1day",
  "description": "Daily closing price for the asset.",
  "metadata": {
    "source": "Quandl",
    "frequency": "1 day",
    "type": "Price Volume"
  },
  "processed": true
}


---

#### Example Usage

```python
# Initialize catalog
catalog = DataCatalog(storage_path="./data_lake/equity")

# Add a new category
catalog.add_category("PV_LF")

# Add datasets with metadata
catalog.add_datasets("PV_LF", [
    {
        "name": "close.1day",
        "description": "Daily closing price for the asset.",
        "metadata": {"source": "Quandl", "frequency": "1 day", "type": "Price Volume"},
        "processed": True
    }
])

# Save and reload
catalog.save_catalog()
catalog.load_catalog()

# List all categories
print(catalog.list_categories())

# Search datasets by keyword
print(catalog.search_by_keyword("debt"))

# Search by metadata
datasets = catalog.search_by_metadata("source", "Quandl")

# Retrieve Parquet dataset
df = catalog.retrieve_data("close.1day")

### Data Workbench

The **Data Workbench** is the transformation and processing layer of the platform. It provides quant researchers with a programmable interface for retrieving, cleaning, aggregating, and transforming datasets before they are used in modeling or analysis workflows.

It tightly integrates with the **Data Lake** and **Data Catalog**, supporting both single-asset and multi-asset operations across raw or processed datasets.

---

#### Key Features

- **Flexible Data Retrieval**
  - Retrieve datasets by ticker and name
  - Combine a dataset (e.g. `"close"`) across multiple instruments
  - Aggregate all datasets in a category (e.g. `"FC"`) for a single ticker

- **Cleaning & Preprocessing**
  - Automatic missing value handling (e.g., fillna or interpolation)
  - Automatic datetime conversion for key date columns
  - Works with datasets loaded through the Workbench or retrieved on the fly

- **Transformation & Storage**
  - Apply any custom transformation function (e.g. `log`, `normalize`)
  - Store transformed results back into the Data Lake (as processed datasets)
  - Supports transformation pipelines

---

#### Typical Workflow

```python
# Step 1: Initialize platform components
data_catalog = DataCatalog("./data_lake/equity")
data_lake = DataLake("./data_lake/equity")
workbench = DataWorkbench(data_catalog, data_lake)

# Step 2: Retrieve data by ticker and dataset name
df = workbench.retrieve_data("AAPL", "eps_diluted_net")

# Step 3: Clean dataset in-place (handles NaNs, formats dates)
workbench.clean_data("eps_diluted_net")
cleaned_df = workbench.datasets["eps_diluted_net"]

# Step 4: Retrieve cross-sectional dataset across instruments
combined = workbench.retrieve_data_by_dataset(
    dataset_name="close",
    instruments=["AAPL", "MSFT", "TSLA"],
    frequency="1day",
    start_date="2024-01-01",
    end_date="2024-11-17"
)

# Step 5: Retrieve all financials under category 'FC' for AAPL
fc_data = workbench.retrieve_data_by_category(
    category_name="FC",
    ticker="AAPL",
    start_date="2020-01-01",
    end_date="2024-11-17"
)

### Quant Data Models

The **Quant Data Models** layer defines reusable, standardized data structures for key financial objects — such as intraday prices and news articles — to ensure consistency across transformations, retrievals, and modeling workflows.

These classes abstract away the raw data structures and expose intuitive, purpose-specific methods for quantitative analysis.

---

#### IntradayDataModel

Used for high-frequency price and volume data across minute/hour intervals.

**Attributes:**
- `timestamp` – Timestamps for intraday bars  
- `close` – Closing prices  
- `volume` – Trading volume  
- `symbol` – Ticker identifier

**Key Method:**
```python
aggregate_by_interval(interval: str) → pd.DataFrame

---

#### Example Usage

intraday_model = IntradayDataModel(
    timestamp=intraday_data["date"],
    close=intraday_data["close"],
    volume=intraday_data["volume"],
    symbol="AAPL"
)

aggregated = intraday_model.aggregate_by_interval("30min")




