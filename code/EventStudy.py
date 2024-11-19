import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import QuantDataModels
import importlib
importlib.reload(QuantDataModels)
from QuantDataModels import NewsDataModel, IntradayDataModel
from DataLake import DataLake
import DataWorkbench
importlib.reload(DataWorkbench)
from DataCatalog import DataCatalog
from DataWorkbench import DataWorkbench
import os

class EventStudy:
    def __init__(self, data_catalog: DataCatalog, data_lake: DataLake, data_workbench: DataWorkbench):
        """
        Initialize the EventStudy class with the platform components.
        """
        self.data_catalog = data_catalog
        self.data_lake = data_lake
        self.data_workbench = data_workbench
        self.news_model = None  # Initialize with NewsDataModel later
        self.intraday_model = None  # Will initialize IntradayDataModel per ticker

    def retrieve_data(self, ticker: str, news_start_date: str, news_end_date: str, intraday_start_date: str, intraday_end_date: str):
        """
        Retrieve news and intraday data for the given ticker and date ranges.
        """
        # Retrieve news data
        news_data = self.data_workbench.retrieve_data(
            ticker=ticker,
            dataset_name="news",
            start_date=news_start_date,
            end_date=news_end_date
        )
        if news_data.empty:
            raise ValueError(f"No news data found for ticker '{ticker}' between {news_start_date} and {news_end_date}.")

        # Analyze sentiment for news data
        news_data = NewsDataModel.analyze_dataframe(news_data, headline_column="title", sentiment_column="sentiment")
        print("News sentiment analysis completed.")

        # Retrieve intraday data
        intraday_data = self.data_workbench.retrieve_data_by_category(
            category_name='PV_HF',
            ticker=ticker,
            frequency="30m",
            start_date=intraday_start_date,
            end_date=intraday_end_date
        )
        if intraday_data.empty:
            raise ValueError(f"No intraday data found for ticker '{ticker}' between {intraday_start_date} and {intraday_end_date}.")

        # Initialize IntradayDataModel
        self.intraday_model = IntradayDataModel(
            timestamp=intraday_data["date"],
            close=intraday_data["close"],
            volume=intraday_data["volume"],
            symbol=ticker
        )

        return news_data, intraday_data

    def aggregate_intraday_data(self, interval: str):
        """
        Aggregate intraday data by the specified interval (e.g., '60min').
        """
        if self.intraday_model is None:
            raise ValueError("IntradayDataModel has not been initialized.")
        aggregated_data = self.intraday_model.aggregate_by_interval(interval=interval)
        print(f"Intraday data aggregated to {interval} intervals.")
        return aggregated_data

    def analyze_event_impact(self, news_data: pd.DataFrame, intraday_data: pd.DataFrame, event_window: int = 120, sentiment_threshold: float = 0.5):
        """
        Analyze the relationship between news sentiment and intraday price movements.

        Parameters:
            news_data (pd.DataFrame): News data with sentiment scores.
            intraday_data (pd.DataFrame): Aggregated intraday data.
            event_window (int): The time window (in minutes) around each news event.
            sentiment_threshold (float): Minimum sentiment score for significant news events.
        """
        # Filter significant news events
        #import pdb; pdb.set_trace()
        significant_news = NewsDataModel.filter_dataframe_by_sentiment(
            news_data, sentiment_column="sentiment", threshold=sentiment_threshold
        )
        if significant_news.empty:
            raise ValueError("No significant news events found.")

        # Perform event study
        results = []
        for _, news_event in significant_news.iterrows():
            event_time = pd.Timestamp(news_event["date"])
            start_window = event_time - pd.Timedelta(minutes=event_window)
            end_window = event_time + pd.Timedelta(minutes=event_window)

            # Extract intraday data within the event window
            window_data = intraday_data[
                (intraday_data["timestamp"] >= start_window) &
                (intraday_data["timestamp"] <= end_window)
            ].copy()

            if not window_data.empty:
                # Calculate relative time from the event
                window_data["time_offset"] = (window_data["timestamp"] - event_time).dt.total_seconds() / 60
                window_data["event_sentiment"] = news_event["sentiment"]
                results.append(window_data)

        if not results:
            raise ValueError("No intraday data found within event windows.")

        event_study_df = pd.concat(results, ignore_index=True)
        return event_study_df

    def visualize_event_impact(self, event_study_df: pd.DataFrame, output_path: str = "./outcome", filename: str = "event_impact_plot.png"):
        """
        Visualize the average price movements around news events.

        Parameters:
            event_study_df (pd.DataFrame): DataFrame from event study analysis.
            output_path (str): Directory to save the plot.
            filename (str): Filename for the saved plot.
        """
        #import pdb; pdb.set_trace()
        event_study_df["time_offset_5m"] = (event_study_df["time_offset"] // 5) * 5
        avg_price = event_study_df.groupby("time_offset_5m")["close"].mean().reset_index()

        # Plot the average price movement
        plt.figure(figsize=(12, 8))
        plt.plot(avg_price["time_offset_5m"], avg_price["close"], marker="o", label="Average Price Movement")
        plt.axvline(0, color="red", linestyle="--", label="Event Time")
        plt.title("Average Price Movements Around News Events", fontsize=16)
        plt.xlabel("Minutes from Event", fontsize=12)
        plt.ylabel("Average Close Price", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.5, linestyle="--")
        plt.tight_layout()

        # Save the plot
        os.makedirs(output_path, exist_ok=True)
        plot_path = os.path.join(output_path, filename)
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved to {plot_path}")

        plt.show()


# Example Usage
if __name__ == "__main__":
    # Initialize platform components
    data_catalog = DataCatalog(storage_path="./data_lake/equity")
    data_lake = DataLake(storage_path="./data_lake/equity")
    data_workbench = DataWorkbench(data_catalog, data_lake)

    # Initialize EventStudy
    event_study = EventStudy(data_catalog, data_lake, data_workbench)

    # Parameters
    ticker = "AAPL"
    news_start_date = "2024-10-17"
    news_end_date = "2024-11-17"
    intraday_start_date = "2024-10-17"
    intraday_end_date = "2024-11-17"
    event_window = 60  # 60-minute window

    # Perform event study
    try:
        news_data, intraday_data = event_study.retrieve_data(
            ticker=ticker,
            news_start_date=news_start_date,
            news_end_date=news_end_date,
            intraday_start_date=intraday_start_date,
            intraday_end_date=intraday_end_date
        )

        # Aggregate intraday data
        intraday_aggregated = event_study.aggregate_intraday_data(interval="30min")

        # Analyze event impact
        event_study_df = event_study.analyze_event_impact(
            news_data=news_data,
            intraday_data=intraday_aggregated,
            event_window=event_window
        )
        #import pdb;pdb.set_trace()

        # Visualize results
        event_study.visualize_event_impact(event_study_df)

    except Exception as e:
        print(f"Error: {e}")
