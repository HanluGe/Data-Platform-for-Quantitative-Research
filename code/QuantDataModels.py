from transformers import pipeline
import pandas as pd
import numpy as np

class IntradayDataModel:
    def __init__(self, timestamp: pd.Series, close: pd.Series, volume: pd.Series, symbol: str):
        """
        Initialize the IntradayDataModel with intraday data.

        Parameters:
            timestamp (pd.Series): Timestamps of the intraday data.
            price (pd.Series): Prices of the intraday data.
            volume (pd.Series): Volume of the intraday data.
            symbol (str): Stock symbol.
        """
        self.timestamp = timestamp
        self.close = close
        self.volume = volume
        self.symbol = symbol
        
    def aggregate_by_interval(self, interval: str) -> pd.DataFrame:
        """
        Aggregates intraday data by a specified time interval.

        Parameters:
            interval (str): The time interval for aggregation (e.g., '60min', '2H', '1D').

        Returns:
            pd.DataFrame: A DataFrame with aggregated open, high, low, close, and volume values for each interval.
        """
        # Combine data into a single DataFrame
        data = pd.DataFrame({
            "timestamp": self.timestamp,
            "close": self.close,
            "volume": self.volume
        })

        # Set the timestamp as the index for resampling
        data.set_index("timestamp", inplace=True)

        # Resample the data to the specified interval
        aggregated = data.resample(interval).agg({
            "close": ["first", "max", "min", "last"],
            "volume": "sum"
        })

        # Rename the columns for clarity
        aggregated.columns = ["open", "high", "low", "close", "volume"]

        # Reset the index to make the DataFrame easier to work with
        aggregated.reset_index(inplace=True)

        return aggregated


class NewsDataModel:
    """
    A class to analyze sentiment scores for financial news headlines and manage individual articles.
    """
    def __init__(self, timestamp: pd.Timestamp, title: str, relevance: float):
        """
        Initializes an instance of NewsDataModel for a single news article.
        """
        self.timestamp = timestamp
        self.title = title
        self.sentiment_score = None  # Placeholder until sentiment is analyzed

        # Initialize the FinBERT sentiment analyzer (shared across all instances)
        if not hasattr(NewsDataModel, 'classifier'):
            try:
                NewsDataModel.classifier = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
            except Exception as e:
                raise RuntimeError("Failed to initialize FinBERT model. Ensure transformers library is installed.") from e

    def analyze_sentiment(self) -> float:
        """
        Analyze the sentiment of the current article's headline using FinBERT.
        """
        try:
            result = NewsDataModel.classifier(self.headline)[0]
            sentiment = result['label']
            score = result['score']
            import pdb; pdb.set_trace()
            self.sentiment_score = score if sentiment == 'POSITIVE' else -score
            return self.sentiment_score
        except Exception as e:
            print(f"Failed to analyze sentiment for headline '{self.headline}': {e}")
            self.sentiment_score = 0.0
            return self.sentiment_score

    @staticmethod
    def analyze_dataframe(df: pd.DataFrame, headline_column: str, sentiment_column: str = 'sentiment') -> pd.DataFrame:
        """
        Analyze sentiment for a DataFrame containing multiple news headlines.
        Adds a new column with sentiment scores.
        """
        if headline_column not in df.columns:
            raise ValueError(f"Headline column '{headline_column}' not found in DataFrame.")

        try:
            # Use FinBERT to analyze all headlines
            classifier = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
            df[sentiment_column] = df[headline_column].apply(lambda headline: 
                classifier(headline)[0]['score'] if classifier(headline)[0]['label'] == 'POSITIVE' 
                else -classifier(headline)[0]['score']
            )
            print(f"Analyzed sentiment for {len(df)} headlines.")
            return df
        except Exception as e:
            print(f"Failed to analyze sentiment for DataFrame: {e}")
            return df

    def filter_by_sentiment(self, threshold: float) -> bool:
        """
        Filters the current article based on a sentiment score threshold.

        Parameters:
            threshold (float): Sentiment score threshold for filtering.

        Returns:
            bool: True if the article's sentiment score is above the threshold, otherwise False.
        """
        if self.sentiment_score is None:
            raise ValueError("Sentiment score has not been analyzed yet. Call `analyze_sentiment()` first.")
        return abs(self.sentiment_score) >= threshold

    @staticmethod
    def filter_dataframe_by_sentiment(df: pd.DataFrame, sentiment_column: str, threshold: float) -> pd.DataFrame:
        """
        Filters a DataFrame of articles based on a sentiment score threshold.

        Parameters:
            df (pd.DataFrame): The DataFrame containing news articles with sentiment scores.
            sentiment_column (str): Column name containing sentiment scores.
            threshold (float): Sentiment score threshold for filtering.

        Returns:
            pd.DataFrame: Filtered DataFrame with articles meeting the threshold criteria.
        """
        if sentiment_column not in df.columns:
            raise ValueError(f"Sentiment column '{sentiment_column}' not found in DataFrame.")
        
        filtered_df = df[abs(df[sentiment_column]) >= threshold]
        print(f"Filtered DataFrame to {len(filtered_df)} articles with sentiment >= {threshold}.")
        return filtered_df
