import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class TextPreprocessor:
    def clean_text(self, text):
        """Cleans raw text by removing punctuation, links, numbers, and stopwords."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
        text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
        text = re.sub(r"\d+", "", text)  # remove numbers
        tokens = text.split()
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
        return " ".join(tokens)

    def get_text_length(self, text):
        return len(text) if isinstance(text, str) else 0

    def get_word_count(self, text):
        return len(str(text).split()) if isinstance(text, str) else 0

    def has_link(self, text):
        if not isinstance(text, str):
            return 0
        return int("http" in text.lower() or "www" in text.lower())

    def has_dollar_sign(self, text):
        return int("$" in str(text)) if isinstance(text, str) else 0

    def preprocess_dataframe(self, dataFrame: pd.DataFrame):
        """Applies all preprocessing and feature engineering to the dataframe."""
        dataFrame = dataFrame.copy()

        # Clean the 'Message' and combined 'Subject + Message'
        dataFrame["cleaned_text"] = dataFrame["Message"].apply(self.clean_text)
        dataFrame["combined"] = (
            dataFrame["Subject"] + " " + dataFrame["Message"]
        ).apply(self.clean_text)

        # Feature Engineering
        dataFrame["text_length"] = dataFrame["Message"].apply(self.get_text_length)
        dataFrame["word_count"] = dataFrame["Message"].apply(self.get_word_count)
        dataFrame["has_link"] = dataFrame["Message"].apply(self.has_link)
        dataFrame["has_dollar"] = dataFrame["Message"].apply(self.has_dollar_sign)

        return dataFrame
