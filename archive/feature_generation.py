import re
from statistics import mean

import nltk
import pandas as pd
from better_profanity import profanity
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download('vader_lexicon')


class FeatureGenerator(BaseEstimator, TransformerMixin):
    excl_pattern = r'^!+$'
    sia = SentimentIntensityAnalyzer()

    def __init__(self):
        pass

    @classmethod
    def _basic_row_cleaning(self, text: str):
        split_text = text.split()
        if re.match(self.excl_pattern, split_text[0]):
            split_text = split_text[1:]
        return " ".join([word for word in split_text if not (re.match(r"@\w+", word) or word == "RT")])

    @classmethod
    def _annotate_rt(self, text: str):
        split_text = text.split()
        return 1 if len(split_text) > 0 and 'RT' in split_text else 0

    @classmethod
    def _contains_profanity_words(self, text: str):
        return int(profanity.contains_profanity(text))

    @classmethod
    def _sentiment_score_of_row(self, text: str):
        return self.sia.polarity_scores(text)['compound']

    @classmethod
    def _number_of_words_in_row(self, text: str):
        return len(text.split())

    @classmethod
    def _average_len_of_words(self, text: str):
        return mean([len(word) for word in text.split()])

    @classmethod
    def _number_of_commas(self, text: str):
        return text.count(",")

    @classmethod
    def _number_of_excl_points(self, text: str):
        return text.count("!")

    @classmethod
    def _number_of_question_marks(self, text: str):
        return text.count("?")

    @classmethod
    def _number_of_full_stops(self, text: str):
        return text.count(".")

    def fit(self, df: pd.DataFrame, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        rt_flag = X.apply(lambda row: self._number_of_full_stops(row['tweet']), axis=1)

        cleaned = X.apply(lambda row: self._basic_row_cleaning(row['tweet']), axis=1)

        contains_profanity_words_flag = cleaned.apply(self._contains_profanity_words)
        sentiment_score_of_row = cleaned.apply(self._sentiment_score_of_row)
        number_of_words_in_row = cleaned.apply(self._number_of_words_in_row)
        number_of_commas = cleaned.apply(self._number_of_commas)
        number_of_excl_points = cleaned.apply(self._number_of_excl_points)
        number_of_question_marks = cleaned.apply(self._number_of_question_marks)
        number_of_full_stops = cleaned.apply(self._number_of_full_stops)

        new_columns = {
            'rt_flag': rt_flag,
            'contains_profanity_words_flag': contains_profanity_words_flag,
            'sentiment_score_of_row': sentiment_score_of_row,
            'number_of_words_in_row': number_of_words_in_row,
            'number_of_commas': number_of_commas,
            'number_of_excl_points': number_of_excl_points,
            'number_of_question_marks': number_of_question_marks,
            'number_of_full_stops': number_of_full_stops
        }

        return pd.DataFrame(new_columns)
