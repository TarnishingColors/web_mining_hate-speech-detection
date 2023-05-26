import html
import re
from typing import Callable

import nltk
import pandas as pd
import torch
from emot.emo_unicode import UNICODE_EMOJI  # For emojis
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import TransformerMixin, BaseEstimator
from transformers import BertTokenizer, BertModel

# Specify the pre-trained model name.
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'  # uncased means this tokenizer will also lower-case automatically

# Load the BERT tokenizer.
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


class SeriesConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X['tweet']


class TextCleaner(TransformerMixin, BaseEstimator):
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

    re_username = r"@\w+"
    re_link = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
               '[!*,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    re_all_words = "[^a-z ]+"
    stop_words = set(stopwords.words('english'))

    lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X \
            .pipe(self.apply_to_tweet_column(self.delete_usernames)) \
            .pipe(self.apply_to_tweet_column(self.delete_links)) \
            .pipe(self.apply_to_tweet_column(self.convert_emojis)) \
            .pipe(self.apply_to_tweet_column(self.to_lower)) \
            .pipe(self.apply_to_tweet_column(self.delete_special_symbols)) \
            .pipe(self.apply_to_tweet_column(self.lemmatize)) \
            .pipe(self.apply_to_tweet_column(self.remove_stopwords))

    def get_wordnet_pos(self, treebank_tag: str):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    @staticmethod
    def convert_emojis(text: str) -> str:
        text = html.unescape(text)
        for emot in UNICODE_EMOJI:
            text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",", "").replace(":", "").split()))
        return text

    def lemmatize(self, text: str) -> str:
        lemmatized = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos_tag))
            for (word, pos_tag) in nltk.pos_tag(nltk.word_tokenize(text))
        ]
        return " ".join(lemmatized)

    def remove_stopwords(self, tweet: str) -> str:
        tokens_without_sw = " ".join(
            [word for word in word_tokenize(tweet) if not word in [*self.stop_words, 'rt']]
        )
        return tokens_without_sw

    def delete_usernames(self, text: str) -> str:
        return re.sub(self.re_username, "username", text)

    def delete_links(self, text: str) -> str:
        return re.sub(self.re_link, "link", text)

    def delete_special_symbols(self, text: str):
        return re.sub(self.re_all_words, " ", text)

    @staticmethod
    def to_lower(text: str) -> str:
        return text.lower()

    def apply_to_tweet_column(self, f: Callable) -> Callable:
        return lambda x: pd.DataFrame(x['tweet'].apply(f))


class Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer: callable, **kargs):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = tokenizer

        self.config = {
            'add_special_tokens': True,
            'padding': 'max_length',
            'max_length': 4,
            'truncation': True,
            'pad_to_max_length': True,
            'return_attention_mask': False,
            'return_tensors': 'pt',
        }
        self.config.update(kargs)

    def fit(self, df: pd.DataFrame, y=None, **fit_params):
        return self

    def transform(self, df: pd.DataFrame, y=None, **transform_params):
        with torch.no_grad():
            _df = df \
                .apply(lambda x: self.tokenizer(x['tweet'], **self.config)['input_ids'], axis=1) \
                .apply(lambda x: pd.Series(self.model(x)[0].flatten()))

        return _df
