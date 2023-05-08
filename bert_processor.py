import html
import re

import nltk
import pandas as pd
import torch
from emot.emo_unicode import UNICODE_EMOJI  # For emojis
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import TransformerMixin, BaseEstimator
from transformers import BertTokenizer, BertModel

# Specify the pre-trained model name.
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'  # uncased means this tokenizer will also lower-case automatically

# Load the BERT tokenizer.
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


# class for preprocessing
class DataframePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, func: callable):
        self.func = func

    def fit(self, df: pd.DataFrame, y=None, **fit_params):
        return self

    def transform(self, df: pd.DataFrame, y=None, **transform_params):
        _df = df.copy()
        _df["tweet"], _df["rt_flag"] = zip(*df["tweet"].transform(lambda x: self.func(x)))
        return _df


# Lemmatization

nltk.download('stopwords')
nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag):
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


# Functions for extracting 'rt_flag' feature
def annotate_rt(text):
    return 1 if len(text.split()) > 0 and text.split()[0] == 'rt' else 0


def delete_rt_info(text):
    if len(text) > 0:
        return text if text.split()[0] != 'rt' else " ".join(text.split()[1:])
    return text


# Function for converting emojis into word
def convert_emojis(text):
    text = html.unescape(text)
    for emot in UNICODE_EMOJI:
        text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",", "").replace(":", "").split()))
    return text


def lemmatize(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in tagged]

    return " ".join(lemmatized)


def remove_stopwords(tweet):
    tokens_tweet = word_tokenize(tweet)
    tokens_without_sw = " ".join([word for word in tokens_tweet if not word in stop_words])

    return tokens_without_sw


# also adds a new column 'rt_flag'
def preprocessing(tweet):
    re_username = '@[\w\-]+'
    re_exclamation = ""
    re_link = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
               '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    re_all_words = "[^a-z ]+"

    tweet = re.sub(re_username, "", tweet)
    tweet = re.sub(re_link, "link", tweet)
    tweet = tweet.lower()
    tweet = convert_emojis(tweet)
    tweet = re.sub(re_all_words, " ", tweet)
    tweet = lemmatize(tweet)
    tweet = remove_stopwords(tweet)
    rt_flag = annotate_rt(tweet)
    tweet = delete_rt_info(tweet)

    return tweet, rt_flag


class BertVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer: callable):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = tokenizer

        self.config = {
            'add_special_tokens': True,
            'padding': 'max_length',
            'max_length': 16,
            'truncation': True,
            'pad_to_max_length': True,
            'return_attention_mask': False,
            'return_tensors': 'pt',
        }

    def fit(self, df: pd.DataFrame, y=None, **fit_params):
        return self

    def transform(self, df: pd.DataFrame, y=None, **transform_params):
        with torch.no_grad():
            _df = df \
                .apply(lambda x: self.tokenizer(x['tweet'], **self.config)['input_ids'], axis=1) \
                .apply(lambda x: pd.Series(self.model(x)[0].flatten()))

        return _df
