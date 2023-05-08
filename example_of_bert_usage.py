from sklearn.pipeline import Pipeline

from notebooks.Dzim.web_mining.bert_processor import DataframePreprocessor

from notebooks.Dzim.web_mining.bert_processor import preprocessing

from notebooks.Dzim.web_mining.bert_processor import BertVectorizer

from notebooks.Dzim.web_mining.bert_processor import tokenizer

pipeline = Pipeline([
    ("preprocessing", DataframePreprocessor(preprocessing)),
    ("vectorizing", BertVectorizer(tokenizer))
])