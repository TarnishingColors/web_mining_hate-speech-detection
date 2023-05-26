from sklearn.pipeline import Pipeline

from notebooks.Dzim.web_mining.bert_processor import DataframePreprocessor

from notebooks.Dzim.web_mining.bert_processor import preprocessing

from notebooks.Dzim.web_mining.bert_processor import BertVectorizer

from notebooks.Dzim.web_mining.bert_processor import tokenizer

pipeline = Pipeline([
    ("preprocessing", DataframePreprocessor(preprocessing)),
    ("vectorizing", BertVectorizer(tokenizer))
])

import pandas as pd

data_sample = pd.DataFrame({
    'tweet': [
        'Sasha goes to Mannheim University',
        'Danylo will create a great application',
        'Danylo will not create a great application',
        'Dasha didnt fulfill her task'
    ],
    'class': [1, 1, 0, 0]
})

pipeline.transform(data_sample)
