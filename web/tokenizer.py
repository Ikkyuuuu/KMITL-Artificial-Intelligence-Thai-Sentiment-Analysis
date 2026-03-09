import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords

stopwords = set(thai_stopwords())
stopwords.discard('ไม่')
stopwords.discard('ดี')

def smart_tokenizer(text):
    tokens = word_tokenize(text, engine="newmm")
    result = []
    skip_next = False
    for i in range(len(tokens)):
        if skip_next:
            skip_next = False
            continue
        if tokens[i] == 'ไม่' and i + 1 < len(tokens):
            result.append('ไม่_' + tokens[i+1])
            skip_next = True
        else:
            if tokens[i] not in stopwords and tokens[i].strip():
                result.append(tokens[i])
    return result