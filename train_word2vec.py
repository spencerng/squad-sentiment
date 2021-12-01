#!/bin/env python

import sys
import json
import nltk
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

with open("squad-data/train-v1.1.json") as open_file:
    data_file = json.loads(open_file.read())

questions = [y['question'] for d in data_file['data'] for x in d['paragraphs'] for y in x['qas']]
paragraphs = [x['context'] for d in data_file['data'] for x in d['paragraphs']]

tokenized_questions = [nltk.word_tokenize(x) for x in questions]
tokenized_paragraphs = [nltk.word_tokenize(x) for x in paragraphs]

model = Word2Vec(tokenized_paragraphs + tokenized_questions, min_count=0, workers=8, window =4, sg = 1, epochs = 20)
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
sims = model.wv.most_similar('founded', topn=10) 
print(sims)
