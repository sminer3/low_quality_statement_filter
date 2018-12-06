# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 13:49:20 2018

@author: sminer
"""
import pandas as pd
import numpy as np
import gensim
import csv
from scripts.functions import preprocess



#Need to change with where you have google news vectors stored locally
filepath = "C:/Users/sminer/Downloads/GoogleNews-vectors-negative300.bin.gz"

print('loading model')
model = gensim.models.KeyedVectors.load_word2vec_format(filepath,binary=True)
print('model loaded')

train = pd.read_csv("cst_reviewed_statements.csv")
train['statementText'] = train['statementText'].apply(str).apply(preprocess)
train['question'] = train['question'].apply(str).apply(preprocess)
words = set(train['statementText'].str.split(' ', expand=True).stack().unique())
words_q = set(train['question'].str.split(' ', expand=True).stack().unique())
words = words.union(words_q)

out = {}
found = 0
for word in words:
    if word in model.vocab:
        out[word] = model[word]
        found = found+1


with open('google_word2vec.csv','w') as f:
    fieldnames = ['word']
    fieldnames.extend(list(range(0,300)))
    writer = csv.DictWriter(f,fieldnames=fieldnames,lineterminator='\n')
    writer.writeheader()
    data = [dict(zip(fieldnames,[w]+list(v))) for w,v in out.items()]
    writer.writerows(data)
print('Word embeddings successfully saved')