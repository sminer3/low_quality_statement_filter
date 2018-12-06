# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 12:52:51 2018

@author: sminer
"""
print("Loading libraries and creating features")
import pandas as pd
import numpy as np

from scripts.functions import preprocess, get_all_averages
from scripts.nn_cv import nn


#Subset training data
train = pd.read_csv("../data/cst_reviewed_statements.csv")
train = train[pd.isnull(train.exclude)==False]
train = train[pd.isnull(train.statementText)==False]
train = train.fillna(0)
train = train[np.logical_not(train['statementText']=='')]
train = train[(train.gibberish==1)==False]
train = train[(train.offensive==1)==False]
train = train[(train.profanity==1)==False]

#Feature creation
train["nchar"] = train.statementText.str.len()
train["nchar_q"] = train.question.str.len()
train["nword"] = [len(str(x).split()) for x in train.statementText]
train["nword_q"] = [len(str(x).split()) for x in train.question]
train["statementText"] = train.loc[:,"statementText"].fillna("").apply(preprocess)
train["question"] = train.loc[:,"question"].fillna("").apply(preprocess)
train["nchar_no_punc"] = train.statementText.str.len()
train["nchar_no_punc_q"] = train.question.str.len()

#Check for inclusion of words (words that seemed indicative of zero value)
train["idk"]= ["idk" in str(x).split() for x in train.statementText]
train["comment"]= ["comment" in str(x).split() for x in train.statementText]
train["skip"] = ["skip" in str(x).split() for x in train.statementText]
train['unsure'] = ['unsure' in str(x).split() for x in train.statementText]
train["survey"]= ["survey" in str(x).split() for x in train.statementText]
train["answer"]= ["answer" in str(x).split() for x in train.statementText]
train["test"]= ["test" in str(x).split() for x in train.statementText]
train["question_word"]= ["question" in str(x).split() for x in train.statementText]
train["yes"]= ["yes" in str(x).split() for x in train.statementText]
train["none"]= ["none" in str(x).split() for x in train.statementText]
train["sure"]= ["sure" in str(x).split() for x in train.statementText]
train["else"]= ["else" in str(x).split() for x in train.statementText]
train["nothing"]= ["nothing" in str(x).split() for x in train.statementText]    
train["cant"]= ["cant" in str(x).split() for x in train.statementText]
train["know"]= ["know" in str(x).split() for x in train.statementText]
train["thats"]= ["thats" in str(x).split() for x in train.statementText]
train["already"]= ["already" in str(x).split() for x in train.statementText]
train["cool"]= ["cool" in str(x).split() for x in train.statementText]
train["idea"]= ["idea" in str(x).split() for x in train.statementText]
train["same"]= ["same" in str(x).split() for x in train.statementText]
train["dont"]= ["dont" in str(x).split() for x in train.statementText]
train["no"]= ["no" in str(x).split() for x in train.statementText]

emb = pd.read_csv("../data/google_word2vec.csv",index_col=0)

print("Getting sentence embeddings (by averaging the word embeddings in each sentence)")
sent_emb, not_found = get_all_averages(train.statementText,emb,'')
perc_not_found = not_found / np.array(train["nword"])
print("Subsetting created features and training model")

from sklearn import metrics 

# little hack to filter out Proba(y==1)
def roc_auc_score_proba(y_true, proba):
    return metrics.roc_auc_score(y_true, proba[:, 1])

# define your scorer
auc = metrics.make_scorer(roc_auc_score_proba, needs_proba=True)


x = np.hstack((np.array(train.loc[:,['nchar','idk','comment','skip','unsure','survey','answer','test','question_word','yes','none','sure']]),sent_emb))
y = np.reshape(np.array(train.loc[:,['exclude']]),len(train)).astype('int')

#Neural network training using cross validation
scores_nn, preds = nn(x,y[:,None],cv=5)
print("AUC nn: %0.2f (+/- %0.2f)" % (np.mean(scores_nn), np.std(scores_nn) * 2))
train['preds'] = preds
train.to_csv("../data/cst_reviewed_statements_preds.csv",index=False)

