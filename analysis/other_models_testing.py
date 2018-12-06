# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 12:52:51 2018

@author: sminer
"""
print("Loading libraries and creating features")
import pandas as pd
import numpy as np

from scripts.functions import preprocess, get_all_averages
from sklearn import metrics, model_selection, linear_model, ensemble, neighbors, svm, naive_bayes


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

#Load embeddings
emb = pd.read_csv("../data/google_word2vec.csv",index_col=0)

print("Getting sentence embeddings (by averaging the word embeddings in each sentence)")
sent_emb, not_found = get_all_averages(train.statementText,emb,'')
perc_not_found = not_found / np.array(train["nword"])
print("Subsetting created features and training models")

# little hack to filter out Proba(y==1)
def roc_auc_score_proba(y_true, proba):
    return metrics.roc_auc_score(y_true, proba[:, 1])

# define your scorer
auc = metrics.make_scorer(roc_auc_score_proba, needs_proba=True)

x = np.hstack((np.array(train.loc[:,['nchar','idk','comment','skip','unsure','survey','answer','test','question_word','yes','none','sure']]),sent_emb))
y = np.reshape(np.array(train.loc[:,['exclude']]),len(train)).astype('int')

print("Training Logistic Regression Classifier")
log_reg = linear_model.LogisticRegression()
scores_log = model_selection.cross_val_score(log_reg,x,y,cv=5,scoring=auc)
print("AUC log_regression: %0.2f (+/- %0.2f)" % (scores_log.mean(), scores_log.std() * 2))

print("Training random Forest Classifier")
rf = ensemble.RandomForestClassifier()
scores_rf = model_selection.cross_val_score(rf,x,y,cv=5,scoring=auc)
print("AUC rf: %0.2f (+/- %0.2f)" % (scores_rf.mean(), scores_rf.std() * 2))

print("Training K-Nearest Neighboars Classifier")
knn = neighbors.KNeighborsClassifier()
scores_knn = model_selection.cross_val_score(knn,x,y,cv=5,scoring=auc)
print("AUC knn: %0.2f (+/- %0.2f)" % (scores_knn.mean(), scores_knn.std() * 2))

print("Training Bagging Classifier (using K-nearest neighbors)")
bagging = ensemble.BaggingClassifier(neighbors.KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
scores_bagging = model_selection.cross_val_score(bagging,x,y,cv=5,scoring=auc)
print("AUC bagging: %0.2f (+/- %0.2f)" % (scores_bagging.mean(), scores_bagging.std() * 2))

print("Training Bagging Classifier (using Logistic regression)")
log_bagging = ensemble.BaggingClassifier(linear_model.LogisticRegression(),max_samples=0.5, max_features=0.5)
scores_log_bagging = model_selection.cross_val_score(log_bagging,x,y,cv=5,scoring=auc)
print("AUC log_bagging: %0.2f (+/- %0.2f)" % (scores_log_bagging.mean(), scores_log_bagging.std() * 2))

print("Training Extra Trees Classifier")
et = ensemble.ExtraTreesClassifier()
scores_et = model_selection.cross_val_score(et,x,y,cv=5,scoring=auc)
print("AUC et: %0.2f (+/- %0.2f)" % (scores_et.mean(), scores_et.std() * 2))

print("Training Adaptive Boosting Classifier")
ada_boost = ensemble.AdaBoostClassifier()
scores_ada_boost = model_selection.cross_val_score(ada_boost,x,y,cv=5,scoring=auc)
print("AUC ada_boost: %0.2f (+/- %0.2f)" % (scores_ada_boost.mean(), scores_ada_boost.std() * 2))

print("Training Gradient Boosting Classifier")
gb = ensemble.GradientBoostingClassifier()
scores_gb = model_selection.cross_val_score(gb,x,y,cv=5,scoring=auc)
print("AUC gb: %0.2f (+/- %0.2f)" % (scores_gb.mean(), scores_gb.std() * 2))

print("Training Ensemble (using logistic regression, knn, extra trees, adaBoost, gradBoost)")
eclf = ensemble.VotingClassifier(estimators=[('lr', log_reg), ('rf', rf), ('knn', knn), ('et',et),('ada_boost',ada_boost),('gb',gb)], voting='soft',weights=[5,1,1,1,1,1])
scores_eclf = model_selection.cross_val_score(eclf,x,y,cv=5,scoring=auc)
print("AUC eclf: %0.2f (+/- %0.2f)" % (scores_eclf.mean(), scores_eclf.std() * 2))

print("Training Naive Bayes Classifier")
nb = naive_bayes.GaussianNB()
scores_nb = model_selection.cross_val_score(nb,x,y,cv=5,scoring=auc)
print("AUC nb: %0.2f (+/- %0.2f)" % (scores_nb.mean(), scores_nb.std() * 2))

# print("Training support vector machine classifier")
# svc = svm.SVC(probability = True)
# scores_svc = model_selection.cross_val_score(svc,x,y,cv=5,scoring=auc)
# print("AUC svc: %0.2f (+/- %0.2f)" % (scores_svc.mean(), scores_svc.std() * 2))



