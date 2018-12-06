print("Loading Libraries and data")
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

from scripts.functions import preprocess, get_all_averages

def nn_train(x,y):
    
	itrain, ival = model_selection.train_test_split(list(range(len(x))),test_size=.15)
	x_train = x[itrain]
	y_train = y[itrain]
	
	x_val = x[ival]
	y_val = y[ival]
	
	inputs = Input(shape=(x_train.shape[1],), dtype="float32")
	layer_one = Dense(150,activation='relu')(inputs)
	layer_one = Dropout(0.5)(layer_one)
	layer_two = Dense(75,activation='relu')(layer_one)
	layer_two = Dropout(0.5)(layer_two)
	out = Dense(1,activation='sigmoid')(layer_two)
	
	model = Model(inputs=inputs, outputs=out)
	model.compile(loss="binary_crossentropy", optimizer='adam')
	early_stopping = EarlyStopping(monitor="val_loss", patience=5)
	model_checkpoint = ModelCheckpoint("zero_value_model.h5",save_best_only=True,save_weights_only=False)
	model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=20, batch_size=32, shuffle=True,callbacks=[early_stopping, model_checkpoint], verbose=2)

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

#Neural network
nn_train(x,y)
print("Model Saved")
