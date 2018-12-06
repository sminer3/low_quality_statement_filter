# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
import numpy as np


#Notes for training
def nn(x,y,cv):
    kf=model_selection.KFold(cv,shuffle=True)
    auc = []
    probs = np.zeros(len(x))[:,None]
    i = 1
    #Get rid of for loop
    for itrain, iother in kf.split(x):
        print('Model ',i)
        i = i+1
        #Split .85 .15
        ival, itest = model_selection.train_test_split(iother,test_size=.5)
        x_train = x[itrain]
        y_train = y[itrain]
        
        x_val = x[ival]
        y_val = y[ival]
        
        #Get rid of test
        x_test = x[itest]
        y_test = y[itest]
        
        inputs = Input(shape=(x_train.shape[1],), dtype="float32")
        layer_one = Dense(150,activation='relu')(inputs)
        layer_one = Dropout(0.5)(layer_one)
        layer_two = Dense(75,activation='relu')(layer_one)
        layer_two = Dropout(0.5)(layer_two)
        out = Dense(1,activation='sigmoid')(layer_two)
        
        model = Model(inputs=inputs, outputs=out)
        model.compile(loss="binary_crossentropy", optimizer='adam')
        early_stopping = EarlyStopping(monitor="val_loss", patience=5)
        model_checkpoint = ModelCheckpoint("NN_test_model.h5",save_best_only=True,save_weights_only=True)
        model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=20, batch_size=32, shuffle=True,callbacks=[early_stopping, model_checkpoint], verbose=2)
        
        # Save the weights
        model.load_weights("NN_test_model.h5")
        
        #Change this validation data for testing auc
        preds = model.predict(x_test)
        test_auc = roc_auc_score(y_test,preds)
        auc.append(test_auc)
        probs[itest] = preds
        probs[ival] = model.predict(x_val)
        #Save validation with predictions
    return auc, probs

