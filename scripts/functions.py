import numpy as np

def preprocess(string):
    #Remove punctuation and make lower case
    string = string.lower()
    replace_with_space = str.maketrans('/\\"#$%&!()*+,-.:;<=>?@[]^_`{|}~', ' '*len('/\\"#$%&!()*+,-.:;<=>?@[]^_`{|}~'))
    string = string.translate(replace_with_space).rstrip().lstrip()
    return string

def get_average_embedding(sentence,embeddings,stopwords):
    #Returns a 1xd vector containing an average of the embeddings of all the words in a statement, stopwords not included
    words=sentence.split()
    emb_length = embeddings.shape[1]
    emb = np.array([0] * emb_length, dtype=object)
    not_in_emb = 0
    for i in range(len(words)):
        if words[i] in embeddings.index and words[i] not in stopwords:
            emb = np.vstack([emb,embeddings.loc[words[i],:]])
        else:
            not_in_emb = not_in_emb + 1
    if(len(emb.shape)>1):
        emb = emb[1:,:]
        emb = np.mean(emb,axis=0)
    return emb, not_in_emb

def get_all_averages(statements,embeddings,stopwords):
    emb_length = embeddings.shape[1]
    train_emb = np.empty((len(statements),emb_length))
    not_found=np.zeros(len(statements))
    for i, s in enumerate(list(statements)):
        train_emb[i], not_found[i] = get_average_embedding(s,embeddings,stopwords)
    return train_emb, not_found
        