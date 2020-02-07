#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pickle
import json

from chemdataextractor.nlp.tokenize import ChemWordTokenizer
cwt = ChemWordTokenizer()

from gensim.models import KeyedVectors
from gensim.models import Word2Vec

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import json
import logging


logging.basicConfig(format="%(levelname)s: %(funcName)s, %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

with open("/home/nlp/NN_word_classification/cem_classcification_model", "rb") as f:
    mlp = pickle.load(f)
with open("/home/nlp/NN_word_classification/classification_dic.json", "rb") as f:
    classification_dic = json.load(f)
    
wv = Word2Vec.load('/home/nlp/word2vec_training/materials-word-embeddings/bin/word2vec_embeddings-SNAPSHOT.model').wv


# In[14]:


def get_word_vector(token, wv=wv):
    vector = np.zeros((100,))
    
    if token in wv:
        vector = wv[token]
    else:
#         logging.info("token: {}".format(cwt.tokenize(token)))
        for word in cwt.tokenize(token):
            try:
                vector += wv[word]
            except:
                pass
    return vector

labels = {0: "metal-source",1: "linker",2: "solvent",3: "modulator",4: "temp",5: "time",6: "name",7: "others"}

def cem_classifier(cem):
#     logging.info("cem: {}".format(cem))
    label = labels[mlp.predict(get_word_vector(cem).reshape(1, -1))[0]]
    for label_in_dic in classification_dic.keys():
        if cem in classification_dic[label_in_dic].keys():
            label = label_in_dic
    return label


# In[15]:


# cem_classifier("ZrCl4")


# In[ ]:




