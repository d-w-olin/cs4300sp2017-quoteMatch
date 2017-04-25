from collections import defaultdict
from collections import Counter
import json
import numpy as np
from scipy.stats import entropy    
from nltk.stem import WordNetLemmatizer as wnl
from nltk.tokenize import RegexpTokenizer
from stemming.porter2 import stem
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle
import scipy.sparse as sps


def read(filename,f):
    with open(filename+'.'+f,'rb') as files:
        if f=='p':
            return pickle.load(files)
        elif f=='json':
            return json.load(files)
        else:
            print('check format please.')
def recover_Matrix(Sparse,m,n):
    ip = Sparse.indptr[m:n].copy()
    d = Sparse.data[ip[0]:ip[-1]]
    i = Sparse.indices[ip[0]:ip[-1]]
    ip -= ip[0]
    rows = sps.csr_matrix((d, i, ip))
    return rows

class LemmaTokenizer(object):
    def __init__(self):
            self.wnl = stem
    def __call__(self, doc,stemming=True):
        #stemming is a boolean value indicating if the results need to be stemmed
        if stemming==True:
            return [stem(t) for t in regtokenizer.tokenize(expand_contractions(doc.lower()))]
        else:
            return [t for t in regtokenizer.tokenize(expand_contractions(doc.lower()))]


print "Recovering files... "
doc_by_vocab=read('doc_by_vocab','p') 
doc_by_vocab=recover_Matrix(doc_by_vocab,0,doc_by_vocab.shape[0]); print'document-term matrix loaded...'
counts =read('LDA_trainingMatrix','p');print 'trainning_matrix loaded'
counts=recover_Matrix(counts,0,counts.shape[0]); print 'training data loaded...'
res= read('LDA_fittedMatrix','p'); print 'Fitted Topic Matrix loaded...'
vocab_to_index=read('vocab_to_index','json'); print 'vocab_to_index loaded...'
index_to_vocab=read('index_to_vocab','json'); print 'index_to_vocab loaded...'
contractions_dict=read('contractions','json'); print 'contractions dictionary loaded...'

print "constructing tokenizer"
#Construct a Tokenizer that deals with contractions 
regtokenizer =RegexpTokenizer("[a-z]+\'?")
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)
