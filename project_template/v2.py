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
import csv

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
    def __call__(self, doc,stemming=False):
        #stemming is a boolean value indicating if the results need to be stemmed
        if stemming==True:
            return [stem(t) for t in regtokenizer.tokenize(expand_contractions(doc.lower()))]
        else:
            return [t for t in regtokenizer.tokenize(expand_contractions(doc.lower()))]


#If the user has inputs a query that has more 200 words in it. Use topic modeling
print "Recovering files... "
cv=read('LDA_model','p'); print 'Vectorizer loaded'
counts =read('LDA_trainingMatrix','p');print 'trainning_matrix loaded'
counts=recover_Matrix(counts,0,counts.shape[0]); print 'training data loaded...'
res= read('LDA_fittedMatrix','p'); print 'Fitted Topic Matrix loaded...'
vocab_to_index_bi=read('vocab_to_index_bi','p'); print 'vocab_to_index for biterm loaded...'
index_to_vocab=read('index_to_vocab','json'); print 'index_to_vocab loaded...'
contractions_dict=read('contractions','json'); print 'contractions dictionary loaded...'
# Otherwise use biterm model
phiwz=read('phiwz','p');print 'word-topic distribution loaded'
theta_z=read('thetaz','p');print 'topic distribution loaded'
biterm_matrix=read('biterm_matrix1','p');print 'biterm_matrix loaded'
stop_words=read('stop_words','p');print 'stop_words loaded'

print "constructing index"
#Construct Index 
ID_to_author=defaultdict(str)
ID_to_quote=defaultdict(str)
with open('quotes.csv','rb') as f:
    reader=csv.reader(f)
    for row in reader:
        ID_to_quote[int(row[0])]=row[2]
        ID_to_author[int(row[0])]=row[1]
        
        
print "constructing tokenizer"
#Construct a Tokenizer that deals with contractions 
regtokenizer =RegexpTokenizer("[a-z]+\'?")
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)


def TMRetrieval(s,rank,similarity_measure=entropy,reverse=-1):
    query = cv.transform(s)
    unnormalized = np.matrix(model.transform(query))
    normalized=unnormalized/unnormalized.sum(axis =1)
    all_scores = []
    for i,data in enumerate(res):
        all_scores.append(similarity_measure(np.asarray(data).reshape(-1),np.asarray(normalized).reshape(-1)))

    top20=np.asarray(all_scores).argsort()[reverse*rank:]
    result = []
    for index in top20:
        result.append((ID_to_quote[index+1],ID_to_author[index+1],all_scores[index]))
    return result

#===================================================Biterm Model=====================================
def biterm_prior(biterms):
    biterm_counter=Counter(biterms)
    result = dict(zip(biterm_counter.keys(),1.0*np.array(biterm_counter.values())/sum(np.array(biterm_counter.values()))))
    return result

def get_biterms(doc,dictionary):
    #doc is a list of words
    #Return a list of tuples, contain the id of the word in the document
    result =[]
    for i,w in enumerate(doc):
        for j,w1 in enumerate(doc):
            if i<j and w in dictionary.keys() and w1 in dictionary.keys():
                result.append((dictionary[w],dictionary[w1]))
            
    return result

def topic_given_biterm(z,biterm,theta_z,pWZ):
    b0 =biterm[0]
    b1 =biterm[1]
    evidence =0
    for i in range(len(theta_z)):
        evidence+=theta_z[i]*pWZ[b0,i]*pWZ[b1,i]
    result= theta_z[z]*pWZ[b0,z]*pWZ[b1,z]/evidence*1.0
    return result


def BTMRetrieval(s,rank,similarity_measure=entropy,reverse=-1):
    bow = [t for t in regtokenizer.tokenize(expand_contractions(s.lower())) if t not in stop_words]
    print(len(bow))
    result=get_biterms(bow,vocab_to_index_bi)
    print (len(result))
    topic_doc=[]
    prior = biterm_prior(result)
    for z in range(20):
        s=0
        for word in result:
            s+=topic_given_biterm(z,word,theta_z,phiwz)*prior[word]
        topic_doc.append(s)
    all_scores = []
    for i,data in enumerate(biterm_matrix):
        if similarity_measure(np.asarray(data).reshape(-1),np.asarray(topic_doc).reshape(-1))==float('inf'):
                all_scores.append(10000) # if the matrix has no biterm in it, then set the divergence to 10000
        else:
            all_scores.append(similarity_measure(np.asarray(data).reshape(-1),np.asarray(topic_doc).reshape(-1)))
    

    top20=np.asarray(all_scores).argsort()[0:rank]
    result = []
    for index in top20:
        result.append((ID_to_quote[index+1],ID_to_author[index+1],all_scores[index]))
    return result
