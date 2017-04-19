print "importing"
import csv
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
with open('contractions.json','r') as con:
    contractions_dict=json.load(con)

print "constructing tokenizer"
#Construct a Tokenizer that deals with contractions 
regtokenizer =RegexpTokenizer("[a-z]+\'?")
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

class LemmaTokenizer(object):
    def __init__(self):
            self.wnl = stem
    def __call__(self, doc,stemming=True):
        #stemming is a boolean value indicating if the results need to be stemmed
        if stemming==True:
            return [stem(t) for t in regtokenizer.tokenize(expand_contractions(doc.lower()))]
        else:
            return [t for t in regtokenizer.tokenize(expand_contractions(doc.lower()))]

print "constructing index"
#Construct Index 
ID_to_author=defaultdict(str)
ID_to_quote=defaultdict(str)
with open('quotes.csv','rb') as f:
    reader=csv.reader(f)
    for row in reader:
        ID_to_quote[int(row[0])]=row[2]
        ID_to_author[int(row[0])]=row[1]

n_feats=5000;#Assuming the word usages are not more diverse than the movie scripts
print "building tfidf vector"
tfidf_vec = TfidfVectorizer('content',tokenizer=LemmaTokenizer(),min_df=5,max_df=0.8,max_features=n_feats,norm=None )
print "building doc_by_vocab matrix"
doc_by_vocab=tfidf_vec.fit_transform([quote for quote in ID_to_quote.values()])
# Construct a inverted map from feature index to feature value (word) for later use
index_to_vocab = {i:v for i, v in enumerate(tfidf_vec.get_feature_names())}
vocab_to_index={v:i for i,v in enumerate(tfidf_vec.get_feature_names())}
doc_by_vocab = doc_by_vocab.toarray()
valid_type=tfidf_vec.get_feature_names()

"building inverted index"
def build_inverted_index(msgs):
    index= defaultdict(list)
    for i, msg in enumerate(msgs):
        all_toks=list(set(regtokenizer.tokenize(expand_contractions(msg))))
        for tok in all_toks:
            try:
                j=vocab_to_index[tok]
                index[tok].append((i+1,doc_by_vocab[i,j]))
            except KeyError:
                continue
            except IndexError:
                print(j)
    return index
pass
inD=build_inverted_index(ID_to_quote.values())

"computing idf and doc norms"
def compute_idf(inv_idx, n_docs):
    idf={}
    all_words=inv_idx.keys()
    for word in all_words:
        idf[word]=np.log(n_docs*1.0/(1+len(inv_idx[word])))
    return idf
    pass
idf=compute_idf(inD,len(ID_to_quote.values()))

def compute_doc_norms(index,n_docs):
    norms=np.zeros(n_docs)
    all_words=index.keys()
    for word in all_words:
        doc_idx=index[word]
        for docID,tfidf in doc_idx:
            norms[docID-1]=norms[docID-1]+(tfidf)**2
    norms=np.sqrt(norms)
    return norms
    pass

docnorms= compute_doc_norms(inD,len(ID_to_quote.values()))

#Build the base system of the information retrieval using cosine similarity
def baseIR(query):
    query=query.lower()
    query_modified=[stem(t) for t in regtokenizer.tokenize(expand_contractions(query)) if stem(t) in inD.keys()]
    qCounter=Counter(query_modified)
    score=defaultdict(float)
    qtfidf=[]
    for w in query_modified:
        qtfidf.append(qCounter[w]*idf[w])
    qnorm=np.sqrt(sum(np.array(qtfidf)**2))
    results= []
    for qword in qCounter.keys():
        all_docs= inD[qword]
        for docID,tfidf in all_docs:
            score[docID-1]+=tfidf*qCounter[qword]*idf[qword]/(1.0*(docnorms[docID-1]*qnorm))
    for dID in ID_to_quote.keys():
        results.append((score[dID-1],ID_to_quote[dID],ID_to_author[dID])) 
    results.sort(key=lambda x: x[0],reverse=True)
    return results

print "done"

# from sklearn.decomposition import LatentDirichletAllocation as LDA

# cv = TfidfVectorizer(tokenizer=LemmaTokenizer(),stop_words='english', max_df=.9,min_df=10**(-6),
#                      max_features=5000)
# counts = cv.fit_transform([quote for quote in ID_to_quote.values()])
# feature_names = cv.get_feature_names()

# n_topic = 20
# model = LDA(n_topics=n_topic, max_iter=10, n_jobs=1, verbose=0)
# res = model.fit_transform(counts)

# def categorize_top_words(model, feature_names, n_top_words):
#     result=[]
#     for topic_idx, topic in enumerate(model.components_):
#         st=" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
#         result.append(st)
#     return result

# def TMRetrieval(s,rank,similarity_measure=entropy,reverse=-1):
#     query = tfidf_vec.transform(s)
#     unnormalized = np.matrix(model.transform(query))
#     normalized=unnormalized/unnormalized.sum(axis =1)

#     all_scores = []
#     for i,data in enumerate(res):
#         all_scores.append(similarity_measure(np.asarray(data).reshape(-1),np.asarray(normalized).reshape(-1)))
    

#     top20=np.asarray(all_scores).argsort()[reverse*rank:]
#     print("\n")
#     for index in top20:
#         print (ID_to_quote[index+1],ID_to_author[index+1],'similarity: ',all_scores[index])
#         print ('\n')
#     return top20

