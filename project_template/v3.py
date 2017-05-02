
from collections import defaultdict
from collections import Counter
import json
import numpy as np
from scipy.stats import entropy    
from nltk.stem import WordNetLemmatizer as wnl
from nltk.tokenize import RegexpTokenizer
from stemming.porter2 import stem
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()  
# import en_core_web_sm as en_core
from scipy.sparse.linalg import svds
# nlp=en_core.load()
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
## All datasets and dictionary
contractions_dict=read('contractions','json'); print 'contractions dictionary loaded...'
stop_words=read('stop_words','p');print 'stop_words loaded'
word_to_attitude=read('word_to_attitude','json'); print 'word_attitude loaded'
author_feature_words=read('author_feature_words','json'); print 'feature words for authors loaded'
topic_encoder=read('topic_encoder','p');print 'topic decoded loaded'
author_to_index=read('author_to_index','json');print 'author_to_index loaded'
index_to_author=read('index_to_author','json');print 'index_to_author loaded'
# topic_prediction=read('topic_prediction_vectorizer','p');print 'topic prediction vectorizer loaded'
# topic_prediction_model=read('topic_prediction_model','p');print 'topic prediction model loaded'
author_matrix_compressed = read('author_matrix_compressed','p'); print 'author matrix (after SVD) loaded'
author_matrix = read('author_matrix','p'); print 'author matrix (before SVD) loaded'
author_prediction_vectorizer=read('author_prediction_vectorizer','p');print 'author prediction vectorizer loaded'
topic_list = read('all_topics_prediction','p'); print 'all topic loaded'
ID_to_quote=read("ID_to_quote",'p');print 'ID_to_quote loaded'
ID_to_author=read("ID_to_author",'p');print 'ID_to_author loaded'
topic_predictor=read("topic_predictor",'p'); print 'topic_predictor loaded'


print "constructing tokenizer"
#Construct a Tokenizer that deals with contractions 
regtokenizer =RegexpTokenizer("[a-z]+\'?")
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))



def importOption(InputString):
    lenquery= len([t for t in regtokenizer.tokenize(expand_contractions(InputString.lower()))])
    if lenquery<=200:
        ## Apply biterm model if the length of the query is less than 200
        # Otherwise use biterm model
        vocab_to_index_bi=read('vocab_to_index_1','json'); print 'vocab_to_index for biterm loaded...'
        index_to_vocab=read('index_to_vocab','json'); print 'index_to_vocab loaded...'
        phiwz=read('phiwz','p');print 'word-topic distribution loaded'
        theta_z=read('thetaz','p');print 'topic distribution loaded'        
        biterm_matrix=read('biterm_matrix_full','p');print 'biterm_matrix loaded'
    else:## Topic Model:
        cv=read('LDA_model','p'); print 'Vectorizer loaded'
        counts =read('LDA_trainingMatrix','p');print 'trainning_matrix loaded'
        counts=recover_Matrix(counts,0,counts.shape[0]); print 'training data loaded...'
        res= read('LDA_fittedMatrix','p'); print 'Fitted Topic Matrix loaded...'

def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)


def TMRetrieval(s,rank,similarity_measure=entropy,reverse=-1):
    query_tokens = cv.transform(s)
    unnormalized = np.matrix(model.transform(query_tokens))
    normalized=unnormalized/unnormalized.sum(axis =1)
    all_scores = []
    for i,data in enumerate(res):
        all_scores.append(similarity_measure(np.asarray(data).reshape(-1),np.asarray(normalized).reshape(-1)))

    top20=np.asarray(all_scores).argsort()[reverse*rank:]
    result = []
    for index in top20:
        result.append((ID_to_quote[index],ID_to_author[index+1],all_scores[index]))
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


def BTMRetrieval(s,rank,filter_by=False,matrix=biterm_matrix,similarity_measure=entropy,reverse=-1):
    if filter_by !=False:
        indexes= np.where(topic_lists==filter_by)[0]
        matrix = np.matrix(matrix)[np.array(indexes),:]
    query_tokens = [t for t in regtokenizer.tokenize(expand_contractions(s.lower())) if t not in stop_words]
    result=get_biterms(query_tokens,vocab_to_index_bi)
    topic_doc=[]
    prior = biterm_prior(result)
    for z in range(20):
        s=0
        for word in result:
            s+=topic_given_biterm(z,word,theta_z,phiwz)*prior[word]
        topic_doc.append(s)
    all_scores = []
    for i,data in enumerate(matrix):
        if similarity_measure(np.asarray(data).reshape(-1),np.asarray(topic_doc).reshape(-1))==float('inf'):
                all_scores.append(10000) # if the matrix has no biterm in it, then set the divergence to 10000
        else:
            all_scores.append(similarity_measure(np.asarray(data).reshape(-1),np.asarray(topic_doc).reshape(-1)))
    

    top20=np.asarray(all_scores).argsort()[0:rank]
    result = []
    if filter_by ==False:
        for index in top20:
            result.append((ID_to_quote[index],ID_to_author[index],index))
            print "{}: {}\n\n".format(ID_to_quote[index], all_scores[index])
        return result
    else:
        for index in top20:
            result.append((ID_to_quote[indexes[index]],ID_to_author[indexes[index]],indexes[index]))
            print "{}: {}\n\n".format(ID_to_quote[indexes[index]], all_scores[index])
        return result

##=====================================================Rocchio Update==========================================================
##Use Rocchio updating to update user query
#def irrelevant(docs, all_docs):
#  return set(all_docs)-set(docs)
  
def Rocchio_updating(docs,query,all_docs,matrix,alpha=1, beta=0.8,theta=0.1):
    # docs as list of IDs and query is the original query (in the form of a vector)
    # Now we treat each doc in docs as 'relevant' and all_docs-docs as irrelevant
    #other_docs=list(irrelevant(docs, all_docs))
    
    #denote tuning parameters as \alpha, \beta and \theta
    query_modified = alpha*query+beta*matrix[docs-1,:].sum(axis=0)/len(docs)#-theta*matrix[other_docs-1,:].sum(axis=0)/len(other_docs)
    return query_modified
  
##===================================================Predict Author============================================================

##Recommended authors, based on the query and the quote that the user clicks on
def relevant_author (query,ID,matrix=author_matrix,vectorizer=author_prediction_vectorizer,numReturn=5,similarity_measure=entropy):
    longstring = "{} {}".format(query, ID_to_quote[ID])
    print longstring
    new_vector = vectorizer.transform([longstring]).toarray()[0, 169:]
    print "created vector"
    print np.shape(new_vector)
    all_scores = []
    print "calculating scores"
    for row in recover_Matrix(matrix, 0, matrix.shape[0]):
        print type(np.asarray(row).reshape(-1))
        print type(np.asarray(new_vector[0]))
        all_scores.append(similarity_measure(np.asarray(row).reshape(-1),np.asarray(new_vector[0])))
        print "appended"
    
    results=np.asarray(all_scores).argsort()[0:numReturn]
    return results.tolist()

## Similar authors, based entirely on the authors' textual features
def similar_author (ID,matrix=author_matrix_compressed,numReturn=5,similarity_measure=entropy):
    all_scores = []
    author=author_to_index[ID_to_author[ID]]
    for row in matrix:
        all_scores.append(similarity_measure(np.asarray(row).reshape(-1),np.asarray(matrix[author,:])))
    print all_scores[:5]
    results=np.asarray(all_scores).argsort()[1:numReturn+1]
    print results

    for i in range(numReturn):
        results[i] = ID_to_author[results[i]]

    return results

def show_feature_words(author):
    return author_feature_words[author]
##=================================================Predict Topic ==========================================
def decode_topic(topicID):
    return topic_encoder.inverse_transform(topicID)

def related_topics(query, quoteID, topic_predictor ):
    new_vec = tfidf_vec.transform([query + ID_to_quote[quoteID]])
    topics = topic_encoder.inverse_transform(np.argsort(ovr.decision_function(new_vec).ravel())[::-1][:8]).tolist()
    return topics
##=================================================Other Utility Functions=====================================================
## Unstem words,may be used to decode key features
def unstem(word):
    return unstem[words]
## Indicate whether a word is a name entity word or not
# def isEntity(word):
#     entities=nlp(word.title())
#     if entities.ents==():
#         return False 
#     else:
#         return True
#Takes in the query and quote, compare the sentiments of the query
def sentimental_analysis(string):
    #Using Vader's sentiments to display the intensity score 
    intensity_score=analyzer.polarity_scores(string)
    #Use the ANEW system to determine various sentimental domain
    words = [stem(t) for t in regtokenizer.tokenize(expand_contractions(string.lower()))]
    anew_score = np.zeros((1,3))
    for word in words:
        if stem(word) in word_to_attitude.keys():
            anew_score+=np.array(word_to_attitude[stem(word)])
    return (intensity_score,anew_score)

