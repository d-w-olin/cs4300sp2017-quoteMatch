
from collections import defaultdict
from collections import Counter
import json
import numpy as np
from scipy.stats import entropy    
from nltk.stem import WordNetLemmatizer as wnl
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import brown
from nltk.corpus import wordnet as wn
import operator
from stemming.porter2 import stem
from sklearn.feature_extraction.text import TfidfVectorizer
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# analyzer = SentimentIntensityAnalyzer()  
#from operator import itemgetter
# import en_core_web_sm as en_core
from scipy.sparse.linalg import svds
# nlp=en_core.load()
import re
import pickle
import scipy.sparse as sps
import csv

#nltk.data.path.append('../nltk_data/')

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
#word_to_attitude=read('word_to_attitude','json'); print 'word_attitude loaded'
author_feature_words=read('author_feature_words','json'); print 'feature words for authors loaded'
topic_encoder=read('topic_encoder','p');print 'topic decoded loaded'
author_to_index=read('author_to_index','json');print 'author_to_index loaded'
index_to_author=read('index_to_author','json');print 'index_to_author loaded'
topic_prediction_vectorizer=read('topic_prediction_vectorizer','p');print 'topic prediction vectorizer loaded'
# topic_prediction_model=read('topic_prediction_model','p');print 'topic prediction model loaded'
# author_matrix_compressed = read('author_matrix_compressed','p'); print 'author matrix (after SVD) loaded'
author_matrix = read('author_matrix','p'); print 'author matrix (before SVD) loaded'
author_prediction_vectorizer=read('author_prediction_vectorizer','p');print 'author prediction vectorizer loaded'
#topic_list = read('all_topics_prediction','p'); print 'all topic loaded'
ID_to_quote=read("ID_to_quote",'p');print 'ID_to_quote loaded'
ID_to_author=read("ID_to_author",'p');print 'ID_to_author loaded'
topic_predictor=read("topic_predictor",'p'); print 'topic_predictor loaded'
vocab_to_index = read('vocab_to_index','p');print 'vocab_to_index loaded'
#index_to_vocab=read('index_to_vocab','json'); print 'index_to_vocab loaded...'
phiwz=read('phiw_zz','p');print 'word-topic distribution loaded'
theta_z=read('theta_zz','p');print 'topic distribution loaded'        
biterm_matrix=read('bmatrix','p');print 'biterm_matrix loaded'
primary_indexes=read('primary_topics','p');print 'primary topics loaded'
secondary_indexes=read('secondary_topics','p');print 'secondary topics loaded'
freqs=read('freqs','p');print 'Brown word frequencies loaded'

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


#def TMRetrieval(s,rank,similarity_measure=entropy,reverse=-1):
#    query_tokens = cv.transform(s)
#    unnormalized = np.matrix(model.transform(query_tokens))
#    normalized=unnormalized/unnormalized.sum(axis =1)
#    all_scores = []
#    for i,data in enumerate(res):
#        all_scores.append(similarity_measure(np.asarray(data).reshape(-1),np.asarray(normalized).reshape(-1)))

#    top20=np.asarray(all_scores).argsort()[reverse*rank:]
#    result = []
#    for index in top20:
#        result.append((ID_to_quote[index],ID_to_author[index+1],all_scores[index]))
#    return result

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

negative=np.ones(len(theta_z))
for i,theta in enumerate(theta_z):
    if theta<0:
        theta_z[i]=-1*theta_z[i]
        negative[i]=-1

Evidence=np.sqrt(np.array(theta_z))*phiwz

def topic_given_biterm(biterm,theta_z,pWZ):
    b0 =biterm[0]
    b1 =biterm[1]
    evidence=np.sum(Evidence[b0-168,:]*Evidence[b1-168,:])
    result= Evidence[b0-168,:]*Evidence[b1-168,:]/evidence*1.0
    return result

#get the most common synonym for a given word
def get_best_synonym(word):
    syns = []
    syn_whl_words = []
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            if l.name() != word:
                syns.append((l.name(),l.count()))
    #print "Syns" , len(syns)
    for word in syns:
        #print word
        if '_' not in word[0]:
            #print word[0]
            syn_whl_words.append(word)
    syn_whl_words.sort(key=operator.itemgetter(1),reverse=True)
    if len(syn_whl_words) > 0:
        return  syn_whl_words[0][0]
    else:
        return None
        
#add synonyms to the query
def augment_query(token_list):
    new_query = []
    for token in token_list:
        new_query.append(token)
        if freqs[token] < 1000:
            new_query.append(get_best_synonym(token))
    return new_query

def BTMRetrieval(s,rank,filter_by=False,matrix=biterm_matrix,similarity_measure=entropy,reverse=-1):
    if filter_by !=False:
        all_indices =[]
        for f in filter_by:
            main_indexes=primary_indexes[f]
            second_indexes=secondary_indexes[f]
            all_indices.extend(main_indexes)
            all_indices.extend(secondary_indexes)
        all_indices = list((set(all_indices)))
        matrix = np.matrix(matrix)[np.array(all_indices),:]
    bow = [t for t in regtokenizer.tokenize(expand_contractions(s.lower())) if t not in stop_words]
    #removing synonyms at line below because they don't seem to add much, and we need more speed
    bow = augment_query(bow)
    result=get_biterms(bow,vocab_to_index)
    topic_doc=np.zeros((1,len(theta_z)))
    prior = biterm_prior(result)
    for word in result:
            topic_doc+=np.array(topic_given_biterm(word,theta_z,phiwz)*prior[word]) 
    all_scores = []
    for i,data in enumerate(matrix):
            all_scores.append(similarity_measure(np.asarray(data).reshape(-1)+10**(-8),np.asarray(topic_doc).reshape(-1))+10**(-8))
    top20=np.asarray(all_scores).argsort()[0:rank]
    result = []
    if filter_by ==False:
        print("\n")
        for index in top20:
            result.append((ID_to_quote[index],ID_to_author[index],index))
        return result
    else:
        print("\n")
        for index in top20:
            result.append((ID_to_quote[all_indices[index]],ID_to_author[all_indices[index]],all_indices[index]))
        return result
##=====================================================Rocchio Update==========================================================
##Use Rocchio updating to update user query
#def irrelevant(docs, all_docs):
#  return set(all_docs)-set(docs)
  
#def Rocchio_updating(docs,query,all_docs,matrix,alpha=1, beta=0.8,theta=0.1):
    # docs as list of IDs and query is the original query (in the form of a vector)
    # Now we treat each doc in docs as 'relevant' and all_docs-docs as irrelevant
    #other_docs=list(irrelevant(docs, all_docs))
    
    #denote tuning parameters as \alpha, \beta and \theta
#    query_modified = alpha*query+beta*matrix[docs-1,:].sum(axis=0)/len(docs)#-theta*matrix[other_docs-1,:].sum(axis=0)/len(other_docs)
#    return query_modified

#def RocchioRetrieval(rocchio_query,rank,filter_by=False,matrix=biterm_matrix,similarity_measure=entropy,reverse=-1):
#    if filter_by !=False:
#       all_indices =[]
#        for f in filter_by:
#            main_indexes=primary_indexes[f]
#            second_indexes=secondary_indexes[f]
#            all_indices.extend(main_indexes)
#            all_indices.extend(secondary_indexes)
#        all_indices = list((set(all_indices)))
#        matrix = np.matrix(matrix)[np.array(all_indices),:]
#    all_scores = []
#    for i,data in enumerate(matrix):
#            all_scores.append(similarity_measure(np.asarray(data).reshape(-1)+10**(-8),rocchio_query+10**(-8)))
#    top20=np.asarray(all_scores).argsort()[0:rank]
#    result = []
#    if filter_by ==False:
#        for index in top20:
#            result.append((ID_to_quote[index],ID_to_author[index],index))
#        return result
#    else:
#        for index in top20:
#            result.append((ID_to_quote[all_indices[index]],ID_to_author[all_indices[index]],all_indices[index]))
#        return result 
##===================================================Predict Author============================================================

##Recommended authors, based on the query and the quote that the user clicks on
def relevant_author (query,ID,matrix=author_matrix,vectorizer=author_prediction_vectorizer,numReturn=5,similarity_measure=entropy):
    longstring = query+' '+ID_to_quote[ID]
    new_vector = vectorizer.transform([longstring]).toarray()[0,167:]
    all_scores = []
    matrix = matrix.todense()
    for row in matrix:
            all_scores.append(similarity_measure(np.asarray(row).reshape(-1)+10**-6,np.asarray(new_vector)+10**-6))
    results=np.asarray(all_scores).argsort()[0:numReturn]
    res = []
    for i in range(len(results)):
        res.append(ID_to_author[results[i]])
    return res

## Similar authors, based entirely on the authors' textual features
#def similar_author (ID,matrix=author_matrix_compressed,numReturn=5,similarity_measure=entropy):
#    all_scores = []
#    author=author_to_index[ID_to_author[ID]]
#    for row in matrix:
#        all_scores.append(similarity_measure(np.asarray(row).reshape(-1),np.asarray(matrix[author,:])))
#    print all_scores[:5]
#   results=np.asarray(all_scores).argsort()[1:numReturn+1]
#    print results
#
#    for i in range(numReturn):
#        results[i] = ID_to_author[results[i]]
#
#    return results

def show_feature_words(author):
    return author_feature_words[author]
##=================================================Predict Topic ==========================================
def decode_topic(topicID):
    return topic_encoder.inverse_transform(topicID)

def related_topics(query, quoteID, vectorizer=topic_prediction_vectorizer, topic_predictor=topic_predictor):
    new_vec = vectorizer.transform([query + ' ' + ID_to_quote[quoteID]])
    topics = topic_encoder.inverse_transform(np.argsort(topic_predictor.decision_function(new_vec).ravel())[::-1][:8]).tolist()
    return topics

##=================================================Other Utility Functions=====================================================
## Unstem words,may be used to decode key features
# def unstem(word):
#     return unstem[words]
# ## Indicate whether a word is a name entity word or not
# # def isEntity(word):
# #     entities=nlp(word.title())
# #     if entities.ents==():
# #         return False 
# #     else:
# #         return True
# #Takes in the query and quote, compare the sentiments of the query
# def sentimental_analysis(string):
#     #Using Vader's sentiments to display the intensity score 
#     intensity_score=analyzer.polarity_scores(string)
#     #Use the ANEW system to determine various sentimental domain
#     words = [stem(t) for t in regtokenizer.tokenize(expand_contractions(string.lower()))]
#     anew_score = np.zeros((1,3))
#     for word in words:
#         if stem(word) in word_to_attitude.keys():
#             anew_score+=np.array(word_to_attitude[stem(word)])
#     return (intensity_score,anew_score)

#
