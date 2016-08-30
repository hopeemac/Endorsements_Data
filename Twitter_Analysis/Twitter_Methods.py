
# coding: utf-8

# # Methods for Twitter Processing and Sentiment Modeling

# ### Import Packages

# In[6]:

import math
import pandas as pd
import numpy as np
import csv
import sys, os
import string
import re
import datetime
import warnings
# Natural Language Processing
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import EnglishStemmer # load the stemmer module from NLTK
dataloc = '/Volumes/Seagate Backup Plus Drive/PoliTweet/TwitterData/'
import emoji
# Sci-Kit Learn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


# ### Required Data

# In[7]:

twtokenizer = TweetTokenizer()


# In[ ]:

stemmer = EnglishStemmer() # Get an instance of SnowballStemmer for English


# In[8]:

punctuation = list(set(string.punctuation)) + ['…','’','...','—',':/','”','..', '“']


# In[9]:

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so',
'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should'] # Removed 'not','now', 'no','nor'


# In[ ]:

stopwords_short = ['the','an','a', ]


# ### Tweet Processing

# In[ ]:

# Working on Reading In Data from New File Structure
# Update to Read in Files only from Selected Date Range
def getFileList(candidate,orgDataLoc):
    'Get Full Path to all Files in the Organized Twitter Data Set for a Candidate'
    filelist = []
    dataloc = orgDataLoc+candidate+'/by_script_date/'
    walk = os.walk(dataloc)
    for dirpath, dirs, files in walk:
        for f in files:
            #print(f)
            if f != '.DS_Store':
                filelist.append((candidate,(dirpath+'/'+f)))
    if len(filelist) == 0:
        warnings.warn("Warning: No Files Found at the Input Location")
    return filelist


# In[ ]:

def readTwitterCSV(file):
    'Gets Tweets as Pandas DataFrame, Very Slow'
    data = pd.DataFrame(list(csv.reader(open(file),skipinitialspace=True)))
    newCol = []
    counter = 0
    for word in data.iloc[0]:
        if word is None:
            word = 'None'+'_'+ str(counter)
            newCol.append(word)
            counter += 1
        else:
            newCol.append(word)
    
    data.columns = newCol
    
    data = data.drop(0)
    data.dropna(axis = 0)
    return data 


# In[ ]:

def readTwitterCSVaslines(file):
    'Get Tweet Text and Date from Tweets in File, Returns Dict where Tweet ID is the Key'
    data_dict = {}
    rowcount = 0
    with open(file, errors='ignore') as f:
        content = f.readlines()
    
    # Check to make sure no Null Bytes in Tweet Data Row, if True, drop Row
    content_noNull = [row for row in content if row.find('\x00') == -1 ]
    diff = len(content) - len(content_noNull)
    if diff != 0:
        print("Dropped "+str(diff)+" Row from "+file+" due to Null Bytes")

    for row in csv.reader(content_noNull, skipinitialspace=True):
        if rowcount != 0:
            key = row[4] # statusId
            # print(key)
            data_dict[key] = (row[11],row[7]) # (statusText,statusCreatedAt)
            # print(data_dict[key])
        rowcount+=1
        
    return data_dict


# In[ ]:

def clean_tweets(tweet):
    # Need to First Clean Out URLs before Tokenization
    tweet = re.sub('htt[^ ]*' ,'URL', tweet)
    
    #tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
    cleanWords = twtokenizer.tokenize(tweet)
    
    # Convert to Lowercase
    cleanWords = [t.lower() for t in cleanWords]
    
    # Convert Emoji's to Word Label
    cleanWords = [emoji.demojize(word) for word in cleanWords]
    

    # Normalize (remove punctuation)
    #Remove punctuation
    cleanWords = [word for word in cleanWords if word not in punctuation]
    
    # punc = string.punctuation
    # cleanWords = [t for t in cleanWords if t not in punc]
    # cleanWords = [re.sub('[^0-9a-z]', "", x) for x in cleanWords]
    
    # Remove Empty Vectors
    cleanWords = [x for x in cleanWords if x != '']
     
    # Remove StopWords
    # cleanWords = [word for word in cleanWords if word not in stopwords_short]
    cleanWords = [word for word in cleanWords if word not in stopwords]
    
    # Identify Digits & Convert to Num
    # cleanWords = [re.sub("\d+", "NUM", x) for x in cleanWords]
    
    # Remove all Web/URL References (Replace with String Replacement Above)
    # Could be better to replace with 'URL'
    # cleanWords = [word for word in cleanWords if word[0:3] != 'htt']
    # cleanWords = ['URL' if word[0:3] == 'htt' else word for word in cleanWords ]
    
    # Stem Words
    #cleanWords = [stemmer.stem(x) for x in cleanWords] # call stemmer to stem the input
    
    # Remove Multiple Letters, Replace with only 3 so they are distinguishable, but standardized
    # cleanWords = [re.sub(r'(.)\1{2,}', r'\1\1\1', word) for word in cleanWords ]
    
    # Change all @ References to USER
    # cleanWords = ['USER' if word[0] == '@' else word for word in cleanWords ]
    
    
    return cleanWords


# In[1]:

def clean_tweets_opt(tweet, lower = True, demoji = True, punc = True, stopwords = [],                      num = False, url = True, stem = False, repeatedChar = False, users = False):
    # Need to Clean Out URLs before Tokenization
    if url:
        tweet = re.sub('htt[^ ]*' ,'URL', tweet)
    
    #tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
    cleanWords = twtokenizer.tokenize(tweet)
    
    # lower
    # Convert to Lowercase
    if lower:
        cleanWords = [word.lower() for word in cleanWords]
    
    # demoji
    # Convert Emoji's to Word Label
    if demoji:
        cleanWords = [emoji.demojize(word) for word in cleanWords]

    # punc
    # Remove punctuation, only removes puncutation if only char in token
    if punc:
        cleanWords = [word for word in cleanWords if word not in punctuation]
     
    # Remove StopWords
    # Preferred list passed through function parameters
    cleanWords = [word for word in cleanWords if word not in stopwords]
    
    # num
    # Identify Digits & Convert to Num
    if num:
        cleanWords = [re.sub("\d+", "NUM", x) for x in cleanWords]
    
    # url; opt = remove, replace
    # Remove all Web/URL References
    #if url:
    #    cleanWords = [word for word in cleanWords if word[0:3] != 'htt']
    # cleanWords = ['URL' if word[0:3] == 'htt' else word for word in cleanWords ]
    
    # stem
    # Stem Words
    if stem:
        cleanWords = [stemmer.stem(x) for x in cleanWords] # call stemmer to stem the input
    
    # repeatedChar
    # Remove Multiple Letters, Replace with only 3 so they are distinguishable, but standardized
    if repeatedChar:
        cleanWords = [re.sub(r'(.)\1{2,}', r'\1\1\1', word) for word in cleanWords ]
    
    # users
    # Change all @ References to USER
    if users:
        cleanWords = ['USER' if word[0] == '@' else word for word in cleanWords ]
    
    ## Non-Optional Pre-processing
    # Trim whitespace
    
    
    # Remove Empty Vectors
    cleanWords = [x for x in cleanWords if x != '']
    
    return cleanWords


# In[ ]:

def getHTMLcodes(emojis2decode, codesDict):
    'Convert Emoji Labels to HTML codes for rendering on webpage'
    uni = [str(emoji.emojize(e).encode('unicode-escape')) for e in emojis2decode]
    uni_clean = [u.replace("\\","").replace("b'","").replace("'","").replace("U000","&#x") for u in uni]
    uni_clean = ['&#x'+u[1:len(u)] if u[0] == 'u' else u for u in uni_clean]

    # codesDict = {}
    # [codes[c]['html']={} for c,u in list(zip(emoji_counts_DF.columns, uni_clean))]
    for c,u in list(zip(emojis2decode, uni_clean)):
        if c not in codesDict.keys():
            codesDict[c]={}
            codesDict[c]['htmlcode']={}
            codesDict[c]['htmlcode']=u
    return codesDict


# In[ ]:

def clean_ads(ad):
    # print(type(ad))
    #tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
    cleanWords = twtokenizer.tokenize(ad)
    
    # Convert to Lowercase
    cleanWords = [t.lower() for t in cleanWords]
    
    # Convert Emoji's to Word Label
    # cleanWords = [emoji.demojize(word) for word in cleanWords]
    

    # Normalize (remove punctuation)
    #Remove punctuation
    cleanWords = [word for word in cleanWords if word not in punctuation]
    
    # punc = string.punctuation
    # cleanWords = [t for t in cleanWords if t not in punc]
    # cleanWords = [re.sub('[^0-9a-z]', "", x) for x in cleanWords]
    
    # Remove Empty Vectors
    cleanWords = [x for x in cleanWords if x != '']
     
    # Remove StopWords
    # cleanWords = [word for word in cleanWords if word not in stopwords_short]
    cleanWords = [word for word in cleanWords if word not in stopwords]
    
    # Identify Digits & Convert to Num
    #cleanWords = [re.sub("\d+", "NUM", x) for x in cleanWords]
    
    # Remove all Web/URL References
    # Could be better to replace with 'URL'
    # cleanWords = [word for word in cleanWords if word[0:3] != 'htt']
    
    # Stem Words
    #cleanWords = [stemmer.stem(x) for x in cleanWords] # call stemmer to stem the input
    
    # Remove Multiple Letters, Replace with only 3
    
    
    return cleanWords


# ### Daily Count Tweets

# In[ ]:

def get_emoji_counts(master, emoji_counts, candidate):
    if candidate not in emoji_counts.keys():
        emoji_counts[candidate] = {}
    for key in master.keys():
        tweet = master[key][0]
        date = master[key][1]
        date = datetime.datetime.strptime(date,'%a %b %d %H:%M:%S %Z %Y')
        date_ft = date.strftime('%m-%d-%Y')
        
        # Replace all URLs in Tweet (to avoid confusion with emoticon)
        tweet = re.sub('htt[^ ]*' ,'URL', tweet)
        
        tokens = twtokenizer.tokenize(tweet)
        tokens = [emoji.demojize(token) for token in tokens]
        # tokens = [word for word in tokens if word not in string.punctuation]

        for token in tokens:
            if re.match(':+[a-z_]*:*',token):
                if date_ft not in emoji_counts[candidate].keys():
                    emoji_counts[candidate][date_ft] = {}
                if token in emoji_counts[candidate][date_ft]:
                    emoji_counts[candidate][date_ft][token] +=1
                else:
                    emoji_counts[candidate][date_ft][token] = 1
    return emoji_counts


# In[ ]:

def get_daily_tweet_counts(master, counts, candidate):
    'Get number of Tweets for each day'
    if candidate not in counts.keys():
        counts[candidate] = {}
    for key in master.keys():
        #tweet = master[key][0]
        #print(tweet)
        date = master[key][1]
        date = datetime.datetime.strptime(date,'%a %b %d %H:%M:%S %Z %Y')
        date_ft = date.strftime('%Y-%m-%d')

        if date_ft in counts[candidate]:
            counts[candidate][date_ft] +=1
        else:
            counts[candidate][date_ft] = 1
    return counts


# In[ ]:

def get_daily_tweet_counts_PAR(master, candidate):
    'Get number of Tweets for each day for use with parallel package, creates new dictionary'
    counts = {}
    if candidate not in counts.keys():
        counts[candidate] = {}
    for key in master.keys():
        # tweet = master[key][0]
        date = master[key][1]
        date = datetime.datetime.strptime(date,'%a %b %d %H:%M:%S %Z %Y')
        date_ft = date.strftime('%Y-%m-%d')

        if date_ft in counts[candidate]:
            counts[candidate][date_ft] +=1
        else:
            counts[candidate][date_ft] = 1
    return counts


# In[ ]:

def get_hashtag_counts(master, emoji_counts, candidate):
    if candidate not in emoji_counts.keys():
        emoji_counts[candidate] = {}
    for key in master.keys():
        tweet = master[key][0]
        date = master[key][1]
        date = datetime.datetime.strptime(date,'%a %b %d %H:%M:%S %Z %Y')
        date_ft = date.strftime('%m-%d-%Y')
        
        # Replace all URLs in Tweet (to avoid confusion with emoticon)
        tweet = re.sub('htt[^ ]*' ,'URL', tweet)
        
        tokens = twtokenizer.tokenize(tweet)
        
        # Only Tokens with '#' sign
        hashtags = [token.lower() for token in tokens if "#" in token]
        # tokens = [word for word in tokens if word not in string.punctuation]

        
        for token in hashtags:
            # print(token)
            if date_ft not in emoji_counts[candidate].keys():
                emoji_counts[candidate][date_ft] = {}
            if token in emoji_counts[candidate][date_ft]:
                emoji_counts[candidate][date_ft][token] +=1
            else:
                emoji_counts[candidate][date_ft][token] = 1
    return emoji_counts


# ### NB Modeling

# In[ ]:

def getTermFreq(tokenList):
    TF = {}
    #print(word)
    for word in tokenList:
        #print(word)
        if word in TF:
            TF[word] += 1
        else:
            TF[word] = 1
    return TF


# In[ ]:

def get_vocabulary(textlist):
    vocab = {}
    for row in textlist:
        for word in row:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    return vocab   


# In[1]:

def getDocFreq(textlist):
    DF = {}
    for row in textlist:
        for word in set(row):
            # print(word)
            if word in DF:
                DF[word] += 1
            else:
                DF[word] = 1
    return DF


# In[2]:

def create_countVectors(tokens):
    doc_TF = {}
    for token in tokens:
        if token in doc_TF:
            doc_TF[token] += 1
        else:
            doc_TF[token] = 1
    return doc_TF


# In[3]:

# Unigram Language Model
def genUniLM(TF):
    u_theta = pd.DataFrame.from_dict(TF, orient = "index")
    u_theta.columns = ['TF']
    # u_theta.sort('TF', ascending = False)[0:10]
    # Total Number of Words in Training Corpus
    nWords = u_theta['TF'].sum()
    nWords
    # Number of Unique Words in Training Corpus
    vSize = len(u_theta['TF'])
    vSize
    # Calculate Probabilty of Each Word by TTF/N
    u_theta['p'] = u_theta/nWords
    u_theta = u_theta.sort('TF', ascending = False)
    # Check that Probability Sums to 1
    print("Total Probability: ",u_theta['p'].sum())
    return u_theta


# In[4]:

def calc_pSmoothAdditive(tokenList, u_theta, d):
    
    vSize_train = len(u_theta)
    nWords_train = sum(u_theta['TF'])
    
    unseenWords = list(set(tokenList) - set(u_theta.index))
    #print(len(unseenWords))
    if len(unseenWords) == 0:
        return u_theta['p']
    else:
        # Build Series with all unique words in training set + unseen words from test document
        pSmooth = u_theta['TF'].append(pd.Series(([0]*len(unseenWords)), index = unseenWords))
        nWords_train += len(unseenWords)
        vSize_train += len(unseenWords)
        f = lambda x: ((x + d) / (nWords_train + d*vSize_train))
        pSmooth = pSmooth.map(f)
        return pSmooth


# In[ ]:

def calcfX_NBLin(tokenlist, prob1, prob0, pSmooth_1, pSmooth_0):
    X = 0
    for word in tokenlist:
        try:
            pX_y1 = pSmooth_1.ix[word]
        except KeyError:
            pX_y1 = 1
        try:
            pX_y0 = pSmooth_0.ix[word]
        except KeyError:
            pX_y0 = 1
            
        x = math.log(pX_y1)-math.log(pX_y0)
        X = X + x
    fX = math.log(prob1/prob0) + X

    return fX


# ### General

# In[1]:

def sampleKeys(dict):
    return list(dict)[0:10]


# In[ ]:



