
# coding: utf-8

# In[1]:

import math
import pandas as pd
import numpy as np
import csv
import sys, os
import string
import re
import datetime
import importlib
import warnings
# Parrallelization
from joblib import Parallel, delayed
import multiprocessing
# Natural Language Processing
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import EnglishStemmer # load the stemmer module from NLTK
import emoji


# In[2]:

import Twitter_Methods as tm


# In[3]:

# Working on Reading In Data from New File Structure
# Update to Read in Files only from Selected Date Range
def getFileList(candidate,orgDataLoc):
    'Get Full Path to all Files in the Organized Twitter Data Set'
    filelist = []
    dataloc = orgDataLoc+candidate+'/by_script_date/'
    walk = os.walk(dataloc)
    for dirpath, dirs, files in walk:
        for f in files:
            if f != '.DS_Store':
                filelist.append((candidate,(dirpath+'/'+f)))
    return filelist


# ### Get File Information

# In[4]:

dataloc = '/Volumes/Seagate Backup Plus Drive/PoliTweet/TwitterData/'


# In[5]:

filelist = []


# In[6]:

candidate = 'HillaryClinton'
orgDataLoc = "/Volumes/Seagate Backup Plus Drive/Twitter_Organized_Master/at/"
filelist = getFileList(candidate,orgDataLoc)


# In[7]:

candidate = 'realDonaldTrump'
orgDataLoc = "/Volumes/Seagate Backup Plus Drive/Twitter_Organized_Master/at/"
filelist2 = getFileList(candidate,orgDataLoc)


# In[8]:

len(filelist)


# In[9]:

files = filelist+filelist2


# In[10]:

len(files)


# ### Methods

# In[11]:

# Stores Tweets as Tuples instead of in new dict
def readTwitterCSV_dict(file):
    data = pd.DataFrame(list(csv.reader(open(file, errors='ignore'),skipinitialspace=True)))
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
    
    data = data.reset_index()
    # Retain only important data
    data_dict = {}
    for i in range(0,len(data)):
        key = data.loc[i,'statusId']
        data_dict[key] = (data.loc[i,'statusText'], data.loc[i,'statusCreatedAt'])
    return data_dict


# In[12]:

def readTwitterCSVaslines(file):
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


# In[13]:

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


# In[14]:

def getHTMLcodes(emojis2decode, codesDict):
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


# ### Emoji Count Analysis

# Emoji's seem to be missing from the early dates of the Tweet set, not sure how far back it goes. Might be due to a change in the script?

# In[15]:

# Run emoji_counts for only one file at a time, saving memory space
start = datetime.datetime.now()
emoji_counts = {}
num_tweets = 0
# files = filelist[len(filelist)-500:len(filelist)]+filelist2[len(filelist2)-500:len(filelist2)]
for file in files:
    # print(file)
    # Read in One Tweet File
    data = readTwitterCSVaslines(file[1])
    num_tweets = num_tweets + len(data)
    # Calculate Emoji Count
    emoji_counts = get_emoji_counts(data, emoji_counts, file[0])
print("Total number of Tweets: ",num_tweets)
runtime = datetime.datetime.now()-start
print(str(runtime))


# In[16]:

os.system('say "your program is finished"')


# In[17]:

codes = {}
candidates = list(emoji_counts.keys())
emoji_counts_json = {}
for candidate in candidates:
    #print(candidate)
    emoji_counts_DF = pd.DataFrame.from_dict(emoji_counts[candidate], orient = 'index', dtype="float")
    #print(type(emoji_counts_DF.loc['12-14-2015',':(']))
    del emoji_counts_DF[':']
    # Filter Emoji's to Only Emoji's that occur in the period over 50 times.
    count = 0
    for column in emoji_counts_DF.columns:
        if emoji_counts_DF[column].max() < 100 or 'emoji_modifier' in column:
            del emoji_counts_DF[column]
        else:
            count +=1
    #print(emoji_counts_DF)
    #print(count)
    print(len(emoji_counts_DF.columns))
    # 133 with greater than 5 observations
    # 16 with > 100 obs
    # 34 with > 50 obs
    # Need to convert np.NaN to 0's before exporting...
    
    emoji_counts_DF = emoji_counts_DF.fillna(0)
    # Also should sort, maybe alphabetical?
    d = emoji_counts_DF.to_dict(orient='index')
    #print(type(d['12-14-2015'][':(']))
    emoji_counts_json[candidate] = emoji_counts_DF.to_dict(orient='index')
    
    
    codes = getHTMLcodes(emoji_counts_DF.columns, codes)


# In[18]:

saveloc = '/Users/hopeemac/Documents/Code/GIT/atCandidateEmojis/' # Always put '/' with folder 


# In[19]:

with open(saveloc+'emoji_counts.json', 'w') as f:
     json.dump(emoji_counts_json, f)


# In[20]:

with open(saveloc+'e_codes.json', 'w') as f:
     json.dump(codes, f)


# In[21]:

# Save All Emoji Counts, Not Filtered Down
candidates = list(emoji_counts.keys())

all_emoji_counts_json = {}

for candidate in candidates:
  
    emoji_counts_DF = pd.DataFrame.from_dict(emoji_counts[candidate], orient = 'index', dtype="float")

    emoji_counts_DF = emoji_counts_DF.fillna(0)

    d = emoji_counts_DF.to_dict(orient='index')
    
    all_emoji_counts_json[candidate] = emoji_counts_DF.to_dict(orient='index')


# In[22]:

with open(saveloc+'emoji_counts_all.json', 'w') as f:
     json.dump(all_emoji_counts_json, f)


# # Filter Emoji's to Only Emoji's that occur in the period over 50 times.
# count = 0
# for column in emoji_counts_DF.columns:
#     if emoji_counts_DF[column].max() < 100 or 'emoji_modifier' in column:
#         del emoji_counts_DF[column]
#     else:
#         count +=1
#         
# print(count)
# print(len(emoji_counts_DF.columns))
# # 133 with greater than 5 observations
# # 16 with > 100 obs
# # 34 with > 50 obs

# # Need to convert np.NaN to 0's before exporting...
# 
# # Also should sort, maybe alphabetical?

# In[ ]:

# Working with Dict

# Remove all 'emoji_modifier'

# Remove all if max count b/t all days and all candidates is less than 100

# Fill Out 0's


# In[26]:




# In[ ]:




# emoji_counts_json = emoji_counts_DF.to_json(orient='index')

# In[28]:




# In[ ]:

# Remove Duplicate of same Emoji in 1 Tweet
# Drop the skin color emojis
# Sort Emoji Meta alphabetically


# ### Create Emoji Decoder for D3

# In[39]:




# In[ ]:




# ### Basement
# 

# In[ ]:

emoji_counts_DF = emoji_counts_DF.fillna(0)


# In[ ]:

emoji_counts_DF.to_csv(saveloc+'emoji_counts.csv', index_label='date')


# In[ ]:

emoji_counts_DF = pd.DataFrame.from_dict(emoji_counts[candidate], orient = 'index')
print(type(emoji_counts_DF.loc['12-14-2015',':(']))


# In[ ]:

candidates = list(emoji_counts.keys())
print(candidates)
dates = list(emoji_counts[candidates[0]].keys())
print(dates)
arrays = [candidates,dates]
tuples =[]
for c in candidates:
    for d in dates:
        tuples.append((c,d))
print(tuples)
index = pd.MultiIndex.from_tuples(tuples, names=['canidate', 'date'])
print(index)


# In[ ]:

with open(saveloc+'e_codes.json', 'w') as f:
     json.dump(codes, f)


# In[ ]:

with open(saveloc+'e_codes.json', 'w') as f:
     json.dump(e_codes_json, f)


# In[ ]:

codes = {}
codes = getHTMLcodes(emoji_counts_DF.columns, codes)


# In[ ]:

def getHTMLcodes(emoji_counts_DF):
    uni = [str(emoji.emojize(e).encode('unicode-escape')) for e in emoji_counts_DF.columns]
    uni_clean = [u.replace("\\","").replace("b'","").replace("'","").replace("U000","&#x") for u in uni]
    uni_clean = ['&#x'+u[1:len(u)] if u[0] == 'u' else u for u in uni_clean]
    e_codes = pd.DataFrame()
    e_codes['name'] = emoji_counts_DF.columns
    e_codes['htmlcode'] = uni_clean
    e_codes
    e_codes.index = e_codes['name']
    del e_codes['name']
    e_codes_json = e_codes.to_json(orient='index')
    return e_codes_json


# In[ ]:

uni = [str(emoji.emojize(e).encode('unicode-escape')) for e in emoji_counts_DF.columns]
uni_clean = [u.replace("\\","").replace("b'","").replace("'","").replace("U000","&#x") for u in uni]
uni_clean = ['&#x'+u[1:len(u)] if u[0] == 'u' else u for u in uni_clean]

codes = {}
# [codes[c]['html']={} for c,u in list(zip(emoji_counts_DF.columns, uni_clean))]
for c,u in list(zip(emoji_counts_DF.columns, uni_clean)):
    codes[c]={}
    codes[c]['htmlcode']={}
    codes[c]['htmlcode']=u


# In[ ]:

uni = [str(emoji.emojize(e).encode('unicode-escape')) for e in emoji_counts_DF.columns]
uni_clean = [u.replace("\\","").replace("b'","").replace("'","").replace("U000","&#x") for u in uni]
uni_clean = ['&#x'+u[1:len(u)] if u[0] == 'u' else u for u in uni_clean]
e_codes = pd.DataFrame()
e_codes['name'] = emoji_counts_DF.columns
e_codes['htmlcode'] = uni_clean
e_codes
e_codes.index = e_codes['name']
del e_codes['name']
e_codes_json = e_codes.to_json(orient='index')
with open(saveloc+'e_codes.json', 'w') as f:
     json.dump(e_codes_json, f)


# In[ ]:

#def get_emoji_counts(master):
    emoji_counts = {}
    for i in range(0,len(master)):
        tweet = master.loc[i,'statusText']
        date = master.loc[i,'statusCreatedAt']
        date = datetime.datetime.strptime(date,'%a %b %d %H:%M:%S %Z %Y')
        date_ft = date.strftime('%Y_%m_%d')

        tokens = twtokenizer.tokenize(tweet)
        cleanWords = [word for word in cleanWords if word[0:3] != 'htt']
        tokens = [emoji.demojize(token) for token in tokens if token != ':']
        # tokens = [word for word in tokens if word not in string.punctuation]

        for token in tokens:
            if re.match(':+*:',token):
                if date_ft not in emoji_counts.keys():
                    emoji_counts[date_ft] = {}
                if token in emoji_counts[date_ft]:
                    emoji_counts[date_ft][token] +=1
                else:
                    emoji_counts[date_ft][token] = 1
    return emoji_counts


# In[ ]:

#def readTwitterCSV_dict(file):
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
    
    data = data.reset_index()
    # Retain only important data
    data_dict = {}
    for i in range(0,len(data)):
        key = data.loc[i,'statusId']
        data_dict[key] = {}
        data_dict[key]['statusText'] = data.loc[i,'statusText']
        data_dict[key]['statusCreatedAt'] = data.loc[i,'statusCreatedAt']
    
    return data_dict


# In[ ]:

# SUPER SLOW -- Appened too much to DF
start = datetime.datetime.now()
master = {}
num_tweets = 0
for file in filelist[0:10]:
    temp = readTwitterCSV_dict(dataloc+file)
    
    print(file)
    # print(len(temp))
    num_tweets = num_tweets + len(temp)
    # Merge Dictionaries
    master.update(temp)
print("Total number of Tweets: ",num_tweets)
runtime = datetime.datetime.now()-start
print(str(runtime))


# In[ ]:

#def get_emoji_counts2(master):
    emoji_counts = {}
    for key in master.keys():
        tweet = master[key][0]
        date = master[key][1]
        date = datetime.datetime.strptime(date,'%a %b %d %H:%M:%S %Z %Y')
        date_ft = date.strftime('%Y_%m_%d')

        tokens = twtokenizer.tokenize(tweet)
        cleanWords = [word for word in cleanWords if word[0:3] != 'htt']
        tokens = [emoji.demojize(token) for token in tokens if token != ':']
        # tokens = [word for word in tokens if word not in string.punctuation]

        for token in tokens:
            if re.match(':+[a-z_]*:*',token):
                if date_ft not in emoji_counts.keys():
                    emoji_counts[date_ft] = {}
                if token in emoji_counts[date_ft]:
                    emoji_counts[date_ft][token] +=1
                else:
                    emoji_counts[date_ft][token] = 1
    return emoji_counts


# In[ ]:

start = datetime.datetime.now()
emoji_counts = get_emoji_counts2(master)
runtime = datetime.datetime.now()-start
print(str(runtime))

