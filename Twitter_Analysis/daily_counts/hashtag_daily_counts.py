
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
import json


# In[2]:

import Twitter_Methods as tm


# ### Hashtag Counts

# In[3]:

# orgDataLoc = "/nv/vol165/STKING-LAB/Twitter_Organized_Master/from/"
orgDataLoc = "/Volumes/Seagate Backup Plus Drive/Twitter_Organized_Master/at/"
candidates = os.listdir(orgDataLoc)


# In[7]:

start = datetime.datetime.now()

for candidate in ['BernieSanders']:
    
    filelist = tm.getFileList(candidate,orgDataLoc)

    # Get Daily Hashtag Counts -- Not Parrallelized
    
    hashtag_counts = {}

    for file in filelist:
        hashtag_counts = tm.get_hashtag_counts(tm.readTwitterCSVaslines(file[1]), hashtag_counts, file[0])

    # Output to JSON
    with open('./output/'+candidate+' - DailyCounts_hashtags.json', 'w') as f:
        json.dump(hashtag_counts, f)
    
    # Output to CSV
    hashtag_counts_df = pd.DataFrame.from_dict(hashtag_counts[candidate]).transpose()
    hashtag_counts_df.to_csv('./output/'+candidate+' - DailyCounts_hashtags.csv')

runtime = datetime.datetime.now()-start
print(str(runtime))


# In[5]:

# Remove Hashtags Used during Window only 1 Time
    #print(len(tweet_counts_df.columns))
    #for column in tweet_counts_df.columns:
        #if tweet_counts_df[column].max() <= 1:
            #del tweet_counts_df[column]
    #print(len(tweet_counts_df.columns))

