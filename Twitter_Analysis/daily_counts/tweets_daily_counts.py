
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


# ### Tweet Counts

# In[5]:

orgDataLoc = "/nv/vol165/STKING-LAB/Twitter_Organized_Master/from/"
# orgDataLoc = "/Volumes/Seagate Backup Plus Drive/Twitter_Organized_Master/hash/"
candidates = os.listdir(orgDataLoc)


# In[6]:

start = datetime.datetime.now()

for candidate in candidates[0:2]:
    
    filelist = tm.getFileList(candidate,orgDataLoc)

    # Get Daily Tweet Counts -- Not Parrallelized
    
    tweet_counts = {}

    for file in filelist[0:10]:
        tweet_counts = tm.get_daily_tweet_counts(tm.readTwitterCSVaslines(file[1]), tweet_counts, file[0])

    # CSV Output is Sorted
    pd.DataFrame.from_dict(tweet_counts).to_csv('./output/'+candidate+' - DailyCounts_atTweets.csv')

runtime = datetime.datetime.now()-start
print(str(runtime))


# In[ ]:



