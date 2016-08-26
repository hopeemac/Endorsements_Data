
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


# In[2]:

import Twitter_Methods as tm


# ### Tweet Counts

# In[3]:

orgDataLoc = "/nv/vol165/STKING-LAB/Twitter_Organized_Master/hash/"
# orgDataLoc = "/Volumes/Seagate Backup Plus Drive/Twitter_Organized_Master/hash/"
candidates = os.listdir(orgDataLoc)


# In[4]:

start = datetime.datetime.now()

for candidate in candidates[0:2]:
    
    filelist = tm.getFileList(candidate,orgDataLoc)

    # Get Daily Tweet Counts -- Not Parrallelized
    
    tweet_counts = {}

    for file in filelist[0:10]:
        output = Parallel(n_jobs=3)(delayed(tm.get_daily_tweet_counts_PAR)(tm.readTwitterCSVaslines(file[1]), file[0])                    for file in filelist[0:2])
        master = {}
        for o in output:
            # print(o)
            for k in o.keys():
                for date in o[k].keys():
                    if k not in master.keys():
                        master[k] = {}
                    if date in master[k].keys():
                        master[k][date] = master[k][date] + o[k][date]
                    else:
                        master[k][date] = o[k][date]

        # CSV Output is Sorted
        pd.DataFrame.from_dict(master).to_csv('./output/'+candidate+' - DailyTweet_AtCounts.csv')

runtime = datetime.datetime.now()-start
print(str(runtime))


# In[ ]:



