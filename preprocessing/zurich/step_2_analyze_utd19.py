#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import time
import pandas as pd
import json


# # 1. read traffic data

# In[2]:


root_path = "/home/umni2/a/umnilab/users/xue120/umni4/2022_ce/course_project/"
traffic_path = root_path + "utd19_u.csv"
df_chunk = pd.read_csv(traffic_path, chunksize=10000000, iterator = True)
print ("size", sys.getsizeof(df_chunk))


# In[3]:


chunk_list = list()
i = 0
for chunk in df_chunk:
    i += 1
    print ("i", i)
    if i>=14:
        chunk_list.append(chunk)
        print ("add i", i)
df_concat = pd.concat(chunk_list)
df_concat.shape
print (len(df_concat))


# # 2. extract data associated with Zurich

# In[4]:


n_df_concat = len(df_concat)
n_zurich = list(df_concat["city"]).count("zurich")
df_zurich = df_concat[n_df_concat-n_zurich:]
print (len(df_zurich))
df_zurich[0:10]


# # 3. count traffic records marked error=1 in Zurich dataset

# In[5]:


zurich_error_list = list(df_zurich["error"])
n_zurich_error_list = len(zurich_error_list)
print ("count")
print (zurich_error_list.count(1.0))
print (zurich_error_list.count(1.0)/n_zurich_error_list)
print (zurich_error_list.count(0.0)/n_zurich_error_list)


# # 4. get the detid id with error

# In[6]:


detid_list = list(df_zurich["detid"])
error_id = list()

for i in range(len(zurich_error_list)):
    if zurich_error_list[i] == 1.0:
        error_id.append(detid_list[i])
error_id_set = set(error_id)
print (len(error_id_set))
error_dict = {"error":list(error_id_set)}


# In[7]:


savefile = open("error_id.json",'w')
json.dump(error_dict, savefile)
savefile.close()


# In[8]:


df_zurich.to_csv("utd19_u_zurich.csv")


# In[ ]:




