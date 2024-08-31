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
    print ("i", i)
    if i in [3,4,5,6,7]:
        chunk_list.append(chunk)
        print ("add i", i)
    i += 1
df_concat = pd.concat(chunk_list)
df_concat.shape
print (len(df_concat))


# # 2. extract data associated with London

# In[4]:


n_df_concat = len(df_concat)
print (n_df_concat)


# In[5]:


n_hamburg = list(df_concat["city"]).count("hamburg")
print (n_hamburg)
n_innsbruck = list(df_concat["city"]).count("innsbruck")
print (n_innsbruck)
n_kassel = list(df_concat["city"]).count("kassel")
print (n_kassel)


# In[6]:


n_london = list(df_concat["city"]).count("london")
print (n_london)


# In[7]:


df_london = df_concat[n_hamburg+n_innsbruck+n_kassel: n_hamburg+n_innsbruck+n_kassel+n_london]
print (len(df_london))
df_london[0:10]


# In[8]:


df_london.to_csv("utd19_u_london.csv")


# # 3. count traffic records marked error=1 in London dataset

# In[9]:


london_error_list = list(df_london["error"])
n_london_error_list = len(london_error_list)
print ("count")
print (london_error_list.count(1.0)/n_london_error_list)
print (london_error_list.count(1.0))
print (london_error_list.count(0.0)/n_london_error_list)


# In[10]:


n_london_error_list


# # 4. get the detid id with error

# In[11]:


detid_list = list(df_london["detid"])
error_id_count = dict()

for i in range(len(london_error_list)):
    if london_error_list[i] == 1.0:
        detid = detid_list[i]
        if detid not in error_id_count:
            error_id_count[detid] = 1
        else:
            error_id_count[detid] = error_id_count[detid] + 1
            
error_id_set = list()
for detid_id in error_id_count:
    if error_id_count[detid_id] > 0.05 * 23 * 12 * 24:
        error_id_set.append(detid_id)
print(len(error_id_set))


# In[12]:


error_dict = {"error":list(error_id_set)}
savefile = open("error_id.json",'w')
json.dump(error_dict, savefile)
savefile.close()


# In[ ]:




