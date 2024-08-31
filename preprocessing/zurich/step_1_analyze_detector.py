#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # 1. read detector data

# In[2]:


root_path = "/home/umni2/a/umnilab/users/xue120/umni4/2023_mfd_traffic/1_check_data/"
detector_path = root_path + "detectors_public.csv"
df_detector = pd.read_csv(detector_path)
print ("# row", len(df_detector))
print ("# column", len(df_detector.columns))
print (list(df_detector.columns))


# In[3]:


df_detector.describe()


# # 2. examine the data for Zurich

# In[4]:


df_detector_Zurich = df_detector[df_detector["citycode"]=="zurich"] 
print ("# Zurich row", len(df_detector_Zurich))
df_detector_Zurich_road = set(list(df_detector_Zurich["road"]))
print ("# Zurich roads", len(df_detector_Zurich_road))


# In[5]:


df_detector_Zurich.describe()


# In[6]:


df_detector_Zurich[0:5]


# In[7]:


print (list(set(list(df_detector_Zurich["linkid"]))))


# # 3. check the number of lanes for sensors

# In[8]:


lane_list = list(df_detector_Zurich["lanes"])
n_lane_list = len(lane_list)
print (len(lane_list))
print (np.max(lane_list))
print (np.min(lane_list))


# In[ ]:




