#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json


# # 1. read detector data

# In[2]:


root_path = "/home/umni2/a/umnilab/users/xue120/umni4/2023_mfd_traffic_london/1_check_data/"
detector_path = root_path + "detectors_public.csv"
df_detector = pd.read_csv(detector_path)
print ("# row", len(df_detector))
print ("# column", len(df_detector.columns))
print (list(df_detector.columns))


# In[3]:


df_detector.describe()


# # 2. examine the data for London

# In[4]:


df_detector_london = df_detector[df_detector["citycode"]=="london"] 
print ("# london row", len(df_detector_london))
df_detector_london_road = set(list(df_detector_london["road"]))
print ("# london roads", len(df_detector_london_road))


# In[5]:


df_detector_london.describe()


# In[6]:


df_detector_london.head(5)


# # 3. check the number of lanes for sensors

# In[7]:


lane_list = list(df_detector_london["lanes"])
n_lane_list = len(lane_list)
print (n_lane_list)
print (lane_list.count(1.000000))
print (lane_list.count(2.000000))
print (lane_list.count(3.000000))


# In[8]:


london_sensor_id = list(df_detector_london["detid"])
london_sensor_lane = list(df_detector_london["lanes"])
sensor_lane_dict = dict()
for i in range(len(london_sensor_id)):
    if london_sensor_id[i] not in sensor_lane_dict:
        sensor_lane_dict[str(london_sensor_id[i])] = london_sensor_lane[i]


# In[9]:


len(sensor_lane_dict)


# In[10]:


savefile = open("sensor_lane.json",'w')
json.dump(sensor_lane_dict, savefile)
savefile.close()


# In[ ]:




