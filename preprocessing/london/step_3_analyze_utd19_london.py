#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import shapefile
import numpy as np
import pandas as pd
import geopandas as gpd


# In[2]:


import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon
from pyproj import Geod
from shapely import wkt
import json


# # 1. read traffic data associated with London

# In[3]:


root_path = "/home/umni2/a/umnilab/users/xue120/umni4/2023_mfd_traffic_london/1_check_data/"
traffic_london_path = root_path + "utd19_u_london.csv"
df_traffic_london = pd.read_csv(traffic_london_path)
print (len(df_traffic_london))
#df_traffic_zurich[0:10]


# In[4]:


date_list = list(set(list(df_traffic_london["day"])))
date_list.sort()
print ("date list: ", date_list)
print ("# sensor: ", len(set(list(df_traffic_london["detid"]))))


# # 2. check the interval

# In[5]:


london_interval = list(set(list(df_traffic_london["interval"])))
print (len(london_interval))
print (np.max(london_interval))
print (np.mean(london_interval))
london_interval.sort()


# # 3. count the number of data within one day

# In[6]:


#'2015-05-15', '2015-05-16', '2015-05-17': Fri, Sat, Sun
#'2015-05-18', '2015-05-19', '2015-05-20': Mon, Tue, Wed
#'2015-05-21', '2015-05-22', '2015-05-23': Thu, Fri, Sat


# In[7]:


df_traffic_london_1 = df_traffic_london[df_traffic_london["day"] == '2015-05-15']
print (len(df_traffic_london_1))
df_traffic_london_2= df_traffic_london[df_traffic_london["day"] == '2015-05-16']
print (len(df_traffic_london_2))
df_traffic_london_3 = df_traffic_london[df_traffic_london["day"] == '2015-05-17']
print (len(df_traffic_london_3))
df_traffic_london_4 = df_traffic_london[df_traffic_london["day"] == '2015-05-18']
print (len(df_traffic_london_4))
df_traffic_london_5 = df_traffic_london[df_traffic_london["day"] == '2015-05-19']
print (len(df_traffic_london_5))
df_traffic_london_6 = df_traffic_london[df_traffic_london["day"] == '2015-05-20']
print (len(df_traffic_london_6))
df_traffic_london_7 = df_traffic_london[df_traffic_london["day"] == '2015-05-21']
print (len(df_traffic_london_7))
df_traffic_london_8 = df_traffic_london[df_traffic_london["day"] == '2015-05-22']
print (len(df_traffic_london_8))
df_traffic_london_9 = df_traffic_london[df_traffic_london["day"] == '2015-05-23']
print (len(df_traffic_london_9))


# # 4. build shapefiles

# In[8]:


#1. read the default shapefile
folder_census = "/home/umni2/a/umnilab/users/xue120/umni4/2022_python/"
file_census = "df.shp" 
path_census = os.path.join(folder_census, file_census) 
df_london = gpd.read_file(path_census)
df_london = df_london.drop(columns="id_1")
df_london = df_london.drop(index=[0,1,2])


# In[9]:


p1, p2, p3 = [51.508924, -0.119622], [51.525838, -0.136468], [51.532099, -0.120614]
p4, p5, p6 = [51.522548, -0.105431], [51.517284, -0.104093], [51.510825, -0.103875]
poly = Polygon([(p1[1], p1[0]), (p2[1], p2[0]), (p3[1], p3[0]),\
                (p4[1], p4[0]), (p5[1], p5[0]), (p6[1], p6[0])])
df_london_id, df_london_geometry = [0], [poly]
df_london["id"], df_london["geometry"] = df_london_id, df_london_geometry 


# In[10]:


df_london.plot()


# In[12]:


df_london.to_file("/home/umni2/a/umnilab/users/xue120/umni4/"+\
                  "2023_mfd_traffic_london/1_check_data/london_shp/london.shp")


# In[13]:


geod = Geod(ellps="WGS84")
area = abs(geod.geometry_area_perimeter(poly)[0])
print (area/1000/1000)


# # 5. sample sensors

# In[14]:


#find the ID of loops within this region.
df_detector = pd.read_csv("/home/umni2/a/umnilab/users/xue120/"+\
                          "umni4/2022_ce/course_project/detectors_public.csv")
df_detector_london = df_detector[df_detector["citycode"]=="london"] 
print ("# London row", len(df_detector_london))

id_list, lon_list, lat_list = list(df_detector_london["detid"]), list(df_detector_london["long"]),\
list(df_detector_london["lat"])
london_id_lon_lat = {str(id_list[i]):[lon_list[i], lat_list[i]] for i in range(len(id_list))}


# In[15]:


f = open("error_id.json")
error_id = json.load(f)
f.close()
print ("the number of sensors with error", len(error_id["error"]))


# In[16]:


london_within_region = list()
for key in london_id_lon_lat:
    point = Point(london_id_lon_lat[key][0], london_id_lon_lat[key][1])
    if point.within(poly) and key not in error_id["error"]:
        london_within_region.append(key)
len(set(london_within_region))
london_within_region_dict = {str(london_within_region[i]):0 for i in range(len(london_within_region))}


# In[17]:


print (len(london_within_region_dict))


# In[ ]:




