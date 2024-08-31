#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import shapefile
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon
from pyproj import Geod
from shapely import wkt
import json


# # 1. read traffic data associated with Zurich

# In[2]:


root_path = "/home/umni2/a/umnilab/users/xue120/umni4/2023_mfd_traffic/1_check_data/"
traffic_zurich_path = root_path + "utd19_u_zurich.csv"
df_traffic_zurich = pd.read_csv(traffic_zurich_path)
print (len(df_traffic_zurich))
#df_traffic_zurich[0:10]
print ("date: ", set(list(df_traffic_zurich["day"])))
print ("# sensor: ", len(set(list(df_traffic_zurich["detid"]))))


# # 2. check the interval

# In[3]:


zurich_interval = list(set(list(df_traffic_zurich["interval"])))
print (len(zurich_interval))
print (np.max(zurich_interval))
print (np.mean(zurich_interval))
zurich_interval.sort()


# # 3. count the number of data within one day

# In[4]:


#20151026-20151101, Mon-Sun.


# In[5]:


df_traffic_zurich_1 = df_traffic_zurich[df_traffic_zurich["day"] == '2015-10-26']
print (len(df_traffic_zurich_1))
df_traffic_zurich_2= df_traffic_zurich[df_traffic_zurich["day"] == '2015-10-27']
print (len(df_traffic_zurich_2))
df_traffic_zurich_3 = df_traffic_zurich[df_traffic_zurich["day"] == '2015-10-28']
print (len(df_traffic_zurich_3))
df_traffic_zurich_4 = df_traffic_zurich[df_traffic_zurich["day"] == '2015-10-29']
print (len(df_traffic_zurich_4))
df_traffic_zurich_5 = df_traffic_zurich[df_traffic_zurich["day"] == '2015-10-30']
print (len(df_traffic_zurich_5))
df_traffic_zurich_6 = df_traffic_zurich[df_traffic_zurich["day"] == '2015-10-31']
print (len(df_traffic_zurich_6))
df_traffic_zurich_7 = df_traffic_zurich[df_traffic_zurich["day"] == '2015-11-01']
print (len(df_traffic_zurich_7))


# # 4. build shapefiles

# In[6]:


#1. read the default shapefile
folder_census = "/home/umni2/a/umnilab/users/xue120/umni4/2022_python/"
file_census = "df.shp" 
path_census = os.path.join(folder_census, file_census) 
df_zurich = gpd.read_file(path_census)
df_zurich = df_zurich.drop(columns="id_1")
df_zurich = df_zurich.drop(index=[0,1,2])


# In[7]:


df_zurich


# In[8]:


p1, p2, p3 = [47.370478, 8.524712], [47.366947, 8.518524], [47.377578, 8.506667]
p4, p5, p6 = [47.382472, 8.515091], [47.380802, 8.528440], [47.377377, 8.533421]
poly = Polygon([(p1[1], p1[0]), (p2[1], p2[0]), (p3[1], p3[0]),\
                (p4[1], p4[0]), (p5[1], p5[0]), (p6[1], p6[0])])
df_zurich_id, df_zurich_geometry = [0], [poly]
df_zurich["id"], df_zurich["geometry"] = df_zurich_id, df_zurich_geometry 
df_zurich.plot()


# In[9]:


geod = Geod(ellps="WGS84")
area = abs(geod.geometry_area_perimeter(poly)[0])
print (area/1000/1000)


# In[10]:


df_zurich.to_file("/home/umni2/a/umnilab/users/xue120/umni4/2023_mfd_traffic/"+\
                  "1_check_data/zurich_shp/zurich.shp")


# # 5. sample sensors

# In[11]:


#find the ID of 60 loops within this region.
df_detector = pd.read_csv("/home/umni2/a/umnilab/users/xue120/umni4/2022_ce/course_project/detectors_public.csv")
df_detector_Zurich = df_detector[df_detector["citycode"]=="zurich"] 
print ("# Zurich row", len(df_detector_Zurich))

id_list, lon_list, lat_list = list(df_detector_Zurich["detid"]), list(df_detector_Zurich["long"]), list(df_detector_Zurich["lat"])
Zurich_id_lon_lat = {str(id_list[i]):[lon_list[i], lat_list[i]] for i in range(len(id_list))}


# In[12]:


f = open("error_id.json")
error_id = json.load(f)
f.close()


# In[13]:


zurich_within_region = list()
for key in Zurich_id_lon_lat:
    point = Point(Zurich_id_lon_lat[key][0], Zurich_id_lon_lat[key][1])
    if point.within(poly) and key not in error_id["error"]:
        zurich_within_region.append(key)
len(set(zurich_within_region))
zurich_within_region_dict = {str(zurich_within_region[i]):0 for i in range(len(zurich_within_region))}


# In[14]:


len(zurich_within_region_dict)


# # 6. build temporal figures

# In[15]:


x_list, y_list = [i*180 for i in range(int(86400/180))], [0.0 for i in range(int(86400/180))]


# # Monday-Friday

# In[34]:


df_traffic_zurich_day = df_traffic_zurich_1
region_df_traffic_zurich_day = df_traffic_zurich_1[df_traffic_zurich_1["detid"].isin(zurich_within_region)]
print (len(region_df_traffic_zurich_day)/len(df_traffic_zurich_day))


# In[35]:


region_df_traffic_zurich_day


# # regional

# In[36]:


region_time_list = list(region_df_traffic_zurich_day[0:].interval)
region_flow_list = list(region_df_traffic_zurich_day[0:].flow)
region_occ_list = list(region_df_traffic_zurich_day[0:].occ)
print (len(region_time_list))
print (len(region_flow_list))
print (len(region_occ_list))

region_index_flow_dict = {i:[] for i in range(int(86400/180))}
region_index_occ_dict = {i:[] for i in range(int(86400/180))}
region_flow_y = [i for i in range(int(86400/180))]
region_occ_y = [i for i in range(int(86400/180))]

for i in range(len(region_time_list)):
    time_i, flow_i, occ_i = region_time_list[i], region_flow_list[i], region_occ_list[i]
    index = int(time_i/180)
    region_index_flow_dict[index].append(flow_i)
    region_index_occ_dict[index].append(occ_i)

print (len(region_index_flow_dict[1]))

for j in range(len(region_index_flow_dict)):
    region_flow_y[j] = np.mean(region_index_flow_dict[j])
    region_occ_y[j] = np.mean(region_index_occ_dict[j])    
    
region_hour_x = [i/20.0 for i in range(int(86400/180))]


# In[53]:


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5,4),dpi=300)
l1 = plt.scatter(region_hour_x, region_flow_y, marker="o",  s=10, linewidth=0.5,\
                 c="white", alpha=1.0, edgecolor=["red"])
#plt.plot(region_hour_x, region_flow_y, "red", "-o", linewidth=2,ms=1, label='Nov. 1, 2015')
#plt.title("Average traffic on Nov. 1, 2015 (Region)")


#plt.legend()
plt.xlabel("Hour",fontsize=26)
plt.ylabel("Average flow \n rate (veh/hour)",fontsize=24)
plt.xticks([0,6,12,18,24], [0,6,12,18,24])
plt.yticks([i*150 for i in range(4)], [i*150 for i in range(4)])
plt.xticks(fontsize=22, rotation=0)
plt.yticks(fontsize=22, rotation=0)
#plt.ylim(16, 30)
plt.savefig('flow.svg',bbox_inches = 'tight')
plt.show()


# In[73]:


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5,4),dpi=300)
l2 = plt.scatter(region_hour_x, region_occ_y, marker="o",  s=10, linewidth=0.5,\
                 c="white", alpha=1.0, edgecolor=["blue"])
#plt.title("Average traffic on Nov. 1, 2015 (Region)")

plt.xlabel("Hour",fontsize=26)
plt.ylabel("Average\n  occupancy",fontsize=24)
plt.xticks([0,6,12,18,24], [0,6,12,18,24])
plt.xticks(fontsize=22, rotation=0)
plt.yticks(fontsize=22, rotation=0)
plt.yticks([0, 0.1, 0.2, 0.3], [0, 0.1, 0.2, 0.3])
#plt.ylim(16, 30)
plt.savefig("occ.svg",bbox_inches = 'tight')
plt.show()


# # full

# In[21]:


time_list = list(df_traffic_zurich_day[0:].interval)
flow_list = list(df_traffic_zurich_day[0:].flow)
occ_list = list(df_traffic_zurich_day[0:].occ)
print (len(time_list))
print (len(flow_list))
print (len(occ_list))

index_flow_dict = {i:[] for i in range(int(86400/180))}
index_occ_dict = {i:[] for i in range(int(86400/180))}
flow_y = [i for i in range(int(86400/180))]
occ_y = [i for i in range(int(86400/180))]

for i in range(len(time_list)):
    time_i, flow_i, occ_i = time_list[i], flow_list[i], occ_list[i]
    index = int(time_i/180)
    index_flow_dict[index].append(flow_i)
    index_occ_dict[index].append(occ_i)
print (len(index_flow_dict[1]))

for j in range(len(flow_y)):
    if len(index_flow_dict[j]) == 0:
        flow_y[j] = 0
    else:
        flow_y[j] = np.mean(index_flow_dict[j])
    if len(index_occ_dict[j]) == 0:
        occ_y[j] = 0
    else:
        occ_y[j] = np.mean(index_occ_dict[j])    
    
hour_x = [i/20.0 for i in range(int(86400/180))]


# In[22]:


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5,4),dpi=300)
plt.plot(hour_x, flow_y, "blue", linewidth=0.5,ms=1, label='Nov. 1, 2015')
plt.title("Average traffic on Nov. 1, 2015 (Full)")

plt.xticks(fontsize=12,rotation=0)
plt.yticks(fontsize=12,rotation=0)
#plt.legend()

plt.xlabel("Hour",fontsize=12)
plt.ylabel("Average flow on sensors (veh/hour)",fontsize=12)
plt.xticks([0,3,6,9,12,15,18,21,24], [0,3,6,9,12,15,18,21,24])
plt.yticks([i*100 for i in range(6)], [i*100 for i in range(6)])
#plt.ylim(16, 30)
#plt.savefig('nyc_2020.png',bbox_inches = 'tight')
plt.show()


# In[23]:


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5,4),dpi=300)
plt.plot(hour_x, occ_y, "blue", linewidth=0.5,ms=1, label='Nov. 1, 2015')
plt.title("Average traffic on Nov. 1, 2015 (Full)")

plt.xticks(fontsize=12,rotation=0)
plt.yticks(fontsize=12,rotation=0)
#plt.legend()

plt.xlabel("Hour",fontsize=12)
plt.ylabel("Average occ on sensors",fontsize=12)
plt.xticks([0,3,6,9,12,15,18,21,24], [0,3,6,9,12,15,18,21,24])
#plt.yticks([i*200 for i in range(6)], [i*200 for i in range(6)])
#plt.ylim(16, 30)
#plt.savefig('nyc_2020.png',bbox_inches = 'tight')
plt.show()


# # MFD

# In[24]:


len(region_occ_list)
len(region_flow_y)


# In[25]:


plt.figure(figsize=(4,1.5),dpi=300)
l1 = plt.scatter(region_occ_y, region_flow_y , s=1, c= 'r', marker=".", label="Nov. 1, 2015")

plt.xlabel("x: occ", fontsize = 10)
plt.ylabel("y: flow", fontsize = 10)

#my_x_ticks = np.arange(0, 3700, 600)
#my_y_ticks = np.arange(0, 121, 40)
#plt.xticks(my_x_ticks)
#plt.yticks(my_y_ticks) 

plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plt.title("Traffic flow, occ (Region)", fontsize = 8)
plt.legend(loc=4)
plt.grid()
#plt.savefig('baseline1.pdf',bbox_inches = 'tight')
#plt.savefig('1_cell_1.png',bbox_inches = 'tight')
plt.show()


# In[26]:


plt.figure(figsize=(4,1.5),dpi=300)
l1 = plt.scatter(occ_y, flow_y , s=1, c= 'blue', marker=".", label="Nov. 1, 2015")

plt.xlabel('x: occ',fontsize = 10)
plt.ylabel("y: flow", fontsize = 10)

#my_x_ticks = np.arange(0, 3700, 600)
#my_y_ticks = np.arange(0, 121, 40)
#plt.xticks(my_x_ticks)
#plt.yticks(my_y_ticks) 

plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plt.title("Traffic flow, occ (Full)", fontsize = 8)
plt.legend(loc=4)
plt.grid()
#plt.savefig('baseline1.pdf',bbox_inches = 'tight')
#plt.savefig('1_cell_1.png',bbox_inches = 'tight')
plt.show()


# In[ ]:




