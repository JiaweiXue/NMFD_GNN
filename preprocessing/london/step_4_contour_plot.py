#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import json
import copy
import numpy as np
import pandas as pd
from matplotlib import ticker
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon


# # 1. prepare input for the contour plot

# # 1.1. read the data

# In[2]:


df_traffic_london = pd.read_csv("utd19_u_london.csv")
print (set(list(df_traffic_london["day"])))


# In[3]:


day_list = list(set(df_traffic_london["day"]))
day_list.sort()


# In[4]:


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


# # 1.2. idx in the region

# In[5]:


p1, p2, p3 = [51.508924, -0.119622], [51.525838, -0.136468], [51.532099, -0.120614]
p4, p5, p6 = [51.522548, -0.105431], [51.517284, -0.104093], [51.510825, -0.103875]
poly = Polygon([(p1[1], p1[0]), (p2[1], p2[0]), (p3[1], p3[0]),\
                (p4[1], p4[0]), (p5[1], p5[0]), (p6[1], p6[0])])


# In[6]:


#find the ID of 60 loops within this region.
df_detector = pd.read_csv("/home/umni2/a/umnilab/users/xue120/umni4/2022_ce/course_project/detectors_public.csv")
df_detector_london = df_detector[df_detector["citycode"]=="london"] 
print ("# london row", len(df_detector_london))


# In[7]:


id_list = list(df_detector_london["detid"])
lon_list = list(df_detector_london["long"])
lat_list = list(df_detector_london["lat"])
london_id_lon_lat = {str(id_list[i]):[lon_list[i], lat_list[i]] for i in range(len(id_list))}


# In[8]:


f = open("error_id.json")
error_id = json.load(f)
f.close()


# In[9]:


london_within_region = list()
for key in london_id_lon_lat:
    point = Point(london_id_lon_lat[key][0], london_id_lon_lat[key][1])
    if point.within(poly) and key not in error_id["error"]:
        london_within_region.append(key)
print(len(london_within_region))
london_within_region.sort()
london_within_region_dict = {london_within_region[i]:i for i in range(len(london_within_region))}
n_london = len(london_within_region_dict)


# # 1.3. fill the data

# In[10]:


n_col = len(list(set(df_traffic_london_4["interval"]))) * 9
flow_table = [[0 for i in range(n_col)] for j in range(len(london_within_region))]
occ_table = [[0 for i in range(n_col)] for j in range(len(london_within_region))]


# In[11]:


region_1 = df_traffic_london_1[df_traffic_london_1["detid"].isin(london_within_region)]
region_2 = df_traffic_london_2[df_traffic_london_2["detid"].isin(london_within_region)]
region_3 = df_traffic_london_3[df_traffic_london_3["detid"].isin(london_within_region)]
region_4 = df_traffic_london_4[df_traffic_london_4["detid"].isin(london_within_region)]
region_5 = df_traffic_london_5[df_traffic_london_5["detid"].isin(london_within_region)]
region_6 = df_traffic_london_6[df_traffic_london_6["detid"].isin(london_within_region)]
region_7 = df_traffic_london_7[df_traffic_london_7["detid"].isin(london_within_region)]
region_8 = df_traffic_london_8[df_traffic_london_8["detid"].isin(london_within_region)]
region_9 = df_traffic_london_9[df_traffic_london_9["detid"].isin(london_within_region)]


# In[12]:


def inject_to_flow_table(region_i, flow_table, occ_table, i): #i=1,2,3,4,5,6,7,8,9
    interval_list = list(region_i["interval"]) #180, 360, ...
    detid_list = list(region_i["detid"])
    occ_list = list(region_i["occ"])
    flow_list = list(region_i["flow"])
    for k in range(len(interval_list)):
        interval, detid, occ, flow = interval_list[k], detid_list[k], occ_list[k], flow_list[k]
        flow_table[london_within_region_dict[detid]][12*24*(i-1)+int(interval/300)] = flow
        occ_table[london_within_region_dict[detid]][12*24*(i-1)+int(interval/300)] = occ
    flow_table = np.array(flow_table)
    occ_table = np.array(occ_table)
    return flow_table, occ_table


# In[13]:


flow_table_1, occ_table_1 = inject_to_flow_table(region_1, flow_table, occ_table, 1)
flow_table_2, occ_table_2 = inject_to_flow_table(region_2, flow_table_1, occ_table_1, 2)
flow_table_3, occ_table_3 = inject_to_flow_table(region_3, flow_table_2, occ_table_2, 3)
flow_table_4, occ_table_4 = inject_to_flow_table(region_4, flow_table_3, occ_table_3, 4)
flow_table_5, occ_table_5 = inject_to_flow_table(region_5, flow_table_4, occ_table_4, 5)
flow_table_6, occ_table_6 = inject_to_flow_table(region_6, flow_table_5, occ_table_5, 6)
flow_table_7, occ_table_7 = inject_to_flow_table(region_7, flow_table_6, occ_table_6, 7)
flow_table_8, occ_table_8 = inject_to_flow_table(region_8, flow_table_7, occ_table_7, 8)
flow_table_9, occ_table_9 = inject_to_flow_table(region_9, flow_table_8, occ_table_8, 9)


# # 2. draw the contour plot

# In[14]:


x = np.linspace(0, 12*24*9-1, 12*24*9)/12
y = np.linspace(1, n_london, n_london)
x_1, y_1 = np.meshgrid(x, y)


# In[15]:


print (np.max(flow_table_9))
flow_table_9_mod = np.where(np.array(flow_table_9) >= 1000.0, 1000.0, np.array(flow_table_9))
print (np.max(flow_table_9_mod))


# In[16]:


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(24,4),dpi=60)
plt.contourf(x_1, y_1, flow_table_9_mod, cmap='jet', vmin=0, vmax=1000, levels=100)
  
cb = plt.colorbar()
tick_locator = ticker.MaxNLocator(nbins=4)
cb.locator = tick_locator
cb.update_ticks()

for t in cb.ax.get_yticklabels():
     t.set_fontsize(20)
        
plt.xlabel("Hour", fontsize=28)
plt.xticks([12*i for i in range(19)],\
           [0,12,24,12,24,12,24,12,24,12,24,12,24,12,24,12,24,12,24], fontsize=28)
plt.yticks([1,38,75,112], [1,38,75,112], fontsize=28)
plt.ylabel("Sensor ID", fontsize=28)
#plt.savefig('flow.png',bbox_inches = 'tight')
plt.show()


# In[17]:


print (np.max(occ_table_9))
occ_table_9_mod = np.where(np.array(occ_table_9) >= 1.0, 1.0, np.array(occ_table_9))
print (np.max(occ_table_9_mod))


# In[18]:


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(24,4),dpi=60)
plt.contourf(x_1, y_1, occ_table_9_mod, cmap = 'jet',  vmin=0, vmax=1.0, levels=100)
cb = plt.colorbar()
tick_locator = ticker.MaxNLocator(nbins=4)
cb.locator = tick_locator
cb.update_ticks() 
for t in cb.ax.get_yticklabels():
     t.set_fontsize(20)
        
plt.xlabel("Hour",fontsize=28)
plt.xticks([12*i for i in range(19)],\
           [0,12,24,12,24,12,24,12,24,12,24,12,24,12,24,12,24,12,24], fontsize=28)
plt.yticks([1,38,75,112], [1,38,75,112], fontsize=28)
plt.ylabel("Sensor ID", fontsize=28)
#plt.savefig('occ.png',bbox_inches = 'tight')
plt.show()


# # 3. check flow and occupancy that are always 0

# In[19]:


average_flow_zero_list = list()
for i in range(n_london):
    mean_flow = np.mean(flow_table_9_mod[i])
    if mean_flow < 0.001:
        average_flow_zero_list.append(i)
        
average_occ_zero_list = list()
for i in range(n_london):
    mean_occ = np.mean(occ_table_9_mod[i])
    if mean_occ < 0.0001:
        average_occ_zero_list.append(i)
        
print (len(average_flow_zero_list))
print (len(average_occ_zero_list))
print (len(set(average_flow_zero_list).intersection(set(average_occ_zero_list))))
print (average_flow_zero_list)


# # 4. check occupancy that is larger than 1

# In[20]:


average_occ_1_list = list()
for i in range(n_london):
    mean_occ = np.mean(occ_table_9[i])
    if mean_occ >= 0.50:
        average_occ_1_list.append(i)
print (average_occ_1_list)


# # 5. check period with the entry equal to 0

# In[21]:


temporal_mean = np.mean(flow_table_9, axis=0)
len(temporal_mean)
n_temporal_mean_zero_flow = list()
for i in range(len(temporal_mean)):
    if temporal_mean[i] < 0.0001:
        n_temporal_mean_zero_flow.append(i)
print (len(n_temporal_mean_zero_flow))


# In[22]:


temporal_mean = np.mean(occ_table_9, axis=0)
len(temporal_mean)
n_temporal_mean_zero_occ = list()
for i in range(len(temporal_mean)):
    if temporal_mean[i] < 0.000001:
        n_temporal_mean_zero_occ.append(i)
print (len(n_temporal_mean_zero_occ))


# In[23]:


print (np.array(n_temporal_mean_zero_occ))
print (np.array(n_temporal_mean_zero_flow))
print (np.array(n_temporal_mean_zero_occ)-np.array([2*12*24]*len(n_temporal_mean_zero_occ)))
print (np.array(n_temporal_mean_zero_flow)-np.array([2*12*24]*len(n_temporal_mean_zero_occ)))


# # 6. deal with the zero entry issue

# # 6.1. sample the sensor and output files

# In[24]:


final_sensor_dict = dict()
for sensor in london_within_region_dict:
    if london_within_region_dict[sensor] <= 87:
        final_sensor_dict[sensor] = london_within_region_dict[sensor]
len_final = len(final_sensor_dict)
savefile = open("sensor_" + str(len_final) +".json",'w')
json.dump(final_sensor_dict, savefile)
savefile.close()


# # 6.2. visualization

# In[25]:


x = np.linspace(0, 12*24*9-1, 12*24*9)/12
y = np.linspace(1,len_final,len_final)
x_1, y_1 = np.meshgrid(x, y)
flow_table_9_final = flow_table_9[0:len_final]
occ_table_9_final = occ_table_9[0:len_final]
print (np.max(flow_table_9_final))
print (np.max(occ_table_9_final))


# In[26]:


print (np.max(flow_table_9_final))
flow_table_9_final_mod = np.where(np.array(flow_table_9_final) >= 1000.0, 1000.0, np.array(flow_table_9_final))
print (np.max(flow_table_9_final_mod))
print ("----------------------------------------")
print (np.max(occ_table_9_final))
occ_table_9_final_mod = np.where(np.array(occ_table_9_final) >= 1.0, 1.0, np.array(occ_table_9_final))
print (np.max(occ_table_9_final_mod))


# In[27]:


large_flow_count = 0
for i in range(len(flow_table_9_final)):
    for j in range(len(flow_table_9_final[0])):
        if flow_table_9_final[i][j] > 1000:
            #print(flow_table_7[i][j])
            large_flow_count = large_flow_count + 1
print (large_flow_count)
print (large_flow_count/len(flow_table_9_final)/len(flow_table_9_final[0])*100)


# In[28]:


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(24,4), dpi=60)
plt.contourf(x_1, y_1, flow_table_9_final_mod, cmap='jet', vmin=0, vmax=1000, levels=100)
  
cb = plt.colorbar()
tick_locator = ticker.MaxNLocator(nbins=4)
cb.locator = tick_locator
cb.update_ticks()

for t in cb.ax.get_yticklabels():
     t.set_fontsize(20)
        
plt.xlabel("Hour", fontsize=28)
plt.xticks([12*i for i in range(19)],\
           [0,12,24,12,24,12,24,12,24,12,24,12,24,12,24,12,24,12,24], fontsize=28)
plt.yticks([1,29,58,88], [1,29,58,88], fontsize=28)
plt.ylabel("Sensor ID", fontsize=28)
#plt.savefig('flow.png',bbox_inches = 'tight')
plt.show()


# In[29]:


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(24,4),dpi=60)
plt.contourf(x_1, y_1, occ_table_9_final_mod, cmap = 'jet',  vmin=0, vmax=1.0, levels=100)
cb = plt.colorbar()
tick_locator = ticker.MaxNLocator(nbins=4)
cb.locator = tick_locator
cb.update_ticks() 
for t in cb.ax.get_yticklabels():
     t.set_fontsize(20)
        
plt.xlabel("Hour",fontsize=28)
plt.xticks([12*i for i in range(19)],\
           [0,12,24,12,24,12,24,12,24,12,24,12,24,12,24,12,24,12,24], fontsize=28)
plt.yticks([1,29,58,88], [1,29,58,88], fontsize=28)
plt.ylabel("Sensor ID", fontsize=28)
#plt.savefig('occ.png',bbox_inches = 'tight')
plt.show()


# # 7. deal with the missing time issue

# In[30]:


print(len(flow_table_9_final), len(flow_table_9_final[0]))
print (len(n_temporal_mean_zero_occ)/len(flow_table_9_final[0]))


# In[31]:


#s is an n-by-m matrix.
#conduct the linear interpolation between the u-th column and v-th column.
#u, v \in {0,1,...,m-1}
def linear_interpolate(s, u, v):
    column_u, column_v, s_new = s[:, u], s[:, v], copy.copy(s)
    for i in range(v-u-1):
        target = u+1+i
        s_new[:,target] = column_u + (column_v - column_u) * (target-u)/(v-u)
    return s_new


# In[32]:


flow_table_9_final_ip = linear_interpolate(flow_table_9_final, 584, 600)
flow_table_9_final_ip = linear_interpolate(flow_table_9_final_ip, 607, 636)
occ_table_9_final_ip = linear_interpolate(occ_table_9_final, 584, 600)
occ_table_9_final_ip = linear_interpolate(occ_table_9_final_ip, 607, 636)


# In[33]:


print (np.max(flow_table_9_final_ip))
flow_table_9_final_ip_mod = np.where(np.array(flow_table_9_final_ip) >= 1000.0, 1000.0, np.array(flow_table_9_final_ip))
print (np.max(flow_table_9_final_ip_mod))
print ("----------------------------------------")
print (np.max(occ_table_9_final_ip))
occ_table_9_final_ip_mod = np.where(np.array(occ_table_9_final_ip) >= 1.0, 1.0, np.array(occ_table_9_final_ip))
print (np.max(occ_table_9_final_ip_mod))


# In[34]:


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(24,4), dpi=300)
plt.contourf(x_1, y_1, flow_table_9_final_ip_mod, cmap='jet', vmin=0, vmax=1000, levels=100)
  
cb = plt.colorbar()
tick_locator = ticker.MaxNLocator(nbins=4)
cb.locator = tick_locator
cb.update_ticks()

for t in cb.ax.get_yticklabels():
     t.set_fontsize(24)
        
plt.xlabel("Hour", fontsize=28)
plt.xticks([12*i for i in range(19)],\
           [0,12,24,12,24,12,24,12,24,12,24,12,24,12,24,12,24,12,24], fontsize=28)
plt.yticks([1,29,58,88], [1,29,58,88], fontsize=28)
plt.ylabel("Sensor ID", fontsize=28)
plt.savefig('london_flow.png',bbox_inches = 'tight')
plt.show()


# In[35]:


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(24,4),dpi=300)
plt.contourf(x_1, y_1, occ_table_9_final_ip_mod, cmap = 'jet',  vmin=0, vmax=1.0, levels=100)
cb = plt.colorbar()
tick_locator = ticker.MaxNLocator(nbins=4)
cb.locator = tick_locator
cb.update_ticks() 
for t in cb.ax.get_yticklabels():
     t.set_fontsize(24)
        
plt.xlabel("Hour",fontsize=28)
plt.xticks([12*i for i in range(19)],\
           [0,12,24,12,24,12,24,12,24,12,24,12,24,12,24,12,24,12,24], fontsize=28)
plt.yticks([1,29,58,88], [1,29,58,88], fontsize=28)
plt.ylabel("Sensor ID", fontsize=28)
plt.savefig('london_occ.png',bbox_inches = 'tight')
plt.show()


# # 8. check overly large flow values

# In[36]:


large_flow_count = 0
for i in range(len(flow_table_9_final_ip)):
    for j in range(len(flow_table_9_final_ip[0])):
        if flow_table_9_final_ip[i][j] > 2500:
            large_flow_count = large_flow_count + 1
print (large_flow_count)
print (large_flow_count/len(flow_table_9_final_ip)/len(flow_table_9_final_ip[0])*100)


# In[37]:


large_occ_count = 0
for i in range(len(occ_table_9_final_ip)):
    for j in range(len(occ_table_9_final_ip[0])):
        if occ_table_9_final_ip[i][j] > 1.00:
            large_occ_count = large_occ_count + 1
print (large_occ_count/len(flow_table_9_final_ip)/len(flow_table_9_final_ip[0])*100)


# # 9. summarize the traffic tensor considering the length of road segments.

# # 9.1. get the road segment length statistics.

# In[38]:


df_detector = pd.read_csv("detectors_public.csv")
print (list(df_detector.columns))
detid_list, length_list = df_detector["detid"], df_detector["length"]
detector_length = dict()
for i in range(len(detid_list)):
    detector_length[detid_list[i]] = length_list[i]


# In[39]:


london_detid_final = dict()
length_list = [0 for i in range(88)]
for detid in london_within_region_dict:
    if london_within_region_dict[detid]<88:
        london_detid_final[detid] = [london_within_region_dict[detid], detector_length[detid]]
        length_list[int(london_within_region_dict[detid])] = detector_length[detid]
#{'CNTR_N01/089g1': [0, 0.827125439207079],'CNTR_N01/089l1': [1, 0.090777186933951],...,\
#'CNTR_N03/164x1': [87, 0.209551441056184]}
length_list = np.array(length_list)


# In[40]:


savefile = open("london_detid_final.json",'w')
json.dump(london_detid_final, savefile)
savefile.close()


# # 9.2. output tensors

# In[41]:


flow_table_9_final_ip_out = np.where(np.array(flow_table_9_final_ip) > 2500.0, 2500.0, np.array(flow_table_9_final_ip))
flow_table_9_final_ip_list = [list(flow_table_9_final_ip_out[i]) for i in range(len(flow_table_9_final_ip_out))]
occ_table_9_final_ip_out = np.where(np.array(occ_table_9_final_ip) > 1.0, 1.0, np.array(occ_table_9_final_ip))
occ_table_9_final_ip_list = [list(occ_table_9_final_ip_out[i]) for i in range(len(occ_table_9_final_ip_out))]


# In[42]:


london_traffic = {"flow": flow_table_9_final_ip_list, "occ": occ_table_9_final_ip_list}
savefile = open("london_flow_traffic_final.json",'w')
json.dump(london_traffic, savefile)
savefile.close()


# # 9.3. draw MFD figures

# In[43]:


def extract_average_flow_occ_on_day(j):  #j=0,1,...,6,7,8
    ave_flow_list, ave_occ_list = list(), list()
    for k in range(12*24):
        col_idx = j*12*24 + k
        flow = flow_table_9_final_ip_out[:,col_idx]
        occ = occ_table_9_final_ip_out[:,col_idx]
        average_flow = np.dot(flow, length_list)/np.sum(length_list)
        average_occ = np.dot(occ, length_list)/np.sum(length_list)
        ave_flow_list.append(average_flow)
        ave_occ_list.append(average_occ)
    return ave_flow_list, ave_occ_list


# # j=0, Fri

# In[80]:


j = 0
ave_flow_list, ave_occ_list = extract_average_flow_occ_on_day(j)
print (np.mean(ave_flow_list))
print (np.mean(ave_occ_list))


# In[81]:


print (np.max(ave_flow_list))
print (np.max(ave_occ_list))


# In[45]:


plt.figure(figsize=(3,2.5),dpi=300)
l1 = plt.scatter(ave_occ_list, ave_flow_list , s=0.2, c= 'r', marker=".", label="Oct. 26, 2015")

plt.xlabel("Average occupancy", fontsize = 14)
plt.ylabel("Average flow \n rate (veh/hour)", fontsize = 14)

my_x_ticks = np.arange(0, 0.46, 0.15)
my_y_ticks = np.arange(0, 610, 150)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks) 

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title("May 15, 2015 (Fri)", fontsize = 14)
#plt.legend(loc=4)
#plt.grid()
#plt.savefig('sample_mfd/tor.pdf',bbox_inches = 'tight')
plt.savefig("mfd_figure/"+str(j)+".svg",bbox_inches = 'tight')
plt.show()


# # j=1, Sat

# In[82]:


j = 1
ave_flow_list, ave_occ_list = extract_average_flow_occ_on_day(j)
print (np.mean(ave_flow_list))
print (np.mean(ave_occ_list))


# In[83]:


print (np.max(ave_flow_list))
print (np.max(ave_occ_list))


# In[47]:


plt.figure(figsize=(3,2.5),dpi=300)
l1 = plt.scatter(ave_occ_list, ave_flow_list , s=0.2, c= 'r', marker=".", label="Oct. 26, 2015")

plt.xlabel("Average occupancy", fontsize = 14)
plt.ylabel("Average flow \n rate (veh/hour)", fontsize = 14)

my_x_ticks = np.arange(0, 0.46, 0.15)
my_y_ticks = np.arange(0, 610, 150)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks) 

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title("May 16, 2015 (Sat)", fontsize = 14)
#plt.legend(loc=4)
#plt.grid()
#plt.savefig('sample_mfd/tor.pdf',bbox_inches = 'tight')
plt.savefig("mfd_figure/"+str(j)+".svg",bbox_inches = 'tight')
plt.show()


# # j=2, Sun

# In[84]:


j = 2
ave_flow_list, ave_occ_list = extract_average_flow_occ_on_day(j)
print (np.mean(ave_flow_list))
print (np.mean(ave_occ_list))


# In[85]:


print (np.max(ave_flow_list))
print (np.max(ave_occ_list))


# In[49]:


plt.figure(figsize=(3,2.5),dpi=300)
l1 = plt.scatter(ave_occ_list, ave_flow_list , s=0.2, c= 'r', marker=".", label="Oct. 26, 2015")

plt.xlabel("Average occupancy", fontsize = 14)
plt.ylabel("Average flow \n rate (veh/hour)", fontsize = 14)

my_x_ticks = np.arange(0, 0.46, 0.15)
my_y_ticks = np.arange(0, 610, 150)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks) 

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title("May 17, 2015 (Sun)", fontsize = 14)
#plt.legend(loc=4)
#plt.grid()
#plt.savefig('sample_mfd/tor.pdf',bbox_inches = 'tight')
plt.savefig("mfd_figure/"+str(j)+".svg",bbox_inches = 'tight')
plt.show()


# # j=3, Mon

# In[86]:


j = 3
ave_flow_list, ave_occ_list = extract_average_flow_occ_on_day(j)
print (np.mean(ave_flow_list))
print (np.mean(ave_occ_list))


# In[87]:


print (np.max(ave_flow_list))
print (np.max(ave_occ_list))


# In[51]:


plt.figure(figsize=(3,2.5),dpi=300)
l1 = plt.scatter(ave_occ_list, ave_flow_list , s=0.2, c= 'r', marker=".", label="Oct. 26, 2015")

plt.xlabel("Average occupancy", fontsize = 14)
plt.ylabel("Average flow \n rate (veh/hour)", fontsize = 14)

my_x_ticks = np.arange(0, 0.46, 0.15)
my_y_ticks = np.arange(0, 610, 150)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks) 

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title("May 18, 2015 (Mon)", fontsize = 14)
#plt.legend(loc=4)
#plt.grid()
#plt.savefig('sample_mfd/tor.pdf',bbox_inches = 'tight')
plt.savefig("mfd_figure/"+str(j)+".svg",bbox_inches = 'tight')
plt.show()


# # j=4, Tue

# In[88]:


j = 4
ave_flow_list, ave_occ_list = extract_average_flow_occ_on_day(j)
print (np.mean(ave_flow_list))
print (np.mean(ave_occ_list))


# In[89]:


print (np.max(ave_flow_list))
print (np.max(ave_occ_list))


# In[53]:


plt.figure(figsize=(3,2.5),dpi=300)
l1 = plt.scatter(ave_occ_list, ave_flow_list , s=0.2, c= 'r', marker=".", label="Oct. 26, 2015")

plt.xlabel("Average occupancy", fontsize = 14)
plt.ylabel("Average flow \n rate (veh/hour)", fontsize = 14)

my_x_ticks = np.arange(0, 0.46, 0.15)
my_y_ticks = np.arange(0, 610, 150)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks) 

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title("May 19, 2015 (Tue)", fontsize = 14)
#plt.legend(loc=4)
#plt.grid()
#plt.savefig('sample_mfd/tor.pdf',bbox_inches = 'tight')
plt.savefig("mfd_figure/"+str(j)+".svg",bbox_inches = 'tight')
plt.show()


# # j=5, Wed

# In[90]:


j = 5
ave_flow_list, ave_occ_list = extract_average_flow_occ_on_day(j)
print (np.mean(ave_flow_list))
print (np.mean(ave_occ_list))


# In[91]:


print (np.max(ave_flow_list))
print (np.max(ave_occ_list))


# In[55]:


plt.figure(figsize=(3,2.5),dpi=300)
l1 = plt.scatter(ave_occ_list, ave_flow_list , s=0.2, c= 'r', marker=".", label="Oct. 26, 2015")

plt.xlabel("Average occupancy", fontsize = 14)
plt.ylabel("Average flow \n rate (veh/hour)", fontsize = 14)

my_x_ticks = np.arange(0, 0.46, 0.15)
my_y_ticks = np.arange(0, 610, 150)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks) 

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title("May 20, 2015 (Wed)", fontsize = 14)
#plt.legend(loc=4)
#plt.grid()
#plt.savefig('sample_mfd/tor.pdf',bbox_inches = 'tight')
plt.savefig("mfd_figure/"+str(j)+".svg",bbox_inches = 'tight')
plt.show()


# # j=6, Thu

# In[92]:


j = 6
ave_flow_list, ave_occ_list = extract_average_flow_occ_on_day(j)
print (np.mean(ave_flow_list))
print (np.mean(ave_occ_list))


# In[93]:


print (np.max(ave_flow_list))
print (np.max(ave_occ_list))


# In[57]:


plt.figure(figsize=(3,2.5),dpi=300)
l1 = plt.scatter(ave_occ_list, ave_flow_list , s=0.2, c= 'r', marker=".", label="Oct. 26, 2015")

plt.xlabel("Average occupancy", fontsize = 14)
plt.ylabel("Average flow \n rate (veh/hour)", fontsize = 14)

my_x_ticks = np.arange(0, 0.46, 0.15)
my_y_ticks = np.arange(0, 610, 150)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks) 

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title("May 21, 2015 (Thu)", fontsize = 14)
#plt.legend(loc=4)
#plt.grid()
#plt.savefig('sample_mfd/tor.pdf',bbox_inches = 'tight')
plt.savefig("mfd_figure/"+str(j)+".svg",bbox_inches = 'tight')
plt.show()


# # j=7, Fri

# In[94]:


j = 7
ave_flow_list, ave_occ_list = extract_average_flow_occ_on_day(j)
print (np.mean(ave_flow_list))
print (np.mean(ave_occ_list))


# In[95]:


print (np.max(ave_flow_list))
print (np.max(ave_occ_list))


# In[59]:


plt.figure(figsize=(3,2.5),dpi=300)
l1 = plt.scatter(ave_occ_list, ave_flow_list , s=0.2, c= 'r', marker=".", label="Oct. 26, 2015")

plt.xlabel("Average occupancy", fontsize = 14)
plt.ylabel("Average flow \n rate (veh/hour)", fontsize = 14)

my_x_ticks = np.arange(0, 0.46, 0.15)
my_y_ticks = np.arange(0, 610, 150)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks) 

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title("May 22, 2015 (Fri)", fontsize = 14)
#plt.legend(loc=4)
#plt.grid()
#plt.savefig('sample_mfd/tor.pdf',bbox_inches = 'tight')
plt.savefig("mfd_figure/"+str(j)+".svg",bbox_inches = 'tight')
plt.show()


# # j=8, Sat

# In[96]:


j = 8
ave_flow_list, ave_occ_list = extract_average_flow_occ_on_day(j)
print (np.mean(ave_flow_list))
print (np.mean(ave_occ_list))


# In[97]:


print (np.max(ave_flow_list))
print (np.max(ave_occ_list))


# In[61]:


plt.figure(figsize=(3,2.5),dpi=300)
l1 = plt.scatter(ave_occ_list, ave_flow_list , s=0.2, c= 'r', marker=".", label="Oct. 26, 2015")

plt.xlabel("Average occupancy", fontsize = 14)
plt.ylabel("Average flow \n rate (veh/hour)", fontsize = 14)

my_x_ticks = np.arange(0, 0.46, 0.15)
my_y_ticks = np.arange(0, 610, 150)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks) 

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title("May 23, 2015 (Sat)", fontsize = 14)
#plt.legend(loc=4)
#plt.grid()
#plt.savefig('sample_mfd/tor.pdf',bbox_inches = 'tight')
plt.savefig("mfd_figure/"+str(j)+".svg",bbox_inches = 'tight')
plt.show()


# In[ ]:




