#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import json
import numpy as np
import pandas as pd
from matplotlib import ticker
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon


# # 1. get the contour tensor

# # 1.1. read the data

# In[2]:


df_traffic_zurich = pd.read_csv("utd19_u_zurich.csv")
print (set(list(df_traffic_zurich["day"])))


# In[3]:


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


# # 1.2. idx in the region

# In[4]:


p1, p2, p3 = [47.370478, 8.524712], [47.366947, 8.518524], [47.377578, 8.506667]
p4, p5, p6 = [47.382472, 8.515091], [47.380802, 8.528440], [47.377377, 8.533421]
poly = Polygon([(p1[1], p1[0]), (p2[1], p2[0]), (p3[1], p3[0]),\
                (p4[1], p4[0]), (p5[1], p5[0]), (p6[1], p6[0])])


# In[5]:


#find the ID of 60 loops within this region.
df_detector = pd.read_csv("/home/umni2/a/umnilab/users/xue120/umni4/2022_ce/course_project/detectors_public.csv")
df_detector_Zurich = df_detector[df_detector["citycode"]=="zurich"] 
print ("# Zurich row", len(df_detector_Zurich))


# In[6]:


id_list = list(df_detector_Zurich["detid"])
lon_list = list(df_detector_Zurich["long"])
lat_list = list(df_detector_Zurich["lat"])
Zurich_id_lon_lat = {str(id_list[i]):[lon_list[i], lat_list[i]] for i in range(len(id_list))}


# In[7]:


f = open("error_id.json")
error_id = json.load(f)
f.close()


# In[8]:


zurich_within_region = list()
for key in Zurich_id_lon_lat:
    point = Point(Zurich_id_lon_lat[key][0], Zurich_id_lon_lat[key][1])
    if point.within(poly) and key not in error_id["error"]:
        zurich_within_region.append(key)
print(len(zurich_within_region))
zurich_within_region.sort()
zurich_within_region_dict = {zurich_within_region[i]:i for i in range(len(zurich_within_region))}


# In[9]:


n_zurich = len(zurich_within_region_dict)


# In[10]:


#zurich_within_region_dict
#{{'K10D11': 0, 'K10D12': 1, ..., 'K8D20': 119}


# In[11]:


print (np.min(list(df_traffic_zurich_1["interval"])))
print (np.max(list(df_traffic_zurich_1["interval"])))
print (np.min(list(df_traffic_zurich_7["interval"])))
print (np.max(list(df_traffic_zurich_7["interval"])))


# # 1.3. fill the data

# In[12]:


flow_table = [[0 for i in range(3360)] for j in range(len(zurich_within_region))]
occ_table = [[0 for i in range(3360)] for j in range(len(zurich_within_region))]


# In[13]:


region_1 = df_traffic_zurich_1[df_traffic_zurich_1["detid"].isin(zurich_within_region)]
region_2 = df_traffic_zurich_2[df_traffic_zurich_2["detid"].isin(zurich_within_region)]
region_3 = df_traffic_zurich_3[df_traffic_zurich_3["detid"].isin(zurich_within_region)]
region_4 = df_traffic_zurich_4[df_traffic_zurich_4["detid"].isin(zurich_within_region)]
region_5 = df_traffic_zurich_5[df_traffic_zurich_5["detid"].isin(zurich_within_region)]
region_6 = df_traffic_zurich_6[df_traffic_zurich_6["detid"].isin(zurich_within_region)]
region_7 = df_traffic_zurich_7[df_traffic_zurich_7["detid"].isin(zurich_within_region)]


# In[14]:


def inject_to_flow_table(region_i, flow_table, occ_table, i): #i=1,2,3,4,5,6,7
    interval_list = list(region_i["interval"]) #180, 360, ...
    detid_list = list(region_i["detid"])
    occ_list = list(region_i["occ"])
    flow_list = list(region_i["flow"])
    for k in range(len(interval_list)):
        interval, detid, occ, flow = interval_list[k], detid_list[k], occ_list[k], flow_list[k]
        flow_table[zurich_within_region_dict[detid]][20*24*(i-1)+int(interval/180)] = flow
        occ_table[zurich_within_region_dict[detid]][20*24*(i-1)+int(interval/180)] = occ
    return flow_table, occ_table


# In[15]:


flow_table_1, occ_table_1 = inject_to_flow_table(region_1, flow_table, occ_table, 1)
flow_table_2, occ_table_2 = inject_to_flow_table(region_2, flow_table_1, occ_table_1, 2)
flow_table_3, occ_table_3 = inject_to_flow_table(region_3, flow_table_2, occ_table_2, 3)
flow_table_4, occ_table_4 = inject_to_flow_table(region_4, flow_table_3, occ_table_3, 4)
flow_table_5, occ_table_5 = inject_to_flow_table(region_5, flow_table_4, occ_table_4, 5)
flow_table_6, occ_table_6 = inject_to_flow_table(region_6, flow_table_5, occ_table_5, 6)
flow_table_7, occ_table_7 = inject_to_flow_table(region_7, flow_table_6, occ_table_6, 7)


# In[16]:


np.max(occ_table_7)


# # 2. draw the contour plot

# In[17]:


x = np.linspace(0, 20*24*7-1, 20*24*7)/20
y = np.linspace(1, n_zurich, n_zurich)
x_1, y_1 = np.meshgrid(x, y)


# In[18]:


n_zurich


# In[19]:


print (np.max(flow_table_7))
print (np.max(occ_table_7))


# In[20]:


flow_table_7_mod = np.where(np.array(flow_table_7) >= 1000.0, 1000.0, np.array(flow_table_7))
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(20,4),dpi=300)
plt.contourf(x_1, y_1, flow_table_7_mod, cmap='jet', vmin=0, vmax=1000, levels=100)
  
cb = plt.colorbar()
tick_locator = ticker.MaxNLocator(nbins=4)
cb.locator = tick_locator
cb.update_ticks()

for t in cb.ax.get_yticklabels():
     t.set_fontsize(24)
        
plt.xlabel("Hour", fontsize=28)
plt.xticks([12*i for i in range(15)],\
           [0,12,24,12,24,12,24,12,24,12,24,12,24,12,24], fontsize=28)
plt.yticks([1,35,70,106],\
           [1,35,70,106], fontsize=28)
plt.ylabel("Sensor ID", fontsize=28)
plt.savefig('zurich_flow.png',bbox_inches = 'tight')
plt.show()


# In[21]:


occ_table_7_mod = np.where(np.array(occ_table_7) >= 1.0, 1.0, np.array(occ_table_7))


# In[22]:


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(20,4),dpi=300)
plt.contourf(x_1, y_1, occ_table_7_mod, cmap = 'jet',  vmin=0, vmax=1.0, levels=100)
cb = plt.colorbar()
tick_locator = ticker.MaxNLocator(nbins=4)
cb.locator = tick_locator
cb.update_ticks() 
for t in cb.ax.get_yticklabels():
     t.set_fontsize(24)
        
plt.xlabel("Hour",fontsize=28)
plt.xticks([12*i for i in range(15)],\
           [0,12,24,12,24,12,24,12,24,12,24,12,24,12,24], fontsize=28)
plt.yticks([1,35,70,106], [1,35,70,106], fontsize=28)
plt.ylabel("Sensor ID", fontsize=28)
plt.savefig('zurich_occ.png',bbox_inches = 'tight')
plt.show()


# In[23]:


np.max(occ_table_7)


# # 3. check flow and occupancy that are always 0

# In[24]:


average_flow_zero_list = list()
for i in range(n_zurich):
    mean_flow = np.mean(flow_table_7_mod[i])
    if mean_flow < 0.001:
        average_flow_zero_list.append(i)
        
average_occ_zero_list = list()
for i in range(n_zurich):
    mean_occ = np.mean(occ_table_7_mod[i])
    if mean_occ < 0.0001:
        average_occ_zero_list.append(i)
        
print (len(average_flow_zero_list))
print (len(average_occ_zero_list))
print (len(set(average_flow_zero_list).intersection(set(average_occ_zero_list))))


# # 4. check occupancy that is larger than 1.

# In[25]:


average_occ_1_list = list()
for i in range(n_zurich):
    mean_occ = np.mean(occ_table_7_mod[i])
    if mean_occ >= 0.50:
        average_occ_1_list.append(i)


# In[26]:


average_occ_1_list


# # 5. check period with the entry equal to 0.

# In[27]:


temporal_mean = np.mean(flow_table_7_mod, axis=0)
len(temporal_mean)
n_temporal_mean_zero_flow = list()
for i in range(len(temporal_mean)):
    if temporal_mean[i] < 0.0001:
        n_temporal_mean_zero_flow.append(i)
print (len(n_temporal_mean_zero_flow))


# In[28]:


temporal_mean = np.mean(occ_table_7_mod, axis=0)
len(temporal_mean)
n_temporal_mean_zero_occ = list()
for i in range(len(temporal_mean)):
    if temporal_mean[i] < 0.000001:
        n_temporal_mean_zero_occ.append(i)
print (len(n_temporal_mean_zero_occ))


# In[29]:


print (n_temporal_mean_zero_occ)
print (n_temporal_mean_zero_flow)


# # 6. check too large entries

# In[30]:


large_flow_count = 0
for i in range(len(flow_table_7)):
    for j in range(len(flow_table_7[0])):
        if flow_table_7[i][j] > 2500:
            print(flow_table_7[i][j])
            large_flow_count = large_flow_count + 1
print (large_flow_count)
print (large_flow_count/len(flow_table_7)/len(flow_table_7[0])*100)


# In[31]:


large_flow_count = 0
for i in range(len(flow_table_7)):
    for j in range(len(flow_table_7[0])):
        if flow_table_7[i][j] > 1000:
            #print(flow_table_7[i][j])
            large_flow_count = large_flow_count + 1
print (large_flow_count)
print (large_flow_count/len(flow_table_7)/len(flow_table_7[0])*100)


# In[32]:


large_occ_count = 0
for i in range(len(occ_table_7)):
    for j in range(len(occ_table_7[0])):
        if occ_table_7[i][j] > 1.0:
            #print(occ_table_7[i][j])
            large_occ_count = large_occ_count + 1
print (large_occ_count)
print (large_occ_count/len(occ_table_7)/len(occ_table_7[0])*100.0)


# # 7. summarize the traffic tensor considering the length of road segments.

# # 7.1. get the road segment length statistics.

# In[33]:


df_detector = pd.read_csv("detectors_public.csv")
print (list(df_detector.columns))
detid_list, length_list = df_detector["detid"], df_detector["length"]
detector_length = dict()
for i in range(len(detid_list)):
    detector_length[detid_list[i]] = length_list[i]


# In[34]:


zurich_detid_final = dict()
length_list = [0 for i in range(len(zurich_within_region_dict))]
for detid in zurich_within_region_dict:
    zurich_detid_final[detid] = [zurich_within_region_dict[detid], detector_length[detid]]
    length_list[int(zurich_within_region_dict[detid])] = detector_length[detid]
#{'K10D11': [0, 0.073527516921304], 'K10D12': [1, 0.072993633246372],...,
#'K8D19': [104, 0.065080639633257],'K8D20': [105, 0.235968359599021]}
length_list = np.array(length_list)


# In[35]:


savefile = open("zurich_detid_final.json",'w')
json.dump(zurich_detid_final, savefile)
savefile.close()


# # 7.2. output tensors

# In[36]:


flow_table_7_final = np.where(np.array(flow_table_7) > 2500.0, 2500.0, np.array(flow_table_7))
flow_table_7_final_list = [list(flow_table_7_final[i]) for i in range(len(flow_table_7_final))]
occ_table_7_final = np.where(np.array(occ_table_7) > 1.0, 1.0, np.array(occ_table_7))
occ_table_7_final_list = [list(occ_table_7_final[i]) for i in range(len(occ_table_7_final))]


# In[37]:


zurich_traffic = {"flow": flow_table_7_final_list, "occ": occ_table_7_final_list}
savefile = open("zurich_flow_traffic_final.json",'w')
json.dump(zurich_traffic, savefile)
savefile.close()


# # 7.3. draw MFD figures

# In[38]:


def extract_average_flow_occ_on_day(j):  #j=0,1,...,6
    ave_flow_list, ave_occ_list = list(), list()
    for k in range(20*24):
        col_idx = j*20*24 + k
        flow = flow_table_7_final[:,col_idx]
        occ = occ_table_7_final[:,col_idx]
        average_flow = np.dot(flow, length_list)/np.sum(length_list)
        average_occ = np.dot(occ, length_list)/np.sum(length_list)
        ave_flow_list.append(average_flow)
        ave_occ_list.append(average_occ)
    return ave_flow_list, ave_occ_list


# # Monday

# In[107]:


j = 0


# In[108]:


ave_flow_list, ave_occ_list = extract_average_flow_occ_on_day(j)
print (np.mean(ave_flow_list))
print (np.mean(ave_occ_list))
print (np.max(ave_flow_list))
print (np.max(ave_occ_list))


# In[68]:


plt.figure(figsize=(3,2.5),dpi=300)
l1 = plt.scatter(ave_occ_list, ave_flow_list , s=0.2, c= 'r', marker=".", label="Oct. 26, 2015")

plt.xlabel("Average occupancy", fontsize = 14)
plt.ylabel("Average flow \n rate (veh/hour)", fontsize = 14)

my_x_ticks = np.arange(0, 0.46, 0.15)
my_y_ticks = np.arange(0, 410, 100)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks) 

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title("Oct. 26, 2015 (Mon)", fontsize = 14)
#plt.legend(loc=4)
#plt.grid()
#plt.savefig('sample_mfd/tor.pdf',bbox_inches = 'tight')
plt.savefig("mfd_figure/"+str(j)+".svg",bbox_inches = 'tight')
plt.show()


# # Tuesday

# In[109]:


j = 1


# In[110]:


ave_flow_list, ave_occ_list = extract_average_flow_occ_on_day(j)
print (np.mean(ave_flow_list))
print (np.mean(ave_occ_list))


# In[111]:


print (np.max(ave_flow_list))
print (np.max(ave_occ_list))


# In[44]:


plt.figure(figsize=(3,2.5),dpi=300)
l1 = plt.scatter(ave_occ_list, ave_flow_list , s=0.2, c= 'r', marker=".", label="Oct. 26, 2015")

plt.xlabel("Average occupancy", fontsize = 14)
plt.ylabel("Average flow \n rate (veh/hour)", fontsize = 14)

my_x_ticks = np.arange(0, 0.46, 0.15)
my_y_ticks = np.arange(0, 410, 100)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks) 

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title("Oct. 27, 2015 (Tue)", fontsize = 14)
#plt.legend(loc=4)
#plt.grid()
#plt.savefig('sample_mfd/tor.pdf',bbox_inches = 'tight')
plt.savefig("mfd_figure/"+str(j)+".svg",bbox_inches = 'tight')
plt.show()


# # Wednesday

# In[112]:


j = 2


# In[113]:


ave_flow_list, ave_occ_list = extract_average_flow_occ_on_day(j)
print (np.mean(ave_flow_list))
print (np.mean(ave_occ_list))


# In[114]:


print (np.max(ave_flow_list))
print (np.max(ave_occ_list))


# In[47]:


plt.figure(figsize=(3,2.5),dpi=300)
l1 = plt.scatter(ave_occ_list, ave_flow_list , s=0.2, c= 'r', marker=".", label="Oct. 26, 2015")

plt.xlabel("Average occupancy", fontsize = 14)
plt.ylabel("Average flow \n rate (veh/hour)", fontsize = 14)

my_x_ticks = np.arange(0, 0.46, 0.15)
my_y_ticks = np.arange(0, 410, 100)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks) 

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title("Oct. 28, 2015 (Wed)", fontsize = 14)
#plt.legend(loc=4)
#plt.grid()
#plt.savefig('sample_mfd/tor.pdf',bbox_inches = 'tight')
plt.savefig("mfd_figure/"+str(j)+".svg",bbox_inches = 'tight')
plt.show()


# # Thu

# In[115]:


j = 3


# In[116]:


ave_flow_list, ave_occ_list = extract_average_flow_occ_on_day(j)
print (np.mean(ave_flow_list))
print (np.mean(ave_occ_list))


# In[117]:


print (np.max(ave_flow_list))
print (np.max(ave_occ_list))


# In[50]:


plt.figure(figsize=(3,2.5),dpi=300)
l1 = plt.scatter(ave_occ_list, ave_flow_list , s=0.2, c= 'r', marker=".", label="Oct. 26, 2015")

plt.xlabel("Average occupancy", fontsize = 14)
plt.ylabel("Average flow \n rate (veh/hour)", fontsize = 14)

my_x_ticks = np.arange(0, 0.46, 0.15)
my_y_ticks = np.arange(0, 410, 100)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks) 

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title("Oct. 29, 2015 (Thu)", fontsize = 14)
#plt.legend(loc=4)
#plt.grid()
#plt.savefig('sample_mfd/tor.pdf',bbox_inches = 'tight')
plt.savefig("mfd_figure/"+str(j)+".svg",bbox_inches = 'tight')
plt.show()


# # Fri

# In[118]:


j = 4


# In[119]:


ave_flow_list, ave_occ_list = extract_average_flow_occ_on_day(j)
print (np.mean(ave_flow_list))
print (np.mean(ave_occ_list))


# In[120]:


print (np.max(ave_flow_list))
print (np.max(ave_occ_list))


# In[53]:


plt.figure(figsize=(3,2.5),dpi=300)
l1 = plt.scatter(ave_occ_list, ave_flow_list , s=0.2, c= 'r', marker=".", label="Oct. 26, 2015")

plt.xlabel("Average occupancy", fontsize = 14)
plt.ylabel("Average flow \n rate (veh/hour)", fontsize = 14)

my_x_ticks = np.arange(0, 0.46, 0.15)
my_y_ticks = np.arange(0, 410, 100)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks) 

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title("Oct. 30, 2015 (Fri)", fontsize = 14)
#plt.legend(loc=4)
#plt.grid()
#plt.savefig('sample_mfd/tor.pdf',bbox_inches = 'tight')
plt.savefig("mfd_figure/"+str(j)+".svg",bbox_inches = 'tight')
plt.show()


# # Sat

# In[121]:


j = 5


# In[122]:


ave_flow_list, ave_occ_list = extract_average_flow_occ_on_day(j)
print (np.mean(ave_flow_list))
print (np.mean(ave_occ_list))


# In[123]:


print (np.max(ave_flow_list))
print (np.max(ave_occ_list))


# In[56]:


plt.figure(figsize=(3,2.5),dpi=300)
l1 = plt.scatter(ave_occ_list, ave_flow_list , s=0.2, c= 'r', marker=".", label="Oct. 26, 2015")

plt.xlabel("Average occupancy", fontsize = 14)
plt.ylabel("Average flow \n rate (veh/hour)", fontsize = 14)

my_x_ticks = np.arange(0, 0.46, 0.15)
my_y_ticks = np.arange(0, 410, 100)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks) 

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title("Oct. 31, 2015 (Sat)", fontsize = 14)
#plt.legend(loc=4)
#plt.grid()
#plt.savefig('sample_mfd/tor.pdf',bbox_inches = 'tight')
plt.savefig("mfd_figure/"+str(j)+".svg",bbox_inches = 'tight')
plt.show()


# # Sun

# In[124]:


j = 6


# In[125]:


ave_flow_list, ave_occ_list = extract_average_flow_occ_on_day(j)
print (np.mean(ave_flow_list))
print (np.mean(ave_occ_list))


# In[126]:


print (np.max(ave_flow_list))
print (np.max(ave_occ_list))


# In[59]:


plt.figure(figsize=(3,2.5),dpi=300)
l1 = plt.scatter(ave_occ_list, ave_flow_list , s=0.2, c= 'r', marker=".", label="Oct. 26, 2015")

plt.xlabel("Average occupancy", fontsize = 14)
plt.ylabel("Average flow \n rate (veh/hour)", fontsize = 14)

my_x_ticks = np.arange(0, 0.46, 0.15)
my_y_ticks = np.arange(0, 410, 100)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks) 

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.title("Nov. 1, 2015 (Sun)", fontsize = 14)
#plt.legend(loc=4)
#plt.grid()
#plt.savefig('sample_mfd/tor.pdf',bbox_inches = 'tight')
plt.savefig("mfd_figure/"+str(j)+".svg",bbox_inches = 'tight')
plt.show()


# In[ ]:




