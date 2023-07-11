#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import json
import random
import numpy as np
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
from shapely.geometry import Point, LineString, Polygon


# In[2]:


# This code prepares features and labels for the traffic imputation task.
# step 0. set hyperparameters.
# step 1. read traffic and detector files.
# step 2. prepare adjacency matrix.
# step 3. prepare spatiotemporal data.
# step 4. prepare train, validation, and test.
# step 5. save.

# Output:
# 1. sensor_id_order.json.
# 2. sensor_adj.json.
# 3. train.json.
# 4. validation.json.
# 5. test.json.


# # 0: set hyperparameters

# In[3]:


train_ratio, vali_ratio = 0.60, 0.20
M = 10                       #number of time interval in a window
missing_ratio = 0.50
file_name = "m_" + str(M) + "_missing_" + str(int(missing_ratio*100))
print (file_name)


# # 1: read traffic and detector files

# In[4]:


city_name = "zurich"
root_path = "/home/umni2/a/umnilab/users/xue120/umni4/2023_mfd_traffic/"

#1.1: define traffic data path
#flow: 106, 3360; #occ: 106, 3360
df_traffic = json.load(open(root_path + "1_check_data/" + "zurich_flow_traffic_final.json"))

#1.2: read detector data
#[detid, length, pos, fclass, road, limit, citycode, lanes, linkid, long, lat]
#[K2D11, 0.377599, 0.015064, primary, Seebahnstrasse, 50, zurich, 1.0, 1128.0, 8.518458, 47.375536]
detector_path = root_path + "1_check_data/detectors_public.csv"
df_detector = pd.read_csv(detector_path)
df_detector_city = df_detector[df_detector["citycode"]==city_name] 
print ("# detector", len(df_detector_city))

#1.3: read selected detector data
#{K10D11: [0, 0.0735],...}
df_detector_selected = json.load(open(root_path + "1_check_data/" + "zurich_detid_final.json"))


# # 2: prepare adjacency matrix

# In[5]:


#2.1: compute the distance between two points
def compute_distance(loc1, loc2):    #loc1:[lon1,lat1]; loc2:[lon2,lat2]
    R = 6373.0                  #km
    lon1, lat1 = radians(loc1[0]), radians(loc1[1])
    lon2, lat2 = radians(loc2[0]), radians(loc2[1])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
    d = 2*R*atan2(sqrt(a), sqrt(1-a))
    return d

#2.2: extract all detectors in the city
id_list = list(df_detector_city["detid"])
lon_list, lat_list = list(df_detector_city["long"]), list(df_detector_city["lat"])
city_id_lon_lat = {str(id_list[i]):[lon_list[i], lat_list[i]] for i in range(len(id_list))}

#2.3: extract all sensors within the polygon
df_detector = pd.read_csv(root_path + "1_check_data/detectors_public.csv")
detid_list, length_list = df_detector["detid"], df_detector["length"]
detector_length = dict()
for i in range(len(detid_list)):
    detector_length[detid_list[i]] = length_list[i]
    
sensor_dict = dict()
for detid in df_detector_selected:
    sensor_dict[detid] = [df_detector_selected[detid][0], round(city_id_lon_lat[detid][0],8),\
                         round(city_id_lon_lat[detid][1],8), detector_length[detid]]
n_sensor = len(sensor_dict)
print (n_sensor)

#2.4: construct the adjacency matrix
adj = [[0.0 for i in range(n_sensor)] for j in range(n_sensor)]
for detid_1 in sensor_dict:
    loc_1, idx_1 = [sensor_dict[detid_1][1], sensor_dict[detid_1][2]], sensor_dict[detid_1][0]
    for detid_2 in sensor_dict:
        loc_2, idx_2 = [sensor_dict[detid_2][1], sensor_dict[detid_2][2]], sensor_dict[detid_2][0]
        adj[idx_1][idx_2] = compute_distance(loc_1, loc_2)
print (np.mean(adj), np.max(adj), np.min(adj))
adj_dict = {"adj": adj}


# # 3: prepare spatiotemporal data

# In[6]:


flow_table_7, occ_table_7 = df_traffic["flow"], df_traffic["occ"]
print (len(flow_table_7), len(flow_table_7[0]))
print (np.max(flow_table_7), np.min(flow_table_7), np.mean(flow_table_7))
print (len(occ_table_7), len(occ_table_7[0]))
print (np.max(occ_table_7), np.min(occ_table_7), np.mean(occ_table_7))


# In[7]:


#check the outliers in the table
def check_ratio_outlier(input_table, threshold, sign):
    count = 0
    n_row, n_column = len(input_table), len(input_table[0])
    for i in range(n_row):
        for j in range(n_column):
            if sign == "1" and input_table[i][j] > threshold:
                    count = count + 1
            if sign == "-1" and input_table[i][j] < threshold:
                    count = count + 1
    return (count*1.0/n_row/n_column)
    
print (check_ratio_outlier(flow_table_7, 1000, "1"))
print (check_ratio_outlier(flow_table_7, 500, "1"))
print (check_ratio_outlier(flow_table_7, 0.10, "-1"))
print ("--------------------")
print (check_ratio_outlier(occ_table_7, 1, "1"))
print (check_ratio_outlier(occ_table_7, 0.50, "1"))
print (check_ratio_outlier(occ_table_7, 0.001, "-1"))


# # 4: prepare train, validation, and test

# In[8]:


def mask_matrix(m, mask_ratio):
    n_row, n_col = len(m), len(m[0])
    mask = random.sample([i for i in range(n_row * n_col)], int(n_row * n_col * mask_ratio))
    mask_matrix = [[0 for j in range(n_col)] for i in range(n_row)]
    for entry in mask:
        row_idx = int(entry/n_col)
        col_idx = entry - row_idx * n_col
        mask_matrix[row_idx][col_idx] = 1
    mask_m = np.array(m)*np.array(mask_matrix)
    return (mask_m)


# In[9]:


def generate_train_vali_test(flow_acc_table, train_ratio, vali_ratio, M, missing_ratio):
    mask_ratio = 1.0 - missing_ratio
    #generate full train, vali, test
    n_row, n_col = len(flow_acc_table), len(flow_acc_table[0])
    n_col_train, n_col_vali = int(n_col*train_ratio), int(n_col*vali_ratio)
    n_col_test = n_col - n_col_train - n_col_vali
    flow_acc_table = np.array(flow_acc_table)
    full_train = flow_acc_table[:, 0: n_col_train]
    full_vali = flow_acc_table[:, n_col_train : n_col_train+n_col_vali]
    full_test = flow_acc_table[:, n_col_train+n_col_vali :]
    
    train, vali, test = list(), list(), list()
    train_m, vali_m, test_m = list(), list(), list()  #m: masked
    print ("# train", n_col_train-M+1)
    for j in range(n_col_train-M+1):
        train_sample = full_train[:,j:j+M].tolist()
        train.append(train_sample)
        m_train_sample = mask_matrix(train_sample, mask_ratio).tolist()
        train_m.append(m_train_sample)
    
    print ("# vali", n_col_vali-M+1)
    for j in range(n_col_vali-M+1):
        vali_sample = full_vali[:,j:j+M].tolist()
        vali.append(vali_sample)
        m_vali_sample = mask_matrix(vali_sample, mask_ratio).tolist()
        vali_m.append(m_vali_sample)
    
    print ("# test", n_col_test-M+1)
    for j in range(n_col_test-M+1):
        test_sample = full_test[:,j:j+M].tolist()
        test.append(test_sample)
        m_test_sample = mask_matrix(test_sample, mask_ratio).tolist()
        test_m.append(m_test_sample)
    
    return train, train_m, vali, vali_m, test, test_m


# In[10]:


f_train, f_train_m, f_vali, f_vali_m, f_test, f_test_m = generate_train_vali_test(flow_table_7, train_ratio, vali_ratio, M, missing_ratio)
o_train, o_train_m, o_vali, o_vali_m, o_test, o_test_m = generate_train_vali_test(occ_table_7, train_ratio, vali_ratio, M, missing_ratio)


# In[11]:


print (np.mean(f_train), np.mean(f_train_m))
print (np.mean(f_vali), np.mean(f_vali_m))
print (np.mean(f_test), np.mean(f_test_m))
print ("-------------------------------------------------")
print (np.mean(o_train), np.mean(o_train_m))
print (np.mean(o_vali), np.mean(o_vali_m))
print (np.mean(o_test), np.mean(o_test_m))


# In[12]:


f_mean = np.mean(f_train)
f_std = np.std(f_train)
o_mean = np.mean(o_train)
o_std = np.std(o_train)
print (f_mean, f_std, o_mean, o_std)
mean_std = {"f_mean":f_mean, "f_std":f_std, "o_mean":o_mean, "o_std":o_std}


# # 5: save

# In[13]:


order_file = open(file_name + "/" + "sensor_id_order" + ".json",'w')
json.dump(sensor_dict, order_file)
order_file.close()

adj_file = open(file_name + "/" + "sensor_adj" + ".json",'w')
json.dump(adj_dict, adj_file)
adj_file.close()

train_dict = {"flow":f_train, "flow_mask":f_train_m, "occupancy":o_train, "occupancy_mask":o_train_m}
train_file = open(file_name + "/" + "train" + ".json",'w')
json.dump(train_dict, train_file)
train_file.close()

vali_dict = {"flow":f_vali, "flow_mask":f_vali_m, "occupancy":o_vali, "occupancy_mask":o_vali_m}
vali_file = open(file_name + "/" + "vali" + ".json",'w')
json.dump(vali_dict, vali_file)
vali_file.close()

test_dict = {"flow":f_test, "flow_mask":f_test_m, "occupancy":o_test, "occupancy_mask":o_test_m}
test_file = open(file_name + "/" + "test" + ".json",'w')
json.dump(test_dict, test_file)
test_file.close()

mean_std_file = open(file_name + "/" + "mean_std" + ".json",'w')
json.dump(mean_std, mean_std_file)
mean_std_file.close()


# In[ ]:




