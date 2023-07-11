#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from torchsummary import summary


# In[2]:


from nmfd_gnn_hinge import NMFD_GNN


# # 1: set parameters

# In[3]:


print (torch.cuda.is_available())
device = torch.device("cuda:0")
random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
r = random.random


# In[4]:


#1.1: settings
M = 10                       #number of time interval in a window
missing_ratio = 0.50
file_name = "m_" + str(M) + "_missing_" + str(int(missing_ratio*100))
print (file_name)

#1.2: hyperparameters
num_epochs, batch_size, learning_rate = 200, 16, 0.001
beta_flow, beta_occ, beta_phy = 1.0, 1.0, 0.1
batch_size_vt = 16  #batch size for evaluation and test
delta_ratio = 0.1   #the ratio of delta in the standard deviation of flow

hyper = {"n_e": num_epochs, "b_s": batch_size, "b_s_vt": batch_size_vt, "l_r": learning_rate,\
         "beta_f": beta_flow, "beta_o": beta_occ, "beta_p": beta_phy, "delta_ratio": delta_ratio}

gnn_dim_1, gnn_dim_2, gnn_dim_3, lstm_dim = 2, 128, 128, 128
p_dim = 10    #column dimension of L1, L2
c_k = 5.5     #meter, the sum of loop width and uniform vehicle length. based on Gero and Daganzo 2008.
theta_ini = [-2.757, 4.996, -2.409, 1.638, 3.569] 

hyper_model = {"g_dim_1": gnn_dim_1, "g_dim_2": gnn_dim_2, "g_dim_3": gnn_dim_3, "l_dim": lstm_dim,\
               "p_dim": p_dim, "c_k": c_k, "theta_ini": theta_ini}
max_no_decrease = 30

#1.3: set paths
root_path = "/home/umni2/a/umnilab/users/xue120/umni4/2023_mfd_traffic/"
file_path = root_path + "2_prepare_data/" + file_name + "/"
train_path, vali_path, test_path =\
    file_path + "train.json", file_path + "vali.json", file_path + "test.json"
sensor_id_path = file_path + "sensor_id_order.json"
sensor_adj_path = file_path + "sensor_adj.json"
mean_std_path = file_path + "mean_std.json"


# # 2: visualization

# In[5]:


def visualize_train_loss(total_phy_flow_occ_loss):
    plt.figure(figsize=(4,3), dpi=75)
    t_p_f_o_l = np.array(total_phy_flow_occ_loss)
    e_loss, p_loss, f_loss, o_loss = t_p_f_o_l[:,0], t_p_f_o_l[:,1], t_p_f_o_l[:,2], t_p_f_o_l[:,3]
    x = range(len(e_loss))
    plt.plot(x, p_loss, linewidth=1, label = "phy loss")
    plt.plot(x, f_loss, linewidth=1, label = "flow loss")
    plt.plot(x, o_loss, linewidth=1, label = "occ loss")
    plt.legend()
    plt.title('Loss decline on train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(file_name + '/' + 'train_loss.png', bbox_inches = 'tight')
    plt.show()
    
def visualize_flow_loss(vali_f_mae, test_f_mae):
    plt.figure(figsize=(4,3), dpi=75)
    x = range(len(vali_f_mae))    
    plt.plot(x, vali_f_mae, linewidth=1, label="Validate")
    plt.plot(x, test_f_mae, linewidth=1, label="Test")
    plt.legend()
    plt.title('MAE of flow on validate/test')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (veh/h)')
    plt.savefig(file_name + '/' + 'flow_mae.png', bbox_inches = 'tight')
    plt.show()
    
def visualize_occ_loss(vali_o_mae, test_o_mae):
    plt.figure(figsize=(4,3), dpi=75)
    x = range(len(vali_o_mae))    
    plt.plot(x, vali_o_mae, linewidth=1, label="Validate")
    plt.plot(x, test_o_mae, linewidth=1, label="Test")
    plt.legend()
    plt.title('MAE of occupancy on validate/test')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.savefig(file_name + '/' + 'occ_mae.png',bbox_inches = 'tight')
    plt.show()


# # 3: compute the error

# In[6]:


def MAELoss(yhat, y):
    return float(torch.mean(torch.div(torch.abs(yhat-y), 1)))

def RMSELoss(yhat, y):
    return float(torch.sqrt(torch.mean((yhat-y)**2)))

def vali_test(model, f, f_mask, o, o_mask, f_o_mean_std, b_s_vt):    
    flow_std, occ_std, n = f_o_mean_std[1], f_o_mean_std[3], len(f)
    f_mae_list, f_rmse_list, o_mae_list, o_rmse_list, num_list = list(), list(), list(), list(), list()
    for i in range(0, n, b_s_vt):
        s, e = i, np.min([i+b_s_vt, n])
        num_list.append(e-s)
        bf, bo, bf_mask, bo_mask = f[s: e], o[s: e], f_mask[s: e], o_mask[s: e]  
        bf_hat, bo_hat, bq_hat, bq_theta = model.run(bf_mask, bo_mask)
        bf_hat, bo_hat = bf_hat.cpu(), bo_hat.cpu()
        bf_mae, bf_rmse = MAELoss(bf_hat, bf)*flow_std, RMSELoss(bf_hat, bf)*flow_std
        bo_mae, bo_rmse = MAELoss(bo_hat, bo)*occ_std, RMSELoss(bo_hat, bo)*occ_std
        f_mae_list.append(bf_mae)
        f_rmse_list.append(bf_rmse)
        o_mae_list.append(bo_mae)
        o_rmse_list.append(bo_rmse)
    f_mae, o_mae = np.dot(f_mae_list, num_list)/n, np.dot(o_mae_list, num_list)/n
    f_rmse = np.sqrt(np.dot(np.multiply(f_rmse_list, f_rmse_list), num_list)/n)
    o_rmse = np.sqrt(np.dot(np.multiply(o_rmse_list, o_rmse_list), num_list)/n)
    return f_mae, f_rmse, o_mae, o_rmse

def evaluate(model, vt_f, vt_o, vt_f_m, vt_o_m, f_o_mean_std, b_s_vt): #vt: vali_test
    vt_f_mae, vt_f_rmse, vt_o_mae, vt_o_rmse  =\
         vali_test(model, vt_f, vt_f_m, vt_o, vt_o_m, f_o_mean_std, b_s_vt)
    return vt_f_mae, vt_f_rmse, vt_o_mae, vt_o_rmse


# # 4: train

# In[7]:


import torch


# In[8]:


#4.1: one training epoch
def train_epoch(model, opt, criterion, train_f_x, train_f_y, train_o_x, train_o_y, hyper, flow_std_squ, delta): 
    #f: flow; o: occupancy
    model.train()
    losses, p_losses, f_losses, o_losses = list(), list(), list(), list()
    
    beta_f, beta_o, beta_p, b_s = hyper["beta_f"], hyper["beta_o"], hyper["beta_p"], hyper["b_s"]
    n = len(train_f_x)
    print ("# batch: ", int(n/b_s))   
    
    for i in range(0, n-b_s, b_s):
        time1 = time.time()
        x_f_batch, y_f_batch = train_f_x[i: i+b_s], train_f_y[i: i+b_s]   
        x_o_batch, y_o_batch = train_o_x[i: i+b_s], train_o_y[i: i+b_s]

        opt.zero_grad() 
        y_f_hat, y_o_hat, q_hat, q_theta = model.run(x_f_batch, x_o_batch)
        
        #p_loss = criterion(q_hat, q_theta).cpu()                #physical loss 
        #p_loss = p_loss/flow_std_squ
        
        #hinge loss
        q_gap = q_hat - q_theta       
        delta_gap = torch.ones(q_gap.shape, device=device)*delta
        zero_gap = torch.zeros(q_gap.shape, device=device)            #(n, m)
        hl_loss = torch.max(q_gap-delta_gap, zero_gap) + torch.max(-delta_gap-q_gap, zero_gap) 
        hl_loss = hl_loss/flow_std_squ
        p_loss = criterion(hl_loss, zero_gap).cpu()            #(n, m)
        f_loss = criterion(y_f_hat.cpu(), y_f_batch)              #data loss of flow
        o_loss = criterion(y_o_hat.cpu(), y_o_batch)              #data loss of occupancy
        
        loss = beta_f*f_loss + beta_o*o_loss + beta_p*p_loss
        
        loss.backward()
        opt.step()
        losses.append(loss.data.numpy())
        p_losses.append(p_loss.data.numpy())
        f_losses.append(f_loss.data.numpy())
        o_losses.append(o_loss.data.numpy())
        
        if i % (64*b_s) == 0:
            print ("i_batch: ", i/b_s)
            print ("the loss for this batch: ", loss.data.numpy())
            print ("flow loss", f_loss.data.numpy())
            print ("occ loss", o_loss.data.numpy())
            time2 = time.time()
            print ("time for this batch", time2-time1)
            print ("----------------------------------")
        n_loss = float(len(losses)+0.000001)
        aver_loss = sum(losses)/n_loss
        aver_p_loss = sum(p_losses)/n_loss
        aver_f_loss = sum(f_losses)/n_loss
        aver_o_loss = sum(o_losses)/n_loss
    return aver_loss, model, aver_p_loss, aver_f_loss, aver_o_loss

#4.2: all train epochs
def train_process(model, criterion, train, vali, test, hyper, f_o_mean_std):
    total_phy_flow_occ_loss = list()
    
    n_mse_flow_occ = 0 #mse(flow) + mse(occ) for validation sets.
    f_std = f_o_mean_std[1]
    
    vali_f, vali_o = vali["flow"], vali["occupancy"] 
    vali_f_m, vali_o_m = vali["flow_mask"].to(device), vali["occupancy_mask"].to(device) 
    test_f, test_o = test["flow"], test["occupancy"] 
    test_f_m, test_o_m = test["flow_mask"].to(device), test["occupancy_mask"].to(device) 
    
    l_r, n_e = hyper["l_r"], hyper["n_e"]
    opt = optim.Adam(model.parameters(), l_r, betas = (0.9,0.999), weight_decay=0.0001)
    opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[150])
    
    print ("# epochs ", n_e)
    r_vali_f_mae, r_vali_o_mae, r_test_f_mae, r_test_o_mae = list(), list(), list(), list()
    r_vali_f_rmse, r_vali_o_rmse, r_test_f_rmse, r_test_o_rmse = list(), list(), list(), list()
    
    flow_std_squ = np.power(f_std, 2)
    
    no_decrease = 0
    for i in range(n_e):
        print ("----------------an epoch starts-------------------")
        #time1_s = time.time()
        
        time_s = time.time()
        print ("i_epoch: ", i)
        n_train = len(train["flow"])
        number_list = copy.copy(list(range(n_train)))
        random.shuffle(number_list, random = r)
        shuffle_idx = torch.tensor(number_list)
        train_x_f, train_y_f = train["flow_mask"][shuffle_idx], train["flow"][shuffle_idx]
        train_x_o, train_y_o = train["occupancy_mask"][shuffle_idx], train["occupancy"][shuffle_idx] 
        
        delta = hyper["delta_ratio"] * f_std
        aver_loss, model, aver_p_loss, aver_f_loss, aver_o_loss =\
            train_epoch(model, opt, criterion, train_x_f.to(device), train_y_f,\
                        train_x_o.to(device), train_y_o, hyper, flow_std_squ, delta)
        opt_scheduler.step()
        
        total_phy_flow_occ_loss.append([aver_loss, aver_p_loss, aver_f_loss, aver_o_loss])
        print ("train loss for this epoch: ", round(aver_loss, 6))
        
        #evaluate
        b_s_vt = hyper["b_s_vt"]
        vali_f_mae, vali_f_rmse, vali_o_mae, vali_o_rmse =\
            evaluate(model, vali_f, vali_o, vali_f_m, vali_o_m, f_o_mean_std, b_s_vt)
        test_f_mae, test_f_rmse, test_o_mae, test_o_rmse =\
            evaluate(model, test_f, test_o, test_f_m, test_o_m, f_o_mean_std, b_s_vt)  
        
        r_vali_f_mae.append(vali_f_mae)
        r_test_f_mae.append(test_f_mae)
        r_vali_o_mae.append(vali_o_mae)
        r_test_o_mae.append(test_o_mae)
        r_vali_f_rmse.append(vali_f_rmse)
        r_test_f_rmse.append(test_f_rmse)
        r_vali_o_rmse.append(vali_o_rmse)
        r_test_o_rmse.append(test_o_rmse)
        
        visualize_train_loss(total_phy_flow_occ_loss)
        visualize_flow_loss(r_vali_f_mae, r_test_f_mae)
        visualize_occ_loss(r_vali_o_mae, r_test_o_mae)
        time_e = time.time()
        print ("time for this epoch", time_e - time_s)
        
        performance = {"train": total_phy_flow_occ_loss,\
                  "vali": [r_vali_f_mae, r_vali_f_rmse, r_vali_o_mae, r_vali_o_rmse],\
                  "test": [r_test_f_mae, r_test_f_rmse, r_test_o_mae, r_test_o_rmse]}
        subfile =  open(file_name + '/' + 'performance'+'.json','w')
        json.dump(performance, subfile)
        subfile.close()
        
        #early stop
        flow_std, occ_std = f_o_mean_std[1], f_o_mean_std[3]
        norm_f_rmse, norm_o_rmse = vali_f_rmse/flow_std, vali_o_rmse/occ_std
        norm_sum_mse = norm_f_rmse*norm_f_rmse + norm_o_rmse*norm_o_rmse
        
        if n_mse_flow_occ > 0:
            min_until_now = min([min_until_now, norm_sum_mse])
        else:
            min_until_now = 1000000.0  
        if norm_sum_mse > min_until_now:
            no_decrease = no_decrease+1
        else:
            no_decrease = 0
        if no_decrease == max_no_decrease:
            print ("Early stop at the " + str(i+1) + "-th epoch")
            return total_phy_flow_occ_loss, model 
        n_mse_flow_occ = n_mse_flow_occ + 1
        
        print ("No_decrease: ", no_decrease)
    return total_phy_flow_occ_loss, model    


# # 5: prepare tensors

# In[9]:


def tensorize(train_vali_test):
    result = dict()
    result["flow"] = torch.tensor(train_vali_test["flow"]) 
    result["flow_mask"] = torch.tensor(train_vali_test["flow_mask"])     
    result["occupancy"] = torch.tensor(train_vali_test["occupancy"]) 
    result["occupancy_mask"] = torch.tensor(train_vali_test["occupancy_mask"]) 
    return result

def normalize_flow_occ(tvt, f_o_mean_std):  #tvt: train, vali, test
    #flow
    f_mean, f_std = f_o_mean_std[0], f_o_mean_std[1]
    f_mask, f = tvt["flow_mask"], tvt["flow"]
    tvt["flow_mask"] = ((np.array(f_mask)-f_mean)/f_std).tolist()
    tvt["flow"] = ((np.array(f)-f_mean)/f_std).tolist()
    
    #occ
    o_mean, o_std = f_o_mean_std[2], f_o_mean_std[3]
    o_mask, o = tvt["occupancy_mask"], tvt["occupancy"]
    tvt["occupancy_mask"] = ((np.array(o_mask)-o_mean)/o_std).tolist()
    tvt["occupancy"] = ((np.array(o)-o_mean)/o_std).tolist()   
    return tvt

def transform_distance(d_matrix):
    sigma, n_row, n_col = np.std(d_matrix), len(d_matrix), len(d_matrix[0])
    sigma_square = sigma*sigma
    for i in range(n_row):
        for j in range(n_col):
            d_i_j = d_matrix[i][j]
            d_matrix[i][j] = np.exp(0.0-10000.0*d_i_j*d_i_j/sigma_square)
    return d_matrix

def load_data(train_path, vali_path, test_path, sensor_adj_path, mean_std_path, sensor_id_path):
    mean_std = json.load(open(mean_std_path))
    f_mean, f_std, o_mean, o_std =\
        mean_std["f_mean"], mean_std["f_std"], mean_std["o_mean"], mean_std["o_std"]
    f_o_mean_std = [f_mean, f_std, o_mean, o_std]
    
    train = json.load(open(train_path))
    vali = json.load(open(vali_path))
    test = json.load(open(test_path))
    adj = json.load(open(sensor_adj_path))["adj"]
    n_sensor = len(train["flow"][0])    
    
    train = tensorize(normalize_flow_occ(train, f_o_mean_std))
    vali = tensorize(normalize_flow_occ(vali, f_o_mean_std))
    test = tensorize(normalize_flow_occ(test, f_o_mean_std))

    adj = torch.tensor(transform_distance(adj), device=device).float()   
    
    df_sensor_id = json.load(open(sensor_id_path))
    sensor_length = [0.0 for i in range(n_sensor)]
    for sensor in df_sensor_id:
        sensor_length[df_sensor_id[sensor][0]] = df_sensor_id[sensor][3]
        
    return train, vali, test, adj, n_sensor, f_o_mean_std, sensor_length


# # 6: main

# In[10]:


#6.1 load the data
time1 = time.time()
train, vali, test, adj, n_sensor, f_o_mean_std, sensor_length =\
    load_data(train_path, vali_path, test_path, sensor_adj_path, mean_std_path, sensor_id_path)
time2 = time.time()
print (time2-time1)


# In[11]:


print (len(train["flow"]))
print (len(vali["flow"]))
print (len(test["flow"]))
print (f_o_mean_std)


# In[12]:


model = NMFD_GNN(n_sensor, M, hyper_model, f_o_mean_std, sensor_length, adj).to(device)   
cri = nn.MSELoss() 


# In[13]:


#6.2: train the model
total_phy_flow_occ_loss, trained_model = train_process(model, cri, train, vali, test, hyper, f_o_mean_std)


# # 7: apply the model to vali and test

# In[14]:


def apply_to_vali_test(model, vt, f_o_mean_std):
    f = vt["flow"]
    f_m = vt["flow_mask"].to(device)
    o = vt["occupancy"]
    o_m = vt["occupancy_mask"].to(device)
    f_mae, f_rmse, o_mae, o_rmse  = vali_test(model, f, f_m, o, o_m, f_o_mean_std, hyper["b_s_vt"])
    print ("flow_mae", f_mae)
    print ("flow_rmse", f_rmse)
    print ("occ_mae", o_mae)
    print ("occ_rmse", o_rmse)
    return f_mae, f_rmse, o_mae, o_rmse


# # Validate

# In[15]:


vali_f_mae, vali_f_rmse, vali_o_mae, vali_o_rmse =\
    apply_to_vali_test(trained_model, vali, f_o_mean_std)


# # Test

# In[16]:


test_f_mae, test_f_rmse, test_o_mae, test_o_rmse =\
    apply_to_vali_test(trained_model, test, f_o_mean_std)


# In[ ]:





# In[ ]:





# In[ ]:




