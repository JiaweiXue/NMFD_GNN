#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
random.seed(42)
r = random.random
device = torch.device("cuda:0")


# In[3]:


#input1: adj             dim = (n_sensor, n_sensor)
#input2: x_f_batch       dim = (b_s, n_sensor, M)
#input3: x_o_batch       dim = (b_s, n_sensor, M)
#output1: y_f_hat        dim = (b_s, n_sensor, M)  
#output2: y_o_hat        dim = (b_s, n_sensor, M)


# In[4]:


class SpecGCNLayer(nn.Module):
    def __init__(self, dim_1, dim_2):
        super(SpecGCNLayer, self).__init__()
        self.inp_dim = dim_1
        self.hid_dim = dim_2
        
        self.W = nn.Parameter(torch.empty(size=(self.inp_dim, self.hid_dim)))
        self.W.requires_grad = True
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.W_adp = nn.Parameter(torch.empty(size=(self.inp_dim, self.hid_dim)))
        self.W_adp.requires_grad = True
        nn.init.xavier_uniform_(self.W_adp.data, gain=1.414)
    
    def forward(self, feature, D_n_A_D_n):
        feature = torch.mm(feature.float(), self.W)
        H = torch.mm(D_n_A_D_n.float(), feature)  #H.shape: (N, output_dim)
        return H 
    
    def forward_adp(self, feature, A_adp):
        feature = torch.mm(feature.float(), self.W_adp)
        H_adp = torch.mm(A_adp.float(), feature)  #H.shape: (N, output_dim)
        return H_adp


# In[5]:


class SpecGCN(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3, n_s, p_dim):
        super(SpecGCN, self).__init__()
        self.layer1 = SpecGCNLayer(dim_1, dim_2)
        self.layer2 = SpecGCNLayer(dim_2, dim_3)
        self.L1 = nn.Parameter(torch.empty(size=(n_s, p_dim))) 
        self.L2 = nn.Parameter(torch.empty(size=(p_dim, n_s))) 
        self.L1.requires_grad = True
        self.L2.requires_grad = True
        torch.nn.init.normal_(self.L1.data)
        torch.nn.init.normal_(self.L2.data)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.eye_n_s = torch.eye(n_s, device=device)
        
    def compute_D_n_A_D_n(self, adjs): 
        tilde_A = adjs + self.eye_n_s
        tilde_D_n = torch.diag(torch.pow(tilde_A.sum(-1).float(), -0.5))
        D_n_A_D_n = torch.mm(tilde_D_n, torch.mm(tilde_A, tilde_D_n))
        return D_n_A_D_n 
    
    def forward(self, x, D_n_A_D_n):
        x1_fix = self.layer1(x, D_n_A_D_n)
        x1_fix = torch.tanh(x1_fix)
        
        new_adj = torch.mm(self.L1, self.L2)
        A_adp = F.softmax(self.relu(new_adj),dim=0)
        x1_adp = self.layer1.forward_adp(x, A_adp)
        x1_adp = torch.tanh(x1_adp)
        x1 = x1_fix + x1_adp
        
        x2_fix = self.layer2(x1, D_n_A_D_n)
        x2_fix = torch.tanh(x2_fix)
        
        x2_adp = self.layer2.forward_adp(x1, A_adp)
        x2_adp = torch.tanh(x2_adp)
        x2 = x2_fix + x2_adp
        return x2


# In[6]:


class NMFD_GNN(nn.Module):
    def __init__(self, n_sensor, M, hyper_model, f_o_mean_std, sensor_length, adj):
        super(NMFD_GNN, self).__init__()
        self.n_s = n_sensor                         #n_s: n_sensor
        self.m = M
        self.g_dim_1 = hyper_model["g_dim_1"]       #g: gnn
        self.g_dim_2 = hyper_model["g_dim_2"]
        self.g_dim_3 = hyper_model["g_dim_3"]
        self.l_dim = hyper_model["l_dim"]           #l: lstm
        self.p_dim = hyper_model["p_dim"]           #p: L1, L2, dimension
        
        self.f_mean = f_o_mean_std[0]
        self.f_std = f_o_mean_std[1]
        self.o_mean = f_o_mean_std[2]
        self.o_std = f_o_mean_std[3]
        
        self.ck = hyper_model["c_k"]
        
        theta_ini = torch.tensor(hyper_model["theta_ini"], device=device)
        self.theta =  torch.nn.Parameter(theta_ini)
        self.theta.requires_grad = True
        
        self.s_length = torch.tensor(sensor_length, device=device)
        self.s_length_sum = torch.sum(self.s_length)
        
        self.specGCN = SpecGCN(self.g_dim_1, self.g_dim_2, self.g_dim_3, self.n_s, self.p_dim)
        self.bilstm = nn.LSTM(batch_first=True, input_size=self.g_dim_3, \
                            hidden_size=self.l_dim, num_layers=2, bidirectional=True)
        self.W = nn.Parameter(torch.empty(size=(2*self.l_dim, 2)))
        self.W.requires_grad = True
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.D_n_A_D_n = self.specGCN.compute_D_n_A_D_n(adj) 

    def run(self, x_f_batch, x_o_batch):
        #prepare inputs
        #time1_s = time.time()
        n = len(x_f_batch)
        x_f_batch_tras = x_f_batch.transpose(1, 2)    #(n, m, n_s)
        x_o_batch_tras = x_o_batch.transpose(1, 2)    #(n, m, n_s)
        x_f_batch_tras = torch.unsqueeze(x_f_batch_tras, dim=-1)   #(n, m, n_s, 1)
        x_o_batch_tras = torch.unsqueeze(x_o_batch_tras, dim=-1)   #(n, m, n_s, 1)
        x_f_o_batch = torch.cat((x_f_batch_tras, x_o_batch_tras), 3)  #(n, m, n_s, 2)
        
        #GNN
        gcn_output = torch.zeros((n, self.m, self.n_s, self.g_dim_2), device=device).float()       
        for i in range(n):
            x_f_o_batch_i = x_f_o_batch[i]           #(m, n_s, 2)
            for j in range(self.m):
                gcn_output[i][j] = self.specGCN(x_f_o_batch_i[j], self.D_n_A_D_n)  
                #(n_s, g_dim_2)
        
        #bidirectional LSTM
        gcn_output = gcn_output.transpose(1, 2)   #(n, n_s, m, g_dim_2)
        gcn_output = gcn_output.transpose(0, 1)   #(n_s, n, m, g_dim_2)
        
        lstm_output, (hc,cn) = self.bilstm(gcn_output.reshape(self.n_s*n, self.m, self.g_dim_2)) #(n_s*n, m, l_dim)
        lstm_output = lstm_output.reshape(self.n_s, n, self.m, 2*self.l_dim)
        
        #MLP
        mlp_input = lstm_output.transpose(0, 1) #(n, n_s, m, 2*l_dim)       
        mlp_output = torch.matmul(mlp_input, self.W)  #(n, n_s, m, 2)
        flow_y_hat, occ_y_hat = mlp_output[:,:,:,0], mlp_output[:,:,:,1] #(n, n_s, m)
        
        #hat_q
        #time5_s = time.time()
        ##convert the flow and occupancy back using mean and std
        flow_hat_original = flow_y_hat*self.f_std + self.f_mean  #(n, n_s, m)
        occ_hat_original = occ_y_hat*self.o_std + self.o_mean    #(n, n_s, m)
        
        ##convert occupancy to density
        k_hat = occ_hat_original/self.ck                        #(n, n_s, m)
        
        ##aggregate the q and k using the length 
        flow_hat_t = flow_hat_original.transpose(1, 2)                                 #(n, m, n_s)
        q_hat = torch.sum(torch.mul(flow_hat_t, self.s_length), 2)/self.s_length_sum   #(n, m)
        
        k_hat_t = k_hat.transpose(1, 2)                                                #(n, m, n_s)
        k_hat = torch.sum(torch.mul(k_hat_t, self.s_length), 2)/self.s_length_sum      #(n, m)
        
        ##hat_q_theta
        exp_theta = torch.exp(self.theta)
        term_1 = torch.exp(-exp_theta[1]*k_hat)
        term_2 = exp_theta[2] 
        term_3 = torch.exp(-exp_theta[3] + exp_theta[4]*k_hat)
        q_hat_theta = -exp_theta[0] * torch.log(term_1 + term_2 + term_3) 
        
        return flow_y_hat, occ_y_hat, q_hat, q_hat_theta

