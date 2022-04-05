#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim


# 查看设备

# In[2]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print(f'The model will be running on {device} device')


# 定义一个NN类

# In[3]:


class Net(nn.Module):
    def __init__(self, feature_count, class_count):
        super (Net, self).__init__()
        
        self.hidden_layers = nn.Sequential(nn.Linear(feature_count, 10000),
                                            nn.ReLU(True),
                                            nn.Linear(10000, 5000),
                                            nn.ReLU(True),
                                            nn.Linear(5000, 5000),
                                            nn.ReLU(True),
                                            nn.Linear(5000, 1000),
                                            nn.ReLU(True),
                                            nn.Linear(1000, class_count))
        
    def forward(self, x):
        outputs = self.hidden_layers(x)
        return outputs


# 定义训练函数

# In[4]:


def train(whole_train_set):
    train_set, valid_set = random_split(whole_train_set, [400, 100])
    train_loader = DataLoader(train_set, batch_size = 50, shuffle = True)
    valid_loader = DataLoader(valid_set, batch_size = 50, shuffle = True)
    
    model = Net(len(whole_train_set[0][0]), 4)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    best_accuracy = 0
    
    for epoch in range(500):
        running_train_loss = 0.0  
        running_val_loss = 0.0
        correct, total = 0, 0 
        for i, data in enumerate(train_loader, 0):
            X = data[0].to(device)
            y = data[1].to(device)
            optimizer.zero_grad()
            
            outputs = model(X)  
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss
        
        train_loss = running_train_loss/len(train_loader)
        
        with torch.no_grad():
            model.eval()
            
            for i, data in enumerate(valid_loader, 0):
                X = data[0].to(device)
                y = data[1].to(device)
            
                outputs = model(X)
                loss = criterion(outputs, y)
                _, y_pred = torch.max(outputs, dim = 1)
                running_val_loss += loss
                total += outputs.size(0)
                correct += (y == y_pred).sum().item()
            
        val_loss = running_val_loss/len(valid_loader)
        
        accuracy = 100*correct/total
        
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), 'best_model.pth')
            print('The model has been saved for the best accuracy %d %%'%(accuracy))
            best_accuracy = accuracy
        
        if epoch == 0:
            print('The model is working !')

        if (epoch + 1)%100 == 0:
            print('Completed training epoch', epoch + 1, 'Training Loss is: %.4f' %train_loss, 'Validation Loss is: %.4f' %val_loss, 'Accuracy is %d %%' % (accuracy))
        


# 导入数据

# In[5]:


metadata = pd.read_csv('../COAD/metadata.csv', index_col = 2)[['disease_type', 
                                                                'sample_type', 
                                                                'pathologic_stage_label']]
trans_data  = pd.read_csv('../COAD/transcriptome/trancriptome.csv')


# 数据预处理

# In[ ]:


def StageNormalize(stage):
    stage = str(stage)
    if re.search('Stage IV', stage):
        return 3
    elif re.search('Stage III', stage):
        return 2
    elif re.search('Stage II', stage):
        return 1
    elif re.search('Stage I', stage):
        return 0
    else:
        return np.nan

metadata['pathologic_stage_label'] = metadata['pathologic_stage_label'].apply(StageNormalize)


# In[ ]:


metadata.head()


# In[ ]:


trans_data = trans_data.set_index('id').drop('Unnamed: 0', axis = 1)
metadata.dropna(subset = ['pathologic_stage_label'], inplace = True)


# In[ ]:


processed_X  = trans_data.loc[metadata.index]
labels = metadata['pathologic_stage_label']

processed_X = np.array(processed_X).astype(np.float32)
labels = np.array(labels).astype(int)

processed_data = TensorDataset(torch.from_numpy(processed_X), torch.from_numpy(labels))
whole_train_set, whole_valid_set = random_split(processed_data, [500, 25])


# 训练

# In[ ]:


train(whole_train_set)


# 

# In[ ]:




