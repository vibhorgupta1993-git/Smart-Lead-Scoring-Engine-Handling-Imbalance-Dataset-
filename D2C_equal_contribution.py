import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import os
from sklearn.preprocessing import OneHotEncoder
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

torch.manual_seed(10)
data = pd.read_csv('DATA/train_wn75k28.csv', sep = ',', parse_dates = ['created_at'])
print(data.isnull().sum())
print(data.describe())
print(Counter(data['buy']))

# Filling product purchased NAN value with 0
data.fillna({'products_purchased': 0}, inplace = True)

target = 'buy'
reluctant_features = ['id', 'created_at', 'signup_date', 'buy']
independent_columns = [c for c in data.columns.tolist() if c not in reluctant_features]
print('independent_features_name: \n', independent_columns)

# Creating independent dataset and target
X = data[independent_columns]
Y = data[target]

def one_hot(dataframe, column_name):
  one_hot = pd.get_dummies(dataframe[column_name])
  dataframe = dataframe.drop(column_name,axis = 1)
  dataframe = dataframe.join(one_hot)
  return dataframe
#X = one_hot(X, 'products_purchased')
print(X)

# Handling imbalance dataset
smk = SMOTE(random_state = 45)
smk = NearMiss()
X_over, Y_over = smk.fit_resample(X, Y)
print((Y_over.shape))
X_over = X
Y_over = Y
# Splitting dataset into train and Validation dataset
X_train, X_val, Y_train, Y_val = train_test_split(X_over, Y_over, test_size = 0.4, random_state = 45)
print(X_train.shape)
X_train = X_over
Y_train = Y_over

print('>>>>>>>>>', X_train.shape)
print(">>>>>>>>", Y_train.shape)
X_train = pd.concat([X_train, Y_train], axis=1)
print(X_train)
X_train_positive = X_train[X_train['buy'] == 1]
X_train_negative = X_train[X_train['buy'] == 0]
Y_train_positive = X_train_positive[target]
Y_train_negative = X_train_negative[target]
X_train_positive = X_train_positive[independent_columns]
X_train_negative = X_train_negative[independent_columns]

# Normalizing dataset FIrst try z-normalisation
scaler = StandardScaler()
X_train_negative = scaler.fit_transform(X_train_negative)
X_train_positive = scaler.transform(X_train_positive)
X_val = scaler.transform(X_val)
#Train and Validation dataset loader

class TrainData(Dataset):
    
  def __init__(self, X_data, y_data):
    self.X_data = X_data
    self.y_data = y_data
        
  def __getitem__(self, index):
      return self.X_data[index], self.y_data[index]
      
  def __len__ (self):
      return len(self.X_data)
print(X_train_positive.shape)

pos = pd.DataFrame(X_train_positive)
pos['target'] = Y_train_positive.values

neg = pd.DataFrame(X_train_negative)
neg['target'] = Y_train_negative.values


## test data    

class TestData(Dataset):
    
  def __init__(self, X_data):
      self.X_data = X_data
      
  def __getitem__(self, index):
      return self.X_data[index]
      
  def __len__ (self):
      return len(self.X_data)

train_data_positive = TestData(torch.FloatTensor(pos.values))
train_data_negative = TestData(torch.FloatTensor(neg.values))

val_data = TrainData(torch.FloatTensor(X_val), 
                    torch.FloatTensor(Y_val))
test_data = TestData(torch.FloatTensor(X_val))

# DataLoader

train_loader_positive = DataLoader(dataset=train_data_positive, batch_size=32, shuffle=True)
train_loader_negative = DataLoader(dataset=train_data_negative, batch_size=32, shuffle=True)
#train_loader_t = zip(train_loader_positive, train_loader_negative)
val_loader = DataLoader(dataset=val_data, batch_size=100000, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=100000)

#model Creation
class BinaryClassification(nn.Module):
  def __init__(self):
      super(BinaryClassification, self).__init__()        # Number of input features is 12.
      self.layer_1 = nn.Linear(15, 64) 
      self.layer_2 = nn.Linear(64, 64)
      self.layer_out = nn.Linear(64, 1) 
      
      self.relu = nn.ReLU()
      self.dropout = nn.Dropout(p=0.2)
      self.batchnorm1 = nn.BatchNorm1d(64)
      self.batchnorm2 = nn.BatchNorm1d(64)
      
  def forward(self, inputs):
      x = self.relu(self.layer_1(inputs))
      x = self.batchnorm1(x)
      x = self.relu(self.layer_2(x))
      x = self.batchnorm2(x)
      x = self.dropout(x)
      x = self.layer_out(x)
      
      return x

model = BinaryClassification()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr= 0.005)#LEARNING_RATE)

# accuracy prediction
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

model.train()
for e in range(1, 120+1):
  
  epoch_loss = 0
  epoch_acc = 0
  for i, (positive, negative) in enumerate(zip(cycle(train_loader_positive), train_loader_negative)):
    
    data = torch.cat((positive,negative), axis = 0)
    data = data[torch.randperm(data.size()[0])]
    X_batch = data[:,:-1]
    y_batch = data[:,-1]
    #print(X_batch)
    #print(y_batch)
    optimizer.zero_grad()
    y_pred = model(X_batch)
    loss = criterion(y_pred, y_batch.unsqueeze(1))
    acc = binary_acc(y_pred, y_batch.unsqueeze(1))
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
    epoch_acc += acc.item()
    
  model.eval()
  with torch.no_grad():
    for X_batch_val, y_batch_val in val_loader:
        y_test_pred = model(X_batch_val)
        loss = criterion(y_test_pred, y_batch_val.unsqueeze(1))
        acc = binary_acc(y_test_pred, y_batch_val.unsqueeze(1))

  #saving model
  directory = 'model/'
  if not os.path.exists(directory):
    os.mkdir(directory)
  torch.save(model.state_dict(), directory + "Epoch_"+str(e)+"_loss_"+str(loss.item()))
      
  
  print(f'Epoch {e+0:03}: | Loss: {epoch_loss/i:.5f} | Acc: {epoch_acc/i:.3f}')

y_pred_list = []
import glob
files = glob.glob('model/Epoch_*_loss_*')
min_epoch = 10000
min_file = files[0]
for file in files:
  min_epoch_file = float(file.split('/')[1].split('_')[3])
  if min_epoch_file < min_epoch:
    min_epoch = min_epoch_file
    min_file = file


model.load_state_dict(torch.load(min_file))
model.eval()
with torch.no_grad():
  for X_batch in test_loader:
      y_test_pred = model(X_batch)
      y_test_pred = torch.sigmoid(y_test_pred)
      y_pred_tag = torch.round(y_test_pred)
      y_pred_list.append(y_pred_tag.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
print(classification_report(Y_val, y_pred_list[0]))

X_test = pd.read_csv('DATA/test_Wf7sxXF.csv', sep = ',')
X_test.fillna({'products_purchased':0}, inplace = True)

X_test_id = X_test['id'].values
X_test = X_test[independent_columns]
#X_test = one_hot(X_test, 'products_purchased')
X_test = scaler.transform(X_test)
test_data = TestData(torch.FloatTensor(X_test))
test_loader = DataLoader(dataset=test_data, batch_size=100000)

y_pred_list = []
model.load_state_dict(torch.load(min_file))
model.eval()
with torch.no_grad():
  for X_batch in test_loader:
      y_test_pred = model(X_batch)
      y_test_pred = torch.sigmoid(y_test_pred)
      y_pred_tag = torch.round(y_test_pred)
      y_pred_list.append(y_pred_tag.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
print(Counter(y_pred_list[0]))
dict_ = {'id':X_test_id, 'buy':y_pred_list[0]}
dataframe = pd.DataFrame(dict_)
dataframe.to_csv('Final_submission.csv', sep = ',', index = False)