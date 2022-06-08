import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import os
from sklearn.preprocessing import OneHotEncoder

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
reluctant_features = ['id', 'created_at', 'user_activity_var_1',  'user_activity_var_3',\
'user_activity_var_5', 'user_activity_var_6', 'user_activity_var_7', 'user_activity_var_8', 'user_activity_var_10', 'user_activity_var_12','signup_date', 'buy']
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
print(Counter(Y_over))
list_Classes =np.array([Counter(Y_over)[0], Counter(Y_over)[1]])
print(list_Classes)
class_weight = 1./list_Classes#[1- (i/sum(list_Classes)) for i in list_Classes][1]
print(class_weight)


# Splitting dataset into train and Validation dataset
X_train, X_val, Y_train, Y_val = train_test_split(X_over, Y_over, test_size = 0.3, random_state = 45)
print(X_train.shape)
#X_train = X_over
#Y_train = Y_over
sample_weight = np.array([class_weight[t] for t in Y_train])
sample_weight = torch.from_numpy(sample_weight)
from torch.utils.data.sampler import WeightedRandomSampler
sampler = WeightedRandomSampler(sample_weight, len(sample_weight))

# Normalizing dataset FIrst try z-normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
#Train and Validation dataset loader

print(X_train.shape)
print(X_val.shape)

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,Y_train)

y_pred=clf.predict(X_val)
print(Y_val)
feature_imp = pd.Series(clf.feature_importances_).sort_values(ascending=False)
print(feature_imp)
print(classification_report(Y_val, y_pred))
X_test = pd.read_csv('DATA/test_Wf7sxXF.csv', sep = ',')
X_test.fillna({'products_purchased':0}, inplace = True)

X_test_id = X_test['id'].values
X_test = X_test[independent_columns]
#X_test = one_hot(X_test, 'products_purchased')
X_test = scaler.transform(X_test)
y_pred = clf.predict(X_test)
print(Counter(y_pred))
dict_ = {'id':X_test_id, 'buy':y_pred}
dataframe = pd.DataFrame(dict_)
dataframe.to_csv('Final_submission.csv', sep = ',', index = False)
# Number of trees in random forest
n_estimators = np.linspace(100, 1000, int((1000-100)/200) + 1, dtype=int)
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [1, 5, 100, 150, 200]# Minimum number of samples required to split a node
# min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 10, num = 9)]
min_samples_split = [1, 2, 20, 30]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1,  4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Criterion
criterion=['gini', 'entropy']
random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion': criterion}
print(random_grid)
from sklearn.model_selection import RandomizedSearchCV
rf_base = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf_base,
                               param_distributions = random_grid,
                               n_iter = 50, cv = 5,
                               verbose=2,
                               random_state=42, n_jobs = 4)
rf_random.fit(X_train,Y_train)
print(rf_random.best_params_)

'''test_data = TestData(torch.FloatTensor(X_test))
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
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]'''



















'''class TrainData(Dataset):
    
  def __init__(self, X_data, y_data):
    self.X_data = X_data
    self.y_data = y_data
        
  def __getitem__(self, index):
      return self.X_data[index], self.y_data[index]
      
  def __len__ (self):
      return len(self.X_data)


train_data = TrainData(torch.FloatTensor(X_train), 
                       torch.FloatTensor(Y_train))
## test data    

class TestData(Dataset):
    
  def __init__(self, X_data):
      self.X_data = X_data
      
  def __getitem__(self, index):
      return self.X_data[index]
      
  def __len__ (self):
      return len(self.X_data)
    
val_data = TrainData(torch.FloatTensor(X_val), 
                    torch.FloatTensor(Y_val))
test_data = TestData(torch.FloatTensor(X_val))

# DataLoader

train_loader = DataLoader(dataset=train_data, batch_size=64, num_workers = 1, sampler= sampler)
val_loader = DataLoader(dataset=val_data, batch_size=100000, shuffle=False, num_workers = 1)
test_loader = DataLoader(dataset=test_data, batch_size=100000, num_workers = 1, shuffle = False)

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
optimizer = optim.Adam(model.parameters(), lr= 0.01)#LEARNING_RATE)

# accuracy prediction
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc
from sklearn.metrics import f1_score
model.train()
for e in range(1, 40+1):
  epoch_loss = 0
  epoch_acc = 0
  for X_batch, y_batch in train_loader:
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
      

  print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

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
dataframe.to_csv('Final_submission.csv', sep = ',', index = False)'''