import torch
import pandas as pd
import numpy as np
from datetime import datetime as dt

path = '/content/drive/MyDrive/Colab Notebooks/Second_Preprocess/'

print('Loading existing num_properties_tensor...')
num_properties = torch.load(path + 'num_properties_tensor_updated.pt')
print(f'Loaded num_properties_tensor with shape: {num_properties.shape}')

print('Loading user data...')
user = pd.read_json(path + 'user.json') 
print(f'Number of users: {len(user)}') 

print('Parsing created_at dates and computing active_days...')

created_at=user['created_at']
created_at=pd.to_datetime(created_at,unit='s')

date0=dt.strptime('Tue Sep 5 00:00:00 +0000 2020 ','%a %b %d %X %z %Y ')
active_days=[]
for each in created_at:
    active_days.append((date0-each).days)

active_days=pd.DataFrame(active_days)
active_days=active_days.fillna(int(1)).astype(np.float32)

active_days=pd.DataFrame(active_days)
active_days.fillna(int(0))
active_days=active_days.fillna(int(0)).astype(np.float32)

active_days=(active_days-active_days.mean())/active_days.std()
active_days=torch.tensor(np.array(active_days),dtype=torch.float32)

num_properties_updated = torch.cat([num_properties, active_days], dim=1)
print(f'Updated num_properties_tensor shape: {num_properties_updated.shape}')

torch.save(num_properties_updated, path + 'num_properties_tensor_updated.pt')
print('Updated num_properties_tensor saved as num_properties_tensor_updated.pt.')