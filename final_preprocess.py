import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import json

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

path = '/content/drive/MyDrive/Colab Notebooks/Second_Preprocess'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

print('Loading user data...')
user = pd.read_json(path + '/user.json')
print(f'Number of users: {len(user)}')  

print('Creating user index mappings...')
user_ids = user['id'].values 
uid_to_index = {uid: index for index, uid in enumerate(user_ids)}  
index_to_uid = {index: uid for index, uid in enumerate(user_ids)}  

torch.save(uid_to_index, path + '/uid_to_index.pt')
torch.save(index_to_uid, path + '/index_to_uid.pt')

print('User index mappings created and saved.')
print('Loading labels and splits...')
label_df = pd.read_csv(path + '/label.csv')
split_df = pd.read_csv(path + '/split.csv')

label_dict = dict(zip(label_df['id'], label_df['label']))
split_dict = dict(zip(split_df['id'], split_df['split']))

labels_list = []
train_indices = []
val_indices = []
test_indices = []

label_mapping = {'human': 0, 'bot': 1}

print('Processing labels and splits...')
for index, uid in enumerate(tqdm(user_ids)):
    label_value = label_dict.get(uid, None)
    split_value = split_dict.get(uid, None)

    if label_value is None or split_value is None:
        labels_list.append(-1)
    else:
        labels_list.append(label_mapping[label_value])
        if split_value == 'train':
            train_indices.append(index)
        elif split_value == 'val':
            val_indices.append(index)
        elif split_value == 'test':
            test_indices.append(index)
        else:
            pass

labels = torch.LongTensor(labels_list)
train_indices = torch.LongTensor(train_indices)
val_indices = torch.LongTensor(val_indices)
test_indices = torch.LongTensor(test_indices)

torch.save(labels, path + '/labels.pt')
torch.save(train_indices, path + '/train_indices.pt')
torch.save(val_indices, path + '/val_indices.pt')
torch.save(test_indices, path + '/test_indices.pt')

print('Labels and split indices processed and saved.')

print('Processing edge data...')
edge_df = pd.read_csv(path + '/edge-003.csv') 

edge_filtered = edge_df[
    edge_df['source_id'].isin(uid_to_index) & edge_df['target_id'].isin(uid_to_index)
]

edge_index = []
edge_type = []

relation_mapping = {
    'following': 0,
    'friend': 1,
    'follower': 2,
}

print('Creating edge index and edge type tensors...')
for idx, row in tqdm(edge_filtered.iterrows(), total=len(edge_filtered)):
    sid = row['source_id']
    tid = row['target_id']
    relation = row['relation']

    if relation in relation_mapping:
        source_idx = uid_to_index[sid]
        target_idx = uid_to_index[tid]
        edge_index.append([source_idx, target_idx])
        edge_type.append(relation_mapping[relation])
    else:
        continue

edge_index_tensor = torch.LongTensor(edge_index).t()
edge_type_tensor = torch.LongTensor(edge_type)

torch.save(edge_index_tensor, path + '/edge_index.pt')
torch.save(edge_type_tensor, path + '/edge_type.pt')

print('Edge data processed and saved.')
print('Processing numerical features...')
def extract_and_standardize(series):
    series = pd.to_numeric(series, errors='coerce').fillna(0)
    standardized_series = (series - series.mean()) / series.std()
    return torch.tensor(standardized_series.values, dtype=torch.float32).unsqueeze(1)

followers_count = user['public_metrics'].apply(lambda x: x.get('followers_count', 0) if isinstance(x, dict) else 0)
following_count = user['public_metrics'].apply(lambda x: x.get('following_count', 0) if isinstance(x, dict) else 0)
tweet_count = user['public_metrics'].apply(lambda x: x.get('tweet_count', 0) if isinstance(x, dict) else 0)
listed_count = user['public_metrics'].apply(lambda x: x.get('listed_count', 0) if isinstance(x, dict) else 0)
username = user['username'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)

followers_count_tensor = extract_and_standardize(followers_count)
following_count_tensor = extract_and_standardize(following_count)
tweet_count_tensor = extract_and_standardize(tweet_count)
listed_count_tensor = extract_and_standardize(listed_count)
screen_name_length_tensor = extract_and_standardize(username)

num_properties_tensor = torch.cat([
    followers_count_tensor,
    following_count_tensor,
    tweet_count_tensor,
    listed_count_tensor,
    screen_name_length_tensor
], dim=1)

torch.save(num_properties_tensor, path + '/num_properties_tensor.pt')

print('Numerical features processed and saved.')
print('Processing categorical features...')
protected_tensor = torch.tensor(user['protected'].astype(float).values).unsqueeze(1)
verified_tensor = torch.tensor(user['verified'].astype(float).values).unsqueeze(1)

cat_properties_tensor = torch.cat([protected_tensor, verified_tensor], dim=1)

torch.save(cat_properties_tensor, path + '/cat_properties_tensor.pt')

print('Categorical features processed and saved.')

print('Verifying tensor shapes...')
num_users = len(user_ids)
assert labels.shape[0] == num_users, f"Labels tensor has shape {labels.shape}, expected {num_users}"
assert num_properties_tensor.shape[0] == num_users, f"Numerical properties tensor has shape {num_properties_tensor.shape}, expected {num_users}"
assert cat_properties_tensor.shape[0] == num_users, f"Categorical properties tensor has shape {cat_properties_tensor.shape}, expected {num_users}"
print('All tensors have matching shapes.')

print('Data preprocessing completed successfully. All tensors are saved and ready for model training.')
