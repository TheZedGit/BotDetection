import torch
import pandas as pd

path = '/content/drive/MyDrive/Colab Notebooks/Second_Preprocess/'

user = pd.read_json(path + 'user.json') 

if 'location' not in user.columns:
    print("No 'location' column found. Using 'Unknown' for all users.")

user['location'] = user['location'].fillna('Unknown').astype(str)

cat_properties = torch.load(path + 'cat_properties_tensor_updated.pt')

labels = torch.load(path + 'labels.pt')
labels = labels.long()
valid_indices = labels != -1
cat_properties = cat_properties[valid_indices]

user_filtered = user.iloc[valid_indices.cpu().numpy()].reset_index(drop=True)
locations = user_filtered['location']

location_list = []
for loc in locations:
    if loc == 'Unknown':
        location_list.append(0)
    else:
        location_list.append(1)

location_tensor = torch.tensor(location_list, dtype=torch.float32).unsqueeze(1)

cat_properties_updated = torch.cat([cat_properties, location_tensor], dim=1)

torch.save(cat_properties_updated, path + 'cat_properties_tensor_updated_new.pt')
print("Updated cat_properties_tensor_updated.pt saved successfully with binary location feature added.")
print(f"New shape of cat_properties_tensor_updated: {cat_properties_updated.shape}")