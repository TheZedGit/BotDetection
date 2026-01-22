import torch
import pandas as pd

path = '/content/drive/MyDrive/Colab Notebooks/Second_Preprocess/'

cat_properties = torch.load(path + 'cat_properties_tensor.pt')

user = pd.read_json(path + 'user.json')

default_profile_image = []
for each in user['profile_image_url']:
    if pd.notnull(each):
        if each == 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png':
            default_profile_image.append(1)
        elif each == '':
            default_profile_image.append(1)
        else:
            default_profile_image.append(0)
    else:
        default_profile_image.append(1)

default_profile_image_tensor = torch.tensor(default_profile_image, dtype=torch.float32)

assert default_profile_image_tensor.shape[0] == cat_properties.shape[0], "Mismatch in number of nodes"

default_profile_image_tensor = default_profile_image_tensor.unsqueeze(1)

cat_properties_updated = torch.cat([cat_properties, default_profile_image_tensor], dim=1)

torch.save(cat_properties_updated, path + 'cat_properties_tensor_updated.pt')

print('Updated cat_properties_tensor saved as cat_properties_tensor_updated.pt.')