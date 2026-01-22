import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import random
import numpy as np
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors

path = '/content/drive/MyDrive/Colab Notebooks/Second_Preprocess/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Loading processed data...')
labels = torch.load(path + 'labels.pt').to(device)
num_properties = torch.load(path + 'num_properties_tensor_updated.pt').to(device)
cat_properties = torch.load(path + 'cat_properties_tensor_updated.pt').to(device)
edge_index = torch.load(path + 'edge_index.pt').to(device)
edge_type = torch.load(path + 'edge_type.pt').to(device)

labels = labels.long()
num_properties = num_properties.float()
cat_properties = cat_properties.float()
edge_index = edge_index.long()
edge_type = edge_type.long()

tweet_embeddings = torch.load(path + 'tweets_tensor.pt').to(device).float()
user_embeddings = torch.load(path + 'user_embeddings.pt').to(device).float()
user_name_embeddings = torch.load(path + 'user_username_embeddings.pt').to(device).float()

train_idx = torch.load(path + 'train_indices.pt').to(device).long()
val_idx = torch.load(path + 'val_indices.pt').to(device).long()
test_idx = torch.load(path + 'test_indices.pt').to(device).long()

valid_indices = labels != -1
labels = labels[valid_indices]
num_properties = num_properties[valid_indices]
cat_properties = cat_properties[valid_indices]
tweet_embeddings = tweet_embeddings[valid_indices]
user_embeddings = user_embeddings[valid_indices]
user_name_embeddings = user_name_embeddings[valid_indices]

old_to_new_indices = {i: new_idx for new_idx, i in enumerate(valid_indices.nonzero(as_tuple=True)[0].tolist())}
train_idx = torch.tensor([old_to_new_indices[i.item()] for i in train_idx if i.item() in old_to_new_indices]).to(device)
val_idx = torch.tensor([old_to_new_indices[i.item()] for i in val_idx if i.item() in old_to_new_indices]).to(device)
test_idx = torch.tensor([old_to_new_indices[i.item()] for i in test_idx if i.item() in old_to_new_indices]).to(device)

X_train = torch.cat((num_properties[train_idx], cat_properties[train_idx], tweet_embeddings[train_idx], user_embeddings[train_idx]), dim=1)
y_train = labels[train_idx]

X_train_np = X_train.cpu().numpy()
y_train_np = y_train.cpu().numpy()

smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_np, y_train_np)

X_resampled_torch = torch.tensor(X_resampled, dtype=torch.float32).to(device)
y_resampled_torch = torch.tensor(y_resampled, dtype=torch.long).to(device)

num_prop_size = num_properties.shape[1]
cat_prop_size = cat_properties.shape[1]
tweet_embedding_size = tweet_embeddings.shape[1]
user_embedding_size = user_embeddings.shape[1]
user_name_embedding_size = user_name_embeddings.shape[1]
print(f'num property size: {num_prop_size}')
print(f'cat_prop_size: {cat_prop_size}')
print(f'tweet_embedding_size: {tweet_embedding_size}')
print(f'user_embedding_size: {user_embedding_size}')
print(f'user_name_embedding_size: {user_name_embedding_size}')

class CustomRGCN(nn.Module):
    def __init__(self, num_prop_size, cat_prop_size, tweet_embedding_size, user_embedding_size, user_name_embedding_size, embedding_dimension, dropout=0.3):
        super(CustomRGCN, self).__init__()
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.individual_size = int(embedding_dimension / 5)
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, self.individual_size),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, self.individual_size),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_embedding_size, self.individual_size),
            nn.LeakyReLU()
        )
        self.linear_relu_user_emb = nn.Sequential(
            nn.Linear(user_embedding_size, self.individual_size),
            nn.LeakyReLU()
        )
        self.linear_relu_user_name_emb = nn.Sequential(
            nn.Linear(user_name_embedding_size, self.individual_size),
            nn.LeakyReLU()
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(self.individual_size * 5, embedding_dimension),
            nn.LeakyReLU()
        )
        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output2 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output3 = nn.Linear(embedding_dimension, 2)
    def forward(self, num_prop, cat_prop, tweet_emb, user_emb, user_name_emb, edge_index, edge_type):
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        t = self.linear_relu_tweet(tweet_emb)
        u = self.linear_relu_user_emb(user_emb)
        u_name = self.linear_relu_user_name_emb(user_name_emb)
        x = torch.cat((n, c, t, u, u_name), dim=1)
        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_relu_output1(x)
        x = self.linear_relu_output2(x)
        x = self.linear_output3(x)
        return x

model = CustomRGCN(
    num_prop_size=num_prop_size,
    cat_prop_size=cat_prop_size,
    tweet_embedding_size=tweet_embedding_size,
    user_embedding_size=user_embedding_size,
    user_name_embedding_size=user_name_embedding_size,
    embedding_dimension=110,
    dropout=0.3
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=5e-2)
criterion = nn.CrossEntropyLoss()

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    return preds.eq(labels).double().sum() / len(labels)

def train(epoch):
    model.train()
    output = model(num_properties, cat_properties, tweet_embeddings, user_embeddings, user_name_embeddings, edge_index, edge_type)
    loss_train = criterion(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    tqdm.write(f'Epoch: {epoch+1} | Loss: {loss_train.item():.4f} | Accuracy: {acc_train:.4f}')

def test():
    model.eval()
    with torch.no_grad():
        output = model(num_properties, cat_properties, tweet_embeddings, user_embeddings, user_name_embeddings, edge_index, edge_type)
    loss_test = criterion(output[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])
    preds = output.max(1)[1].to('cpu').detach().numpy()
    true_labels = labels.to('cpu').detach().numpy()
    f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)
    precision = precision_score(true_labels, preds, average='weighted', zero_division=0)
    recall = recall_score(true_labels, preds, average='weighted', zero_division=0)
    fpr, tpr, thresholds = roc_curve(true_labels, preds, pos_label=1)
    auc_score = auc(fpr, tpr)
    print("\nTest set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test),
          "precision= {:.4f}".format(precision),
          "recall= {:.4f}".format(recall),
          "f1_score= {:.4f}".format(f1),
          "auc= {:.4f}".format(auc_score))
    cm = confusion_matrix(true_labels, preds)
    print(f"\nConfusion Matrix:\n{cm}")
    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=['Human', 'Bot'], zero_division=0))

model.apply(lambda m: nn.init.kaiming_uniform_(m.weight) if isinstance(m, nn.Linear) else None)

epochs = 1200
for epoch in tqdm(range(epochs), desc='Training Epochs'):
    train(epoch)

test()
torch.cuda.empty_cache()
