import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import add_self_loops
import torch.optim as optim
from sklearn.metrics import (
    precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from tqdm import tqdm
from imblearn.over_sampling import SMOTE

path = '/content/drive/MyDrive/Colab Notebooks/Second_Preprocess/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Loading processed data...')
labels = torch.load(path + 'labels.pt').to(device)
num_properties = torch.load(path + 'num_properties_tensor_updated.pt').to(device)
cat_properties = torch.load(path + 'cat_properties_tensor_updated.pt').to(device)

edge_index = torch.load(path + 'edge_index.pt').to(device)
edge_type  = torch.load(path + 'edge_type.pt').to(device)

labels = labels.long()
num_properties = num_properties.float()
cat_properties = cat_properties.float()
edge_index = edge_index.long()
edge_type = edge_type.long()

tweet_embeddings      = torch.load(path + 'tweets_tensor.pt').to(device).float()
user_embeddings       = torch.load(path + 'user_embeddings.pt').to(device).float()
user_name_embeddings  = torch.load(path + 'user_username_embeddings.pt').to(device).float()

train_idx = torch.load(path + 'train_indices.pt').to(device).long()
val_idx   = torch.load(path + 'val_indices.pt').to(device).long()
test_idx  = torch.load(path + 'test_indices.pt').to(device).long()

valid_indices = labels != -1
labels = labels[valid_indices]
num_properties = num_properties[valid_indices]
cat_properties = cat_properties[valid_indices]
tweet_embeddings = tweet_embeddings[valid_indices]
user_embeddings = user_embeddings[valid_indices]
user_name_embeddings = user_name_embeddings[valid_indices]

old_to_new = {
    int(old): new for new, old in enumerate(valid_indices.nonzero(as_tuple=True)[0].tolist())
}
train_idx = torch.tensor(
    [old_to_new[int(i)] for i in train_idx if int(i) in old_to_new],
    dtype=torch.long, device=device
)
val_idx = torch.tensor(
    [old_to_new[int(i)] for i in val_idx if int(i) in old_to_new],
    dtype=torch.long, device=device
)
test_idx = torch.tensor(
    [old_to_new[int(i)] for i in test_idx if int(i) in old_to_new],
    dtype=torch.long, device=device
)

X_train = torch.cat(
    (
        num_properties[train_idx],
        cat_properties[train_idx],
        tweet_embeddings[train_idx],
        user_embeddings[train_idx],
        user_name_embeddings[train_idx]
    ),
    dim=1
)
y_train = labels[train_idx]
X_train_np = X_train.cpu().numpy()
y_train_np = y_train.cpu().numpy()

smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_np, y_train_np)

X_resampled_torch = torch.tensor(X_resampled, dtype=torch.float32).to(device)
y_resampled_torch = torch.tensor(y_resampled, dtype=torch.long).to(device)

num_prop_size           = num_properties.shape[1]
cat_prop_size           = cat_properties.shape[1]
tweet_embedding_size    = tweet_embeddings.shape[1]
user_embedding_size     = user_embeddings.shape[1]
user_name_embedding_size = user_name_embeddings.shape[1]

print(f'num_prop_size: {num_prop_size}')
print(f'cat_prop_size: {cat_prop_size}')
print(f'tweet_embedding_size: {tweet_embedding_size}')
print(f'user_embedding_size: {user_embedding_size}')
print(f'user_name_embedding_size: {user_name_embedding_size}')

edge_index, edge_type = add_self_loops(
    edge_index=edge_index,
    edge_attr=edge_type,
    num_nodes=labels.size(0),
    fill_value=2
)

class CustomRGCN(nn.Module):
    def __init__(
        self,
        num_prop_size: int,
        cat_prop_size: int,
        tweet_embedding_size: int,
        user_embedding_size: int,
        user_name_embedding_size: int,
        embedding_dimension: int = 128,
        dropout: float = 0.3
    ):
        super(CustomRGCN, self).__init__()
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension

        assert embedding_dimension % 5 == 0, "embedding_dimension must be divisible by 5"
        self.individual_size = embedding_dimension // 5

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
            nn.Linear(5 * self.individual_size, embedding_dimension),
            nn.LeakyReLU()
        )

        self.rgcn1 = RGCNConv(
            in_channels=embedding_dimension,
            out_channels=embedding_dimension,
            num_relations=3
        )
        self.bn1 = nn.BatchNorm1d(embedding_dimension)

        self.rgcn2 = RGCNConv(
            in_channels=embedding_dimension,
            out_channels=embedding_dimension,
            num_relations=3
        )
        self.bn2 = nn.BatchNorm1d(embedding_dimension)
        
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output2 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output3 = nn.Linear(embedding_dimension, 2)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        num_prop: torch.Tensor,
        cat_prop: torch.Tensor,
        tweet_emb: torch.Tensor,
        user_emb: torch.Tensor,
        user_name_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        n = self.linear_relu_num_prop(num_prop)         
        c = self.linear_relu_cat_prop(cat_prop)         
        t = self.linear_relu_tweet(tweet_emb)           
        u = self.linear_relu_user_emb(user_emb)         
        u_name = self.linear_relu_user_name_emb(user_name_emb)  

        x = torch.cat((n, c, t, u, u_name), dim=1)
        x = self.linear_relu_input(x)           

        x = self.rgcn1(x, edge_index, edge_type)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.rgcn2(x, edge_index, edge_type)        
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.linear_relu_output1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_relu_output2(x)                  
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_output3(x)                       

        return x

model = CustomRGCN(
    num_prop_size=num_prop_size,
    cat_prop_size=cat_prop_size,
    tweet_embedding_size=tweet_embedding_size,
    user_embedding_size=user_embedding_size,
    user_name_embedding_size=user_name_embedding_size,
    embedding_dimension=55,
    dropout=0.3
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=5e-2)
criterion = nn.CrossEntropyLoss()

def accuracy(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds = output.argmax(dim=1).type_as(labels)
    return preds.eq(labels).double().sum() / labels.size(0)

def train(epoch: int):
    model.train()
    output = model(
        num_properties,
        cat_properties,
        tweet_embeddings,
        user_embeddings,
        user_name_embeddings,
        edge_index,
        edge_type
    )
    loss_train = criterion(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    tqdm.write(
        f'Epoch: {epoch+1:4d} | Loss: {loss_train.item():.4f} | Accuracy: {acc_train:.4f}'
    )

def test():
    model.eval()
    with torch.no_grad():
        output = model(
            num_properties,
            cat_properties,
            tweet_embeddings,
            user_embeddings,
            user_name_embeddings,
            edge_index,
            edge_type
        )
    loss_test = criterion(output[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])

    preds = output.argmax(dim=1).cpu().numpy()
    true_labels = labels.cpu().numpy()

    f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)
    precision = precision_score(true_labels, preds, average='weighted', zero_division=0)
    recall = recall_score(true_labels, preds, average='weighted', zero_division=0)
    fpr, tpr, _ = roc_curve(true_labels, preds, pos_label=1)
    auc_score = auc(fpr, tpr)

    print(
        "\nTest set results:",
        f"loss= {loss_test.item():.4f}",
        f"accuracy= {acc_test:.4f}",
        f"precision= {precision:.4f}",
        f"recall= {recall:.4f}",
        f"f1_score= {f1:.4f}",
        f"auc= {auc_score:.4f}"
    )

    cm = confusion_matrix(true_labels, preds)
    print(f"\nConfusion Matrix:\n{cm}")
    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=['Human', 'Bot'], zero_division=0))

epochs = 1500
for epoch in tqdm(range(epochs), desc='Training Epochs'):
    train(epoch)

test()
torch.cuda.empty_cache()
