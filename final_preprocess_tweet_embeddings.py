import torch
import os
from transformers import AutoModel

homePath = '/content/drive/MyDrive/Colab Notebooks/'
model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-large-2022-154m")
hidden_size = model.config.hidden_size

def load_checkpoint():
    checkpoint_path = homePath + "tweets_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        print("Loading from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        return checkpoint['tweets_list'], checkpoint['last_processed_user']
    return [], None

tweets_list, last_processed_user = load_checkpoint()

incorrect_size_indices = [idx for idx, tensor in enumerate(tweets_list) if tensor.numel() == 768]

print(f"Found {len(incorrect_size_indices)} tensors with incorrect size.")

for idx in incorrect_size_indices:
    tweets_list[idx] = torch.zeros(hidden_size)

for tensor in tweets_list:
    assert tensor.numel() == hidden_size, "Found a tensor with incorrect size."

tweet_tensor = torch.stack(tweets_list)

path = homePath + "tweets_tensor.pt"
torch.save(tweet_tensor, path)
print('Finished')