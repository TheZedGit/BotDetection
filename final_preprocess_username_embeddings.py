import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import pandas as pd

homePath = '/content/drive/MyDrive/Colab Notebooks/Second_Preprocess/'

user = pd.read_json(homePath + 'user.json')

username_series = user['username']
missing_usernames = username_series.isnull()

user_text = username_series.fillna('').astype(str).tolist()

torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-large-2022-154m")
model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-large-2022-154m").to('cuda')

batch_size = 8
all_embeddings = []

for i in tqdm(range(0, len(user_text), batch_size), desc="Processing embeddings"):
    batch_text = user_text[i:i+batch_size]
    batch_missing = missing_usernames.iloc[i:i+batch_size].values 

    non_missing_indices = [idx for idx, is_missing in enumerate(batch_missing) if not is_missing]
    non_missing_texts = [batch_text[idx] for idx in non_missing_indices]

    if non_missing_texts:
        inputs = tokenizer(non_missing_texts, padding=True, truncation=True, return_tensors="pt").to('cuda')

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
    else:
        embeddings = torch.empty((0, model.config.hidden_size)) 

    batch_embeddings = []

    embedding_idx = 0 

    for is_missing in batch_missing:
        if is_missing:
            zero_embedding = torch.zeros(model.config.hidden_size)
            batch_embeddings.append(zero_embedding)
        else:
            batch_embeddings.append(embeddings[embedding_idx])
            embedding_idx += 1

    batch_embeddings = torch.stack(batch_embeddings)
    all_embeddings.append(batch_embeddings)

all_embeddings_tensor = torch.cat(all_embeddings)

torch.save(all_embeddings_tensor, homePath + 'user_username_embeddings.pt')

print(f"Embeddings saved to {homePath + 'user_username_embeddings.pt'}")