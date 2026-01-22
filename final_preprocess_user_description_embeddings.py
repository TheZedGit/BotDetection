
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import pandas as pd

homePath = '/content/drive/MyDrive/Colab Notebooks/'
user = pd.read_json(homePath + 'user.json')
user_text = list(user['description'])
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-large-2022-154m")
model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-large-2022-154m").to('cuda')

batch_size = 8
all_embeddings = []

for i in tqdm(range(0, len(user_text), batch_size), desc="Processing embeddings"):
    batch_text = user_text[i:i+batch_size]
    inputs = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt").to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
    all_embeddings.append(embeddings)

all_embeddings_tensor = torch.cat(all_embeddings)
torch.save(all_embeddings_tensor, homePath + 'user_embeddings.pt')

print(f"Embeddings saved to {homePath + 'user_embeddings.pt'}")