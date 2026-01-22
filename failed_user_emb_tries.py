import torch
from tqdm import tqdm
import numpy as np
from transformers import pipeline
import os
import pandas as pd
import json
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from torch.utils.data import DataLoader
from torch.amp import autocast
import ijson
from transformers import AutoTokenizer, AutoModel
import os

homePath='/content/drive/MyDrive/Colab Notebooks/'
user=pd.read_json(homePath+'user.json')

user_text=list(user['description'])
#each_user_tweets=json.load(open(homePath+"id_tweet.json",'r')).items()

feature_extract=pipeline('feature-extraction',model='roberta-large',tokenizer='roberta-base',device=0,padding=True, truncation=True,max_length=50, add_special_tokens = True, )
torch.cuda.empty_cache()

def emb_user_description_not_cool(batch_size=8):
    print('Running feature1 embedding')
    path = homePath + "emb_user_description.pt"

    if not os.path.exists(path):
        dataset = Dataset.from_dict({"description": user_text})

        def extract_features(batch):
            key_dataset = KeyDataset(dataset, "description")
            features = feature_extract(key_dataset, batch_size=batch_size)
            return {"embeddings": [torch.Tensor(f[0]).to('cuda').mean(dim=0).cpu().numpy() for f in features]}

        batched_dataset = dataset.map(extract_features, batched=True, batch_size=batch_size)

        user_desc_vec = torch.tensor(batched_dataset["embeddings"])
        torch.save(user_desc_vec, path)
    else:
        user_desc_vec = torch.load(path)

    print('Finished')


def emb_user_description(batch_size=8):
    print('Running feature1 embedding')
    path = homePath + "emb_user_description.pt"

    if not os.path.exists(path):
        dataset = Dataset.from_dict({"description": user_text})

        key_dataset = KeyDataset(dataset, "description")
        features = feature_extract(key_dataset, batch_size=batch_size)
        feature = {"embeddings": [torch.Tensor(f[0]).to('cuda').mean(dim=0).cpu().numpy() for f in tqdm(features, desc="Processing embeddings")]}

        user_desc_vec = torch.tensor(feature["embeddings"])

        torch.save(user_desc_vec, path)
    else:
        user_desc_vec = torch.load(path)

    print('Finished')

    from torch.utils.data import DataLoader

def emb_user_description_not_now(batch_size=512, num_workers=128):
    print('Running feature1 embedding')
    path = homePath + "emb_user_description.pt"

    if not os.path.exists(path):
        dataset = Dataset.from_dict({"description": user_text})
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        embeddings = []

        for batch in tqdm(dataloader, desc="Processing embeddings"):
            batch_descriptions = batch['description']

            features = feature_extract(batch_descriptions, batch_size=1024)
            
            batch_embeddings = [torch.Tensor(f[0]).to('cuda').mean(dim=0).cpu().numpy() for f in features]
            embeddings.extend(batch_embeddings)

        user_desc_vec = torch.tensor(embeddings)

        torch.save(user_desc_vec, path)
    else:
        user_desc_vec = torch.load(path)

    print('Finished')


def emb_user_description_not_now2(batch_size=8):
    print('Running feature1 embedding')
    path = homePath + "emb_user_description.pt"

    if not os.path.exists(path):
        dataset = Dataset.from_dict({"description": user_text})

        dataloader = DataLoader(dataset, batch_size=32)

        embeddings = []

        for batch in tqdm(dataloader, desc="Processing embeddings"):
            features = feature_extract(batch['description'], batch_size=batch_size)
            batch_embeddings = [torch.Tensor(f[0]).to('cuda').mean(dim=0).cpu().numpy() for f in features]
            embeddings.extend(batch_embeddings)

        user_desc_vec = torch.tensor(embeddings)
        torch.save(user_desc_vec, path)
    else:
        user_desc_vec = torch.load(path)

    print('Finished')

def emb_tweets_failed(batch_size=32):
    print('Running feature2 embedding')
    path = homePath + "tweets_tensor.pt"

    if not os.path.exists(path):
        user_ids, tweet_texts = [], []
        for user_id, tweets in each_user_tweets:
            if (len(tweets) > 20):
              tweets = tweets[:20]

            for tweet in tweets:
                user_ids.append(user_id)
                tweet_texts.append(tweet if tweet is not None else "")

        dataset = Dataset.from_dict({"user_id": user_ids, "tweet": tweet_texts})   #here google colab session is crashing as it uses 49,8gb/50gb available vram

        def extract_tweet_features(batch):
            tweets = batch["tweet"]
            features = feature_extract(tweets, batch_size=batch_size)
            return {"embeddings": [torch.Tensor(f[0]).device(0).mean(dim=0).cpu().numpy() for f in features]}


        batched_dataset = dataset.map(extract_tweet_features, batched=True, batch_size=batch_size)

        tweet_embeddings = {}
        for user_id, embedding in zip(batched_dataset["user_id"], batched_dataset["embeddings"]):
            if user_id not in tweet_embeddings:
                tweet_embeddings[user_id] = []
            tweet_embeddings[user_id].append(torch.tensor(embedding))

        tweets_list = []
        for user_id in sorted(tweet_embeddings.keys(), key=lambda x: int(x)):
            tweets_list.append(torch.mean(torch.stack(tweet_embeddings[user_id]), dim=0))
        tweet_tensor = torch.stack(tweets_list)
        torch.save(tweet_tensor, path)
    else:
        tweet_tensor = torch.load(path)

    print('Finished')


def emb_tweets(batch_size=8, chunk_size=10000):
    print('Running feature2 embedding')
    path = homePath + "tweets_tensor.pt"

    if not os.path.exists(path):
        tweet_embeddings = {}
        tweet_tensor_list = []

        user_ids, tweet_texts = [], []
        count = 0

        for user_id, tweets in each_user_tweets:
            if len(tweets) > 20:
                tweets = tweets[:20]

            for tweet in tweets:
                user_ids.append(user_id)
                tweet_texts.append(tweet if tweet is not None else "")

                count += 1
                if count % chunk_size == 0:
                    process_chunk(user_ids, tweet_texts, tweet_embeddings, batch_size)
                    user_ids, tweet_texts = [], []

        if user_ids and tweet_texts:
            process_chunk(user_ids, tweet_texts, tweet_embeddings, batch_size)

        for user_id in sorted(tweet_embeddings.keys(), key=lambda x: int(x)):
            tweet_tensor_list.append(torch.mean(torch.stack(tweet_embeddings[user_id]), dim=0))

        tweet_tensor = torch.stack(tweet_tensor_list)
        torch.save(tweet_tensor, path)
    else:
        tweet_tensor = torch.load(path)

    print('Finished')


def process_chunk(user_ids, tweet_texts, tweet_embeddings, batch_size):
    from datasets import Dataset

    dataset = Dataset.from_dict({"user_id": user_ids, "tweet": tweet_texts})

    def extract_tweet_features(batch):
        tweets = batch["tweet"]
        features = feature_extract(tweets, batch_size=batch_size)
        return {"embeddings": [torch.Tensor(f[0]).to('cuda').mean(dim=0).cpu().numpy() for f in features]}

    batched_dataset = dataset.map(extract_tweet_features, batched=True, batch_size=batch_size)

    for user_id, embedding in zip(batched_dataset["user_id"], batched_dataset["embeddings"]):
        if user_id not in tweet_embeddings:
            tweet_embeddings[user_id] = []
        tweet_embeddings[user_id].append(torch.tensor(embedding))


#emb_user_description()
emb_tweets()
