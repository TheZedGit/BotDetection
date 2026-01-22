import torch
from tqdm import tqdm
import numpy as np
from transformers import pipeline
import os
import pandas as pd
import json
import ijson

homePath='f:/Diploma/Dataset/'
user=pd.read_json(homePath+'Twibot-22/user.json')

user_text=list(user['description'])
each_user_tweets=ijson.items(open(homePath+"processed_data/id_tweet.json",'r'), 'item')

feature_extract=pipeline('feature-extraction',model='roberta-large',tokenizer='roberta-large',device="cpu",padding=True, truncation=True,max_length=50, add_special_tokens = True)

def emb_user_description():
        print('Running feature1 embedding')
        path=homePath+"processed_data/emb_user_description.pt"
        if not os.path.exists(path):
            user_desc_vec=[]
            for k,each in enumerate(tqdm(user_text)):
                if each is None:
                    user_desc_vec.append(torch.zeros(768))
                else:
                    feature=torch.Tensor(feature_extract(each))
                    for (i,tensor) in enumerate(feature[0]):
                        if i==0:
                            feature_tensor=tensor
                        else:
                            feature_tensor+=tensor
                    feature_tensor/=feature.shape[1]
                    user_desc_vec.append(feature_tensor)
                    
            des_tensor=torch.stack(user_desc_vec,0)
            torch.save(des_tensor,path)
        else:
            des_tensor=torch.load(path)
        print('Finished')
        return des_tensor

def emb_tweets():
        print('Running feature2 embedding')
        path=homePath+"processed_data/tweets_tensor.pt"
        if True:
            tweets_list=[]
            for i in tqdm(range(len(each_user_tweets))):
                if len(each_user_tweets[str(i)])==0:
                    total_each_person_tweets=torch.zeros(768)
                else:
                    for j in range(len(each_user_tweets[str(i)])):
                        each_tweet=each_user_tweets[str(i)][j]
                        if each_tweet is None:
                            total_word_tensor=torch.zeros(768)
                        else:
                            each_tweet_tensor=torch.tensor(feature_extract(each_tweet))
                            for k,each_word_tensor in enumerate(each_tweet_tensor[0]):
                                if k==0:
                                    total_word_tensor=each_word_tensor
                                else:
                                    total_word_tensor+=each_word_tensor
                            total_word_tensor/=each_tweet_tensor.shape[1]
                        if j==0:
                            total_each_person_tweets=total_word_tensor
                        elif j==20:
                            break
                        else:
                            total_each_person_tweets+=total_word_tensor
                    if (j==20):
                        total_each_person_tweets/=20
                    else:
                        total_each_person_tweets/=len(each_user_tweets[str(i)])
                        
                tweets_list.append(total_each_person_tweets)
                    
            tweet_tensor=torch.stack(tweets_list)
            torch.save(tweet_tensor,homePath+"processed_data/tweets_tensor.pt")
                        
        else:
            tweets_tensor=torch.load(path)
        print('Finished')

emb_user_description()
emb_tweets()