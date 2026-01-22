import pandas as pd
import torch
from tqdm import tqdm
import ijson
import json

print('loading raw data')
path='f:/Diploma/Dataset/'
user=pd.read_json(path+'Twibot-22/user.json')
edge=pd.read_csv(path+'edge-003.csv')
user_idx=user['id']
uid_index={uid:index for index,uid in enumerate(user_idx.values)}
user_index_to_uid = list(user.id)
uid_to_user_index = {x : i for i, x in enumerate(user_index_to_uid)}


print("extracting each_user's tweets")
id_tweet={i:[] for i in range(len(user_idx))}
for i in range(9):
    name='tweet_'+str(i)+'.json'
    user_tweets = ijson.items(open(path + name, 'r'), 'item')
    for each in user_tweets:
        uid='u'+str(each['author_id'])
        text=each['text']
        try:
            index=uid_index[uid]
            id_tweet[index].append(text)
        except KeyError:
            continue
json.dump(id_tweet,open(path+'processed_data/id_tweet.json','w'))