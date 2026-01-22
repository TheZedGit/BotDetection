import pandas as pd
import torch
from tqdm import tqdm
import json
import ijson

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_data():
    edges_df = pd.read_csv('f:/Diploma/Dataset/edge-003.csv')
    labels_df = pd.read_csv('f:/Diploma/Dataset/TwiBot-22/label.csv')
    split_df = pd.read_csv('f:/Diploma/Dataset/TwiBot-22/split.csv')

    labels_df['label'] = labels_df['label'].map({'human': 0, 'bot': 1})
    labels_df = labels_df.merge(split_df, on='id')

    train_df = labels_df[labels_df['split'] == 'train']
    test_df = labels_df[labels_df['split'] == 'test']

    nodes = labels_df['id'].unique()
    
    node_id_map = {node_id: i for i, node_id in enumerate(tqdm(nodes, desc="Mapping Nodes"))}
    
    print("Extracting each user's tweets")
    author_ids = {}  
    for i in range(9):
        file_name = f'f:/Diploma/Dataset/tweet_{i}.json'
        with open(file_name, 'r') as f:
            parser = ijson.items(f, 'item')  
            for each in tqdm(parser, desc=f"Processing {file_name}", unit="tweet"):
                uid = 'u' + str(each['author_id'])
                tweet_id = each['id']
                if uid not in author_ids:
                    if uid not in node_id_map:
                        node_id_map[uid] = len(node_id_map)
                    author_ids[tweet_id] = node_id_map[uid]

    json.dump(author_ids, open('f:/Diploma/Dataset/tensors/author_ids.json', 'w'))

    edge_index = []
    edge_type = []

    with tqdm(total=edges_df.shape[0], desc="Processing Edges") as pbar:
        for _, row in edges_df.iterrows():
            if row['target_id'] not in node_id_map:
                node_id_map[row['target_id']] = len(node_id_map)

            if row['relation'] == 'follower':            
                edge_index.append([node_id_map[row['target_id']], node_id_map[row['source_id']]])
                edge_type.append(0)
            elif row['relation'] == 'following':
                edge_index.append([node_id_map[row['source_id']], node_id_map[row['target_id']]])
                edge_type.append(1)
            elif row['relation'] == 'mentioned' and row['source_id'] in author_ids:
                edge_index.append([author_ids[row['source_id']], node_id_map[row['target_id']]])
                edge_type.append(3)
            elif row['relation'] == 'retweeted' and row['source_id'] in author_ids and row['target_id'] in author_ids:
                edge_index.append([author_ids[row['source_id']], author_ids[row['target_id']]])
                edge_type.append(4)
            elif row['relation'] == 'quoted' and row['source_id'] in author_ids and row['target_id'] in author_ids:
                edge_index.append([author_ids[row['source_id']], author_ids[row['target_id']]])
                edge_type.append(5)
            elif row['relation'] == 'replied_to' and row['source_id'] in author_ids and row['target_id'] in author_ids:
                edge_index.append([author_ids[row['source_id']], author_ids[row['target_id']]])
                edge_type.append(6)
            pbar.update(1) 

    try:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    except ValueError as e:
        print(f"Error converting edge_index to tensor: {e}")
        print("Sample data from edge_index:", edge_index[:5])  
        return

    try:
        edge_type = torch.tensor(edge_type, dtype=torch.long).t()
    except ValueError as e:
        print(f"Error converting edge_index to tensor: {e}")
        print("Sample data from edge_index:", edge_type[:5])  
        return

    node_features = torch.ones((len(node_id_map), 1), dtype=torch.float32)

    y = torch.zeros(len(node_id_map), dtype=torch.long)
    for _, row in tqdm(train_df.iterrows(), desc="Assigning Labels to Nodes"):
        user_id = row['id']
        if user_id in node_id_map:
            y[node_id_map[user_id]] = row['label']
    
    base_path = 'f:/Diploma/Dataset/tensors/'
    torch.save(edge_index, f'{base_path}edge_index.pt')
    torch.save(edge_type, f'{base_path}edge_type.pt')
    torch.save(node_features, f'{base_path}node_features.pt')
    torch.save(y, f'{base_path}labels.pt')
    torch.save(node_id_map, f'{base_path}node_id_map.pt')
    
    torch.save(train_df, f'{base_path}train_df.pt')
    torch.save(test_df, f'{base_path}test_df.pt')

if __name__ == "__main__":
    preprocess_data()
