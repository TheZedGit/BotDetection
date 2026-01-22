import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import json
import ijson
from sklearn.preprocessing import LabelEncoder

base_path = 'f:/Diploma/Dataset/tensors/'  # Base path for saving tensors

def preprocess_data():
    # Load datasets
    print("Loading datasets...")
    edges_df = pd.read_csv('f:/Diploma/Dataset/edge-003.csv')
    labels_df = pd.read_csv('f:/Diploma/Dataset/TwiBot-22/label.csv')
    split_df = pd.read_csv('f:/Diploma/Dataset/TwiBot-22/split.csv')
    user_df = pd.read_json('f:/Diploma/Dataset/TwiBot-22/user.json')

    # Map labels to 0 and 1
    labels_df['label'] = labels_df['label'].map({'human': 0, 'bot': 1})
    labels_df = labels_df.merge(split_df, on='id')

    train_df = labels_df[labels_df['split'] == 'train']
    test_df = labels_df[labels_df['split'] == 'test']

    # Create a set of all unique node IDs from various sources
    nodes = set(labels_df['id'].unique())
    print(f"Total unique nodes: {len(nodes)}")

    # Map node IDs to indices
    print("Mapping Nodes...")
    node_id_map = {}
    for idx, node_id in enumerate(tqdm(nodes, desc="Mapping Nodes")):
        node_id_map[node_id] = idx

    # Process tweets to extract author IDs
    print("Extracting each user's tweets")
    author_ids = {}
    for i in tqdm(range(9), desc="Processing Tweet Files"):
        file_name = f'f:/Diploma/Dataset/tweet_{i}.json'
        with open(file_name, 'r') as f:
            parser = ijson.items(f, 'item')
            parser = tqdm(parser, desc=f"Processing {file_name}", unit=" tweets", leave=False)
            for each in parser:
                uid = each['author_id']
                tweet_id = each['id']
                if uid not in node_id_map:
                    node_id_map[uid] = len(node_id_map)
                author_ids[tweet_id] = node_id_map[uid]

    json.dump(author_ids, open(f'{base_path}author_ids.json', 'w'))

    # Build edge_index and edge_type
    edge_index = []
    edge_type = []

    print("Processing Edges...")
    with tqdm(total=edges_df.shape[0], desc="Processing Edges") as pbar:
        for _, row in edges_df.iterrows():
            source_id = row['source_id']
            target_id = row['target_id']
            relation = row['relation']

            # Ensure that all IDs are in node_id_map
            if source_id not in node_id_map:
                node_id_map[source_id] = len(node_id_map)
            if target_id not in node_id_map:
                node_id_map[target_id] = len(node_id_map)

            # Build edges based on relation type
            if relation == 'follower':
                edge_index.append([node_id_map[target_id], node_id_map[source_id]])
                edge_type.append(0)
            elif relation == 'following':
                edge_index.append([node_id_map[source_id], node_id_map[target_id]])
                edge_type.append(1)
            elif relation == 'mentioned' and source_id in author_ids:
                edge_index.append([author_ids[source_id], node_id_map[target_id]])
                edge_type.append(3)
            elif relation == 'retweeted' and source_id in author_ids and target_id in author_ids:
                edge_index.append([author_ids[source_id], author_ids[target_id]])
                edge_type.append(4)
            elif relation == 'quoted' and source_id in author_ids and target_id in author_ids:
                edge_index.append([author_ids[source_id], author_ids[target_id]])
                edge_type.append(5)
            elif relation == 'replied_to' and source_id in author_ids and target_id in author_ids:
                edge_index.append([author_ids[source_id], author_ids[target_id]])
                edge_type.append(6)
            pbar.update(1)

    # Convert edge_index and edge_type to tensors
    try:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    except ValueError as e:
        print(f"Error converting edge_index to tensor: {e}")
        print("Sample data from edge_index:", edge_index[:5])
        return

    try:
        edge_type = torch.tensor(edge_type, dtype=torch.long)
    except ValueError as e:
        print(f"Error converting edge_type to tensor: {e}")
        print("Sample data from edge_type:", edge_type[:5])
        return

    # Initialize node features tensor with zeros
    num_nodes = len(node_id_map)
    num_features = 9  # As per the number of features extracted from user data
    node_features = torch.zeros((num_nodes, num_features), dtype=torch.float32)

    # Assign labels to nodes
    y = torch.zeros(num_nodes, dtype=torch.long)
    print("Assigning Labels to Nodes...")
    for _, row in tqdm(labels_df.iterrows(), total=labels_df.shape[0], desc="Assigning Labels to Nodes"):
        user_id = row['id']
        if user_id in node_id_map:
            y[node_id_map[user_id]] = row['label']

    # Process user data and assign features
    print("Processing user data for features...")
    user_features_df = process_user_data(user_df)

    # Map user IDs to node indices and assign features
    print("Assigning user features to nodes...")
    for user_id in tqdm(user_features_df.index, total=user_features_df.shape[0], desc="Mapping User Features"):
        if user_id in node_id_map:
            node_idx = node_id_map[user_id]
            node_features[node_idx] = torch.tensor(user_features_df.loc[user_id].values, dtype=torch.float32)

    # Save all tensors and mappings
    print("Saving processed data...")
    torch.save(edge_index, f'{base_path}edge_index.pt')
    torch.save(edge_type, f'{base_path}edge_type.pt')
    torch.save(node_features, f'{base_path}node_features.pt')
    torch.save(y, f'{base_path}labels.pt')
    torch.save(node_id_map, f'{base_path}node_id_map.pt')

    torch.save(train_df, f'{base_path}train_df.pt')
    torch.save(test_df, f'{base_path}test_df.pt')

    print("Preprocessing complete and data saved.")

def process_user_data(user_df):
    # Initialize lists to store the extracted public_metrics fields
    followers_count = []
    following_count = []
    tweet_count = []
    listed_count = []

    print("Processing 'followers_count'...")
    for each in tqdm(enumerate(user_df['public_metrics']), total=len(user_df), desc="Processing Followers Count"):
        if each is not None and isinstance(each, dict) and each['followers_count'] is not None:
            followers_count.append(each['followers_count'])
        else:
            followers_count.append(0)

    # Manually standardize followers_count
    followers_count = pd.Series(followers_count)
    followers_count = (followers_count - followers_count.mean()) / followers_count.std()

    print("Processing 'following_count'...")
    for each in tqdm(enumerate(user_df['public_metrics']), total=len(user_df), desc="Processing Following Count"):
        if each is not None and isinstance(each, dict) and each['following_count'] is not None:
            following_count.append(each['following_count'])
        else:
            following_count.append(0)

    # Manually standardize following_count
    following_count = pd.Series(following_count)
    following_count = (following_count - following_count.mean()) / following_count.std()

    print("Processing 'tweet_count'...")
    for each in tqdm(enumerate(user_df['public_metrics']), total=len(user_df), desc="Processing Tweet Count"):
        if each is not None and isinstance(each, dict) and each['tweet_count'] is not None:
            tweet_count.append(each['tweet_count'])
        else:
            tweet_count.append(0)

    # Manually standardize tweet_count
    tweet_count = pd.Series(tweet_count)
    tweet_count = (tweet_count - tweet_count.mean()) / tweet_count.std()

    print("Processing 'listed_count'...")
    for each in tqdm(enumerate(user_df['public_metrics']), total=len(user_df), desc="Processing Listed Count"):
        if each is not None and isinstance(each, dict) and each['listed_count'] is not None:
            listed_count.append(each['listed_count'])
        else:
            listed_count.append(0)

    # Manually standardize listed_count
    listed_count = pd.Series(listed_count)
    listed_count = (listed_count - listed_count.mean()) / listed_count.std()

    # Process and encode other features
    print("Processing other features...")
    user_df['verified'] = user_df['verified'].fillna(False).astype(int)
    user_df['protected'] = user_df['protected'].fillna(False).astype(int)
    user_df['location'] = user_df['location'].fillna('Unknown')
    user_df['username'] = user_df['username'].fillna('Unknown')

    # Standardize 'created_at'
    print("Processing 'created_at'...")
    user_df['created_at'] = pd.to_datetime(user_df['created_at'].fillna('1970-01-01'))
    user_df['created_at'] = user_df['created_at'].astype('int64') / 10**9
    user_df['created_at'] = (user_df['created_at'] - user_df['created_at'].mean()) / user_df['created_at'].std()

    # Label encode categorical fields
    print("Encoding 'location' and 'username'...")
    location_encoder = LabelEncoder()
    user_df['location_encoded'] = location_encoder.fit_transform(user_df['location'].astype(str))
    username_encoder = LabelEncoder()
    user_df['username_encoded'] = username_encoder.fit_transform(user_df['username'].astype(str))

    # Create DataFrame of features
    print("Creating features DataFrame...")
    features_df = pd.DataFrame({
        'followers_count': followers_count.values,
        'following_count': following_count.values,
        'tweet_count': tweet_count.values,
        'listed_count': listed_count.values,
        'verified': user_df['verified'].values,
        'protected': user_df['protected'].values,
        'created_at': user_df['created_at'].values,
        'location_encoded': user_df['location_encoded'].values,
        'username_encoded': user_df['username_encoded'].values
    }, index=user_df['id'])

    # Fill any NaN values that may have resulted from standardization
    features_df = features_df.fillna(0)

    return features_df

if __name__ == "__main__":
    preprocess_data()
