import torch
from tqdm import tqdm
import ijson
from transformers import AutoTokenizer, AutoModel
import os

homePath = '/content/drive/MyDrive/Colab Notebooks/'
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-large-2022-154m")
model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-large-2022-154m").to('cuda')

# Get the hidden size from the model configuration
hidden_size = model.config.hidden_size  # This will be 1024

# Load checkpoint if available
def load_checkpoint():
    checkpoint_path = homePath + "tweets_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        print("Loading from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        return checkpoint['tweets_list'], checkpoint['last_processed_user']
    return [], None

# Save checkpoint periodically
def save_checkpoint(tweets_list, last_processed_user):
    checkpoint_path = homePath + "tweets_checkpoint.pt"
    torch.save({
        'tweets_list': tweets_list,
        'last_processed_user': last_processed_user
    }, checkpoint_path)
    print(f"Checkpoint saved at user {last_processed_user}")

# Define the function to embed tweets
def tweets_embedding():
    print('Running feature2 embedding')
    path = homePath + "tweets_tensor.pt"

    # Load progress from checkpoint
    tweets_list, last_processed_user = load_checkpoint()

    # Use ijson to parse the JSON file iteratively
    with open(homePath + "id_tweet.json", 'r') as file:
        parser = ijson.kvitems(file, '')  # Stream key-value pairs from the file
        user_found = False if last_processed_user else True

        # Iterate over each user's tweets
        for user_id, tweets in tqdm(parser, desc="Processing users"):
            if not user_found:
                if user_id == last_processed_user:
                    user_found = True
                continue

            if len(tweets) == 0:
                total_each_person_tweets = torch.zeros(hidden_size)
            else:
                for j, tweet in enumerate(tweets):
                    if j == 20:  # Limit to 20 tweets per user
                        break

                    if tweet is None:
                        each_tweet_tensor = torch.zeros(hidden_size)
                    else:
                        # Tokenize and embed the tweet
                        inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True).to('cuda')
                        with torch.no_grad():
                            outputs = model(**inputs)
                        each_tweet_tensor = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()

                    if j == 0:
                        total_each_person_tweets = each_tweet_tensor
                    else:
                        total_each_person_tweets += each_tweet_tensor

                # Average the embeddings
                total_each_person_tweets /= min(20, len(tweets))

            tweets_list.append(total_each_person_tweets)

            # Save a checkpoint periodically
            if len(tweets_list) % 1000 == 0:
                save_checkpoint(tweets_list, user_id)
                torch.cuda.empty_cache()

    # Stack all user embeddings and save
    tweet_tensor = torch.stack(tweets_list)
    torch.save(tweet_tensor, path)
    print('Finished')

tweets_embedding()
