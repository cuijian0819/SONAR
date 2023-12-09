import random
import numpy as np
import subprocess
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.filters import NearestFilter

import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

import torch
from torch.utils.data import TensorDataset, DataLoader, \
RandomSampler, SequentialSampler

from transformers import AutoTokenizer

import sys
sys.path.append('../')

from model.tweetclassifier import TweetClassifier
from utils.preprocess_text import preprocess_text
from utils.classification_utils import evaluate
import logging

from pdb import set_trace 


def is_dict_of_lists_empty(d):
    for key, value in d.items():
        if not value:  # If the list is empty, it will evaluate to False
            continue
        else:
            return False  # If any list is not empty, return False
    return True  # All lists in the dictionary are empty

def keyword_finder(
    glove_model, category_keywords,
    alpha=0.2, beta=0.3,
    ):

    categories = list(category_keywords.keys()) 

    doc_list = []
    for category, seed_keywords in tqdm(category_keywords.items()):
        keyword_set = set()
        neighbors = []
        # Find nearest neighbors for each seed keyword with threshold α
        for seed_keyword in seed_keywords:
            if seed_keyword not in glove_model:
                continue
            similar_words = glove_model.most_similar(seed_keyword, topn=100)  
            neighbors = [word for word, similarity in similar_words if similarity > alpha]
            
            # Create a list of neighbors that meet the threshold α
            keyword_set.update(neighbors)

        doc_list.append(' '.join(keyword_set))
    
    # Calculate TF-IDF for the list of keywords
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(doc_list)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Find keywords with TF-IDF scores greater than beta
    for i, category in enumerate(categories):
        tfidf_scores = tfidf_matrix[i].toarray()[0]
        selected_keywords = [feature_names[word_idx] for word_idx, score in enumerate(tfidf_scores) if score > beta] 
        category_keywords[category] = selected_keywords

def preprocess_tweet(tweet):
    # Tokenize the tweet
    words = word_tokenize(tweet)

    # Remove punctuation and convert to lowercase
    words = [word.lower() for word in words if word.isalnum()]

    # Lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

def load_glove_model():
    with open('data/glove_sec_tweets_50d.txt', 'r') as f:
        lines = f.readlines()

    # Get the number of words and vector dimension
    num_words = len(lines)
    vector_dim = len(lines[0].strip().split()) - 1

    # Prepend metadata
    lines.insert(0, f"{num_words} {vector_dim}\n")

    with open('data/glove_sec_tweets_50d.txt', 'w') as f:
        f.writelines(lines)
        
    glove_model = KeyedVectors.load_word2vec_format('data/glove_sec_tweets_50d.txt', binary=False)

    return glove_model

def flatten_dict_values(input_dict):
    flat_list = []
    for value in input_dict.values():
        flat_list.extend(value)
    return flat_list

def tag_category(filtered_tweets, device):

    lm = 'bertweet'
    rs = 1234

    print(f"classfier is on {device}")
    model_path = f'../trained_models/tweet_multi_cls_{lm}_{rs}.pt'

    model = TweetClassifier(lm)
    model.load_state_dict(torch.load(model_path,  map_location=device), strict=False)
    model.to(device)

    print('load bertweet tokenizer')
    tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
    
    X = filtered_tweets
    y = torch.tensor(([[0]*7 for _ in range(len(filtered_tweets))])) # dummy y 

    inputs = tokenizer(
        X, return_tensors="pt",
        padding=True, truncation=True)
    test_data = TensorDataset(
        y,
        inputs['input_ids'],
        inputs['attention_mask'],
        y,
    )
    test_sampler = SequentialSampler(test_data)

    test_dataloader = DataLoader(
        test_data,
        sampler=test_sampler,
        batch_size=4096)

    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]).to(device))         

    print("Computing event type...")
    logits_list, labels_list, input_idx_list, val_loss = evaluate(model, test_dataloader, loss_fn, device)

    logits_list = [logits.cpu() for logits in logits_list]
    labels_list = [labels.cpu() for labels in labels_list]
    preds_list = [torch.argmax(logits, dim=1).flatten() for logits in logits_list]
    preds_list = [preds.unsqueeze(dim=1) for preds in preds_list]
    preds_matrix = torch.cat(preds_list, dim=1)

    et_list = preds_matrix.tolist()

    return et_list
    
    
if __name__ == "__main__":

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    tweet_df = pd.read_excel('dummy_tweets.xlsx')
    # tweet_df = tweet_df.drop_duplicates(subset=['text'])
    tweet_df['created_at'] = pd.to_datetime(tweet_df['created_at'])

    start_date = datetime(2022, 11, 5)
    end_date = datetime(2022, 11, 30)
    one_day = timedelta(days=1)

    # Initialize the current date
    current_date = start_date
    concatenated_documents, uid_list, et_list = [], [], [] 
    start_list, end_list = [], []
    # Iterate through the dates in November
    while current_date <= end_date:
        category_keywords = {
            "Data Privacy": ["leak", "breach", "privacy"],
            "Fraud/Phishing": ["fraud", "scam", "phishing"],
            "Ransomware/Malware": ["malware", "bot", "trojan"],
            "DoS/DDoS": ["botnet", "ddos", "dos", "denial", "distributed"],
            "Vulnerability": ["zero-day", "vulnerability", "cve"],
        }
        print(current_date)
        tweets_cur = tweet_df.loc[
            (tweet_df['created_at']>= current_date) 
            & (tweet_df['created_at'] <= current_date + timedelta(days=1))
            ].reset_index(drop=True)
        security_tweets = tweets_cur['text'].tolist()
        tokenized_tweets = [preprocess_tweet(tweet) for tweet in security_tweets]
        with open("data/tokenized_tweets.txt", "w", encoding="utf-8") as file:
            for tweet in tokenized_tweets:
                file.write(tweet + "\n")

        # Train GloVe model
        subprocess.call(['sh', './train_glove.sh'],stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Load the GloVe model
        glove_model = load_glove_model()

        # print(category_keywords)
        keyword_finder(glove_model, category_keywords)
        if is_dict_of_lists_empty(category_keywords): 
            current_date += timedelta(days=1)
            continue
            
        # Filter tweets with updated keywords
        all_keywords = flatten_dict_values(category_keywords)

        filtered_tweets = [doc for doc in security_tweets if any(keyword in doc.lower() for keyword in all_keywords)]
        print(f"# Tweets: {len(security_tweets)}, after keyword filtering: {len(filtered_tweets)}")
        all_et_list = tag_category(filtered_tweets, device)

        if len(filtered_tweets)==0: continue;

        '''
        Clustering part
        '''
        # Initialize the TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocess_tweet)
        tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_tweets)

        # Create a NearPy LSH engine
        dimension = len(tfidf_vectorizer.get_feature_names_out())
        num_projections = 2  # Number of hyperplanes
        num_tables = 10  # Number of hash tables
        threshold = 0.2
        engine = Engine(
            dimension, 
            lshashes=[RandomBinaryProjections('rbp', num_projections)], 
            vector_filters=[NearestFilter(num_tables)],
            )

        # Initialize variables
        inverted_index = defaultdict(list)
        recent_docs = []  # Store the most recent documents for comparison
        events = []

        # Iterate through the documents
        for doc_id, doc in enumerate(filtered_tweets):
            tfidf_vector = tfidf_matrix[doc_id].toarray()[0]  # Extract the TF-IDF vector for the current document
            engine.store_vector(tfidf_vector, doc_id)
        
        # Get event
        doc_checked = set()
        for doc_id, doc in enumerate(filtered_tweets):
            if doc_id in doc_checked:
                continue
            doc_checked.add(doc_id)

            # Query the LSH engine to find similar documents
            neighbors = engine.neighbours(tfidf_matrix[doc_id].toarray()[0])

            event = set()
            # Check neighbors for similarity
            for _, neighbor_id, _ in neighbors:
                if neighbor_id != doc_id:  # Exclude self
                    similarity = cosine_similarity(tfidf_matrix[doc_id], tfidf_matrix[neighbor_id])[0][0]
                    if similarity > threshold and neighbor_id not in doc_checked:
                        doc_checked.add(neighbor_id)
                        event.add(neighbor_id)
            if len(event) > 5:
                events.append(event) 
                
            # # This tweet can be attached to somewhere
            # if min_sim > threshold:
            #     # Look for similar documents in the past 1000 tweets
            #     for recent_doc_id in recent_docs[:1000]:
            #         similarity = cosine_similarity(tfidf_matrix[doc_id], tfidf_matrix[recent_doc_id])
            #         if similarity > threshold:
            #             if len(events) == 0: 
            #                 events.append({doc_id, recent_doc_id})
            #                 continue
            #             # Find the events that the current document and recent document belong to
            #             for event in events:
            #                 added = False
            #                 if recent_doc_id in event: # added to existing cluster
            #                     added = True
            #                     event.add(doc_id)
            #             if not added: # new one! 
            #                 events.append({doc_id, recent_doc_id})
            # recent_docs.append(doc_id)

        # Wrapping up results
        
        for tweet_set in events:
            # uid_list.append([tid2uid[tid] for tid in tweet_set])
            # tid_list.append([tid for tid in tweet_set])
            et_list.append([all_et_list[tid] for tid in tweet_set])
            # score.append(len(uid_list)/len(tid_list))

            sampled_indices = random.sample(list(tweet_set), min(5, len(tweet_set)))
            sampled_documents = [f"T{i}:{filtered_tweets[idx]}" for i, idx in enumerate(sampled_indices)]
            concatenated_documents.append('\n'.join(sampled_documents))

        
        start_list.extend([current_date]*len(events))
        current_date += one_day
        end_list.extend([current_date]*len(events))

    start_list = [date.strftime('%Y-%m-%d %H:%M:%S') for date in start_list]
    end_list = [date.strftime('%Y-%m-%d %H:%M:%S') for date in end_list]

    df = pd.DataFrame({
        'start_date': start_list, 
        'endt_date': end_list, 
        # 'uid_list': uid_list,
        # 'tid_list': tid_list,
        'et_list': et_list,
        # 'score': score_list,
        'events': concatenated_documents,
        })
    df.to_excel('sonar_events.xlsx', index=False)  # Set index=False to exclude row numbers