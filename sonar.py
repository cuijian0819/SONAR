import random
import numpy as np
import subprocess
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

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

import logging

from pdb import set_trace 

category_keywords = {
    "Data Privacy": ["leak", "breach", "privacy"],
    "Fraud/Phishing": ["fraud", "scam", "phishing"],
    "Ransomware/Malware": ["malware", "bot", "trojan"],
    "DoS/DDoS": ["botnet", "ddos", "dos", "denial", "distributed"],
    "Vulnerability": ["zero-day", "vulnerability", "cve"],
}

def keyword_finder(
    glove_model, category_keywords,
    alpha=0.2, beta=0.1,
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
            similar_words = glove_model.most_similar(seed_keyword, topn=10000)  
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


if __name__ == "__main__":
    ## TODO: read tweet data into a list
    security_tweets = [
        "Just attended an informative cybersecurity conference on the latest threats and vulnerabilities. #Cybersecurity #Event",
        "Just attended an informative cybersecurity conference on the latest threats and vulnerabilities. #Cybersecurity #Event",
        "Stay safe online, folks! Make sure your passwords are strong and regularly updated. #OnlineSecurity",
        "Breaking News: Data breach at XYZ Corp. Millions of user accounts compromised. #DataBreach",
        "There's a security drill happening at the office today. It's essential to be prepared for emergencies. #SafetyFirst",
        "There's a security drill happening at the office today. It's essential to be prepared for emergencies. #SafetyFirst",
        "Our IT team just patched a critical security vulnerability in our network. Keeping our systems secure. #NetworkSecurity",
        "In light of recent security threats, we're conducting a company-wide security training session next week. #SecurityAwareness",
        "Security update: Our new firewall system is now operational, providing enhanced protection against cyber threats. #Firewall",
        "The city is hosting a disaster preparedness workshop this weekend. Don't miss it if you're in town. #DisasterPrep",
        "The city is hosting a disaster preparedness workshop this weekend. Don't miss it if you're in town. #DisasterPrep",
        "Our security team is monitoring the network for any unusual activity. Vigilance is key to keeping our data safe. #NetworkMonitoring",
        "Stay informed about security best practices. It's everyone's responsibility to protect sensitive information. #InfoSec",
        "Security Event Alert: Join us at the annual Security Symposium next month for expert insights and solutions. #SecuritySymposium",
        "Security Event Alert: Join us at the annual Security Symposium next month for expert insights and solutions. #SecuritySymposium",
        "Safety tip: Always verify the identity of callers claiming to be from your bank or other organizations. #SafetyTips",
        "An important reminder: Secure your Wi-Fi network with a strong password and encryption. #WiFiSecurity",
        "The company's physical security measures have been enhanced to restrict access to authorized personnel only. #PhysicalSecurity",
        "Breaking: Major security incident reported at the city's financial district. Authorities are on the scene. #SecurityIncident",
        "Breaking: Major security incident reported at the city's financial district. Authorities are on the scene. #SecurityIncident",
        "Breaking: Major security incident reported at the city's financial district. Authorities are on the scene. #SecurityIncident",
        "Don't click on suspicious links or download attachments from unknown sources. Stay vigilant against phishing attempts. #Phishing",
        "Security announcement: The office will be closed tomorrow for a security upgrade. Stay safe and have a great day! #OfficeClosure",
        "Stay safe on the internet. Regularly update your antivirus software to protect against malware and viruses. #Antivirus",
        "Security alert: New malware detected. Ensure your devices have the latest antivirus definitions to stay protected. #Malware",
        "Security alert: New malware detected. Ensure your devices have the latest antivirus definitions to stay protected. #Malware",
        "It's a good time to review your disaster recovery plan. Preparation is the key to resilience in case of unexpected events. #DisasterRecovery",
        "Data breach at a major tech company, millions of user records exposed. #DataBreach #Security",
        "New cybersecurity vulnerability discovered in popular software. #Cybersecurity #Vulnerability",
        "Government agencies increase surveillance measures in response to security threats. #Surveillance #Security",
        "Organizations urged to update their security protocols following a recent cyber attack. #CyberAttack #Security",
        "Security experts warn about the rise in phishing attacks targeting businesses. #Phishing #Security",
        "Security conference happening this week, featuring talks on the latest threats and defenses. #SecurityConference",
    ]

    tokenized_tweets = [preprocess_tweet(tweet) for tweet in security_tweets]
    with open("data/tokenized_tweets.txt", "w", encoding="utf-8") as file:
        for tweet in tokenized_tweets:
            file.write(tweet + "\n")

    # Train GloVe model
    subprocess.call(['sh', './train_glove.sh']) # Thanks @Jim Dennis for suggesting the []
    
    # Load the GloVe model
    glove_model = load_glove_model()

    keyword_finder(glove_model, category_keywords)

    # Filter tweets with updated keywords
    all_keywords = flatten_dict_values(category_keywords)
    print(all_keywords)
    filtered_tweets = [doc for doc in security_tweets if any(keyword in doc.lower() for keyword in all_keywords)]
    print(f"# Tweets: {len(security_tweets)}, after keyword filtering: {len(filtered_tweets)}")
    
    '''
    Clustering part
    '''
    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocess_tweet)
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_tweets)

    # Create a NearPy LSH engine
    dimension = len(tfidf_vectorizer.get_feature_names_out())
    num_projections = 13  # Number of hyperplanes
    num_tables = 70  # Number of hash tables
    max_sim = 0.5  # Maximum distance for similarity
    engine = Engine(
        dimension, 
        lshashes=[RandomBinaryProjections('rbp', num_projections)], 
        vector_filters=[NearestFilter(num_tables)],
        )

    # Initialize variables
    threshold = 0.5
    inverted_index = defaultdict(list)
    recent_docs = []  # Store the most recent documents for comparison
    events = []

    # Iterate through the documents
    for doc_id, doc in enumerate(filtered_tweets):
        tfidf_vector = tfidf_matrix[doc_id].toarray()[0]  # Extract the TF-IDF vector for the current document
        engine.store_vector(tfidf_vector, doc_id)

        # Query the LSH engine to find similar documents
        neighbors = engine.neighbours(tfidf_matrix[doc_id].toarray()[0])

        # Set an initial minimum distance
        min_sim = 0

        # Check neighbors for similarity
        for _, neighbor_id, _ in neighbors:
            if neighbor_id != doc_id:  # Exclude self
                similarity = cosine_similarity(tfidf_matrix[doc_id], tfidf_matrix[neighbor_id])[0][0]
                if similarity > min_sim:
                    min_sim = similarity
                    nearest_neighbor_id = neighbor_id
        
        # This tweet can be attached to somewhere
        if min_sim > max_sim:
            # Look for similar documents in the past 1000 tweets
            for recent_doc_id in recent_docs[:1000]:
                similarity = cosine_similarity(tfidf_matrix[doc_id], tfidf_matrix[recent_doc_id])
                if similarity > threshold:
                    if len(events) == 0: 
                        events.append({doc_id, recent_doc_id})
                        continue
                    # Find the events that the current document and recent document belong to
                    for event in events:
                        added = False
                        if recent_doc_id in event: # added to existing cluster
                            added = True
                            event.add(doc_id)
                    if not added: # new one! 
                        events.append({doc_id, recent_doc_id})
        recent_docs.append(doc_id)

concatenated_documents = []
for tweet_set in events:
    sampled_indices = random.sample(list(tweet_set), min(5, len(tweet_set)))
    sampled_documents = [f"T{i}:{filtered_tweets[idx]}" for i, idx in enumerate(sampled_indices)]
    concatenated_documents.append('\n'.join(sampled_documents))

df = pd.DataFrame(concatenated_documents, columns=['events'])
df.to_excel('sonar_events.xlsx', index=False)  # Set index=False to exclude row numbers
