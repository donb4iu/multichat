from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embedding(sentence):
    # Tokenize sentence and get input tensors
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
    
    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get the embeddings for the [CLS] token
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return embedding

# Define sentences
sentence1 = "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data, without being explicitly programmed."
sentence2 = "Artificial intelligence includes machine learning where statistical techniques are used to enable computers to learn from data and make decisions without being explicitly coded."

# Get embeddings for the sentences
embedding1 = get_embedding(sentence1)
embedding2 = get_embedding(sentence2)


# Compute cosine similarity
similarity1 = cosine_similarity(embedding1, embedding2)

print(f"Cosine similarity between similar sentences: {similarity1[0][0]:.4f}")