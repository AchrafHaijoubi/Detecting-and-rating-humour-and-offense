import pandas as pd
import spacy
from transformers import BertTokenizer, BertModel
import torch

df = pd.read_csv("dataset_annotated_balanced.csv")

# Load spaCy model for tokenization & lemmatization
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Tokenization and Lemmatization using spaCy"""
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])  # Remove stopwords

# Apply preprocessing
df["Joke_Cleaned"] = df["Joke"].astype(str).apply(preprocess_text)

# Load BERT model & tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def get_bert_embedding(text):
    """Extracts BERT embeddings for a given text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Mean pooling

# Apply embedding extraction
df["BERT_Embedding"] = df["Joke_Cleaned"].apply(get_bert_embedding)

# Save processed dataset
df.to_csv("processed_dataset.csv", index=False)

print(" Dataset processed and embeddings saved!")