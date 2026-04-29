import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Load the sample data we created
data_path = "sample_data.csv" 
df = pd.read_csv(data_path)
df = df.dropna()

df['Review'] = df['Review'].astype(str)
df['Review'] = df['Review'].apply(preprocess_text)

train_texts, test_texts = train_test_split(df['Review'].tolist(), test_size=0.2, random_state=42)

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
max_len = 512

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_len)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_len)

train_inputs = torch.tensor(train_encodings.input_ids)
train_masks = torch.tensor(train_encodings.attention_mask)
test_inputs = torch.tensor(test_encodings.input_ids)
test_masks = torch.tensor(test_encodings.attention_mask)

processed_train_df = pd.DataFrame({
    'input_ids': train_inputs.tolist(),
    'attention_mask': train_masks.tolist()
})
processed_train_df.to_csv("processed_train_data.csv", index=False)

processed_test_df = pd.DataFrame({
    'input_ids': test_inputs.tolist(),
    'attention_mask': test_masks.tolist()
})
processed_test_df.to_csv("processed_test_data.csv", index=False)
print("Text preprocessing completed.")
