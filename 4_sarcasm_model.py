import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

data_path = "sample_data.csv" 
df = pd.read_csv(data_path)
texts = df['Review'].tolist()
sentiments = df['Sarcasm_Label'].tolist()

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

train_texts, test_texts, train_sentiments, test_sentiments = train_test_split(texts, sentiments, test_size=0.2, random_state=42)

max_len = 512
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_len)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_len)

train_inputs = torch.tensor(train_encodings.input_ids)
train_masks = torch.tensor(train_encodings.attention_mask)
train_sentiments = torch.tensor(train_sentiments)

test_inputs = torch.tensor(test_encodings.input_ids)
test_masks = torch.tensor(test_encodings.attention_mask)
test_sentiments = torch.tensor(test_sentiments) # Fixed missing parenthesis here

train_dataset = TensorDataset(train_inputs, train_masks, train_sentiments)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataset = TensorDataset(test_inputs, test_masks, test_sentiments)
test_loader = DataLoader(test_dataset, batch_size=8)

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 4

print("Starting Sarcasm Detection Model Training...")
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
predicted_labels = []
true_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask) 
        _, predicted = torch.max(outputs.logits, 1)
        predicted_labels.extend(predicted.tolist())
        true_labels.extend(labels.tolist())

overall_accuracy = accuracy_score(true_labels, predicted_labels)
overall_precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
overall_recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
overall_f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

print("\n=== Overall Sarcasm Detection Evaluation ===")
print(f'Overall Accuracy: {overall_accuracy:.4f}')
print(f'Overall Precision: {overall_precision:.4f}')
print(f'Overall Recall: {overall_recall:.4f}')
print(f'Overall F1 Score: {overall_f1:.4f}')
