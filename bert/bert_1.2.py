import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch.nn as nn

model_name = 'bert-base-chinese' 
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

model.classifier.dropout = nn.Dropout(p=0.5)

df = pd.read_excel('bert_train_data.xlsx')

train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

input_texts = train_df['內文'].tolist()

input_ids = []
attention_masks = []
max_length=256

for text in input_texts:
    encoded_dict = tokenizer.encode_plus(
                        text,                      
                        add_special_tokens=True,
                        max_length=max_length, 
                        truncation=True,
                        padding='max_length', 
                        return_attention_mask=True,
                        return_tensors='pt',
    )
        
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])


input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(df['label'].tolist())

batch_size = 16
dataset = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset, batch_size=batch_size)

model.train()

optimizer = AdamW(model.parameters(), lr=8e-4,eps=1e-8)
criterion = nn.CrossEntropyLoss()
num_train_epochs = 10

for epoch in range(num_train_epochs):
    for batch in dataloader:
        input_ids, attention_mask, label = batch
        input_ids = input_ids.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        attention_mask = attention_mask.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        label = label.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=label)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

model.eval()

eval_texts = eval_df['內文'].tolist()
eval_labels = torch.tensor(eval_df['label'].tolist())

eval_input_ids = []
eval_attention_masks = []

for text in eval_texts:
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )

    eval_input_ids.append(encoded_dict['input_ids'])
    eval_attention_masks.append(encoded_dict['attention_mask'])

eval_input_ids = torch.cat(eval_input_ids, dim=0)
eval_attention_masks = torch.cat(eval_attention_masks, dim=0)
eval_dataset = TensorDataset(eval_input_ids, eval_attention_masks, eval_labels)
eval_dataloader = DataLoader(eval_dataset, batch_size=16)

predicted_labels = []
true_labels = []

for batch in tqdm(eval_dataloader, desc="Predicting"):
    input_ids, attention_mask, label = batch
    input_ids = input_ids.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    attention_mask = attention_mask.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)

    predicted_label = torch.argmax(logits[0], dim=1).tolist()
    predicted_labels.extend(predicted_label)
    true_labels.extend(label.tolist())

accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")



