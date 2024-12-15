#data_utils.py
import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
import random
from nltk.corpus import wordnet  # Yêu cầu tải xuống NLTK WordNet với nltk.download('wordnet')
import logging
from sklearn.feature_extraction.text import TfidfVectorizer


logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Đọc file từ điển khía cạnh
def load_aspect_dict(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        aspect_keywords = f.read().splitlines()
    return set(aspect_keywords)

# Khởi tạo từ điển khía cạnh
aspect_dict = load_aspect_dict('aspect_dict.txt')

# Load PhoBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False, no_warning=True)

# Chuẩn bị TF-IDF vectorizer cho Naive Bayes
vectorizer = TfidfVectorizer(max_features=5000)  # Chọn số lượng đặc trưng tối đa

# Dataset cho PhoBERT
class ABSADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx, 0]
        aspect = self.data.iloc[idx, 1]
        sentiment = self.data.iloc[idx, 2]

        # Mã hóa câu và khía cạnh
        inputs = tokenizer.encode_plus(
            sentence, aspect, add_special_tokens=True, truncation=True, max_length=256, 
            padding="max_length", truncation_strategy="longest_first", return_tensors="pt"
        )
        
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        
        return input_ids, attention_mask, torch.tensor(sentiment)

# Thay thế từ đồng nghĩa để tạo câu neutral mới
def synonym_replacement(sentence):
    words = sentence.split()
    new_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            new_words.append(synonym)
        else:
            new_words.append(word)
    return ' '.join(new_words)

# Hàm để oversampling các lớp nhỏ hơn và tăng cường dữ liệu cho lớp neutral
def oversample_data(data):
    positive_samples = data[data['sentiment'] == 2]
    neutral_samples = data[data['sentiment'] == 1]
    negative_samples = data[data['sentiment'] == 0]
    
    max_samples = max(len(positive_samples), len(neutral_samples), len(negative_samples))
    
    # Oversampling cho lớp neutral và negative
    neutral_samples = resample(neutral_samples, replace=True, n_samples=max_samples, random_state=42)
    negative_samples = resample(negative_samples, replace=True, n_samples=max_samples, random_state=42)
    positive_samples = resample(positive_samples, replace=True, n_samples=max_samples, random_state=42)

    # Tăng cường dữ liệu cho lớp neutral
    augmented_neutral_samples = neutral_samples.copy()
    augmented_neutral_samples['sentence'] = augmented_neutral_samples['sentence'].apply(synonym_replacement)
    neutral_samples = pd.concat([neutral_samples, augmented_neutral_samples])
    
    # Kết hợp các lớp đã oversample
    data_balanced = pd.concat([positive_samples, neutral_samples, negative_samples])
    data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle dữ liệu
    
    return data_balanced

# Hàm trả về DataLoader cho training và testing
def get_data_loaders(batch_size=16):
    # Đọc dữ liệu từ các tệp CSV
    train_data = pd.read_csv('train_final_cleaned.tsv', sep='\t', header=None, names=['sentence', 'aspect', 'sentiment'])
    test_data = pd.read_csv('test_final_cleaned.tsv', sep='\t', header=None, names=['sentence', 'aspect', 'sentiment'])

    # Cân bằng dữ liệu bằng cách oversampling và tăng cường dữ liệu
    train_data_balanced = oversample_data(train_data)

    # Chuẩn bị dữ liệu cho Naive Bayes
    X_train_nb = vectorizer.fit_transform(train_data_balanced['sentence'])
    X_test_nb = vectorizer.transform(test_data['sentence'])
    y_train_nb = train_data_balanced['sentiment']
    y_test_nb = test_data['sentiment']

    # Tạo Dataset cho PhoBERT
    train_dataset = ABSADataset(train_data_balanced)
    test_dataset = ABSADataset(test_data)
    
    # Tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader, X_train_nb, X_test_nb, y_train_nb, y_test_nb
