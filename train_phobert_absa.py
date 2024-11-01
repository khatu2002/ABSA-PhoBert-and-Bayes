from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn
import torch
from torch.optim import AdamW
from data_utils import get_data_loaders, load_texts_and_labels
from evaluate_model import evaluate_naive_bayes
from evaluate_model import evaluate_combined_model

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os

# PhoBERT Configurations
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
optimizer = AdamW(model.parameters(), lr=5e-6)

# Naive Bayes Configurations
texts, labels = load_texts_and_labels()
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)
nb_model = MultinomialNB().fit(X, labels)

# Focal Loss for PhoBERT
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha if alpha else [1, 1, 1]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        at = torch.tensor(self.alpha, device=inputs.device)[targets]
        focal_loss = at * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

criterion = FocalLoss(alpha=[0.7, 0.3, 1.0])

# Weighted Averaging Function
def weighted_average_prediction(phobert_logits, nb_logits, weight_phobert=0.7, weight_nb=0.3):
    combined_logits = (weight_phobert * phobert_logits) + (weight_nb * nb_logits)
    final_predictions = torch.argmax(combined_logits, dim=-1)
    return final_predictions

# Đánh giá mô hình kết hợp
def evaluate_combined_model(test_loader, model, nb_model, vectorizer):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            phobert_logits = outputs.logits.cpu()
            
            # Naive Bayes predictions
            texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids.cpu()]
            nb_features = vectorizer.transform(texts)
            nb_probs = nb_model.predict_proba(nb_features)
            nb_logits = torch.tensor(nb_probs, dtype=torch.float32)
            
            # Weighted Averaging
            final_preds = weighted_average_prediction(phobert_logits, nb_logits)
            all_preds.extend(final_preds.numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Weighted Averaging Accuracy: {accuracy:.4f}")
    return accuracy

# Training PhoBERT
train_loader, test_loader = get_data_loaders(batch_size=16)
best_accuracy = 0
best_model_state = None

for epoch in range(5):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in train_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    
    # Evaluate combined model
    phobert_accuracy = evaluate_combined_model(test_loader, model, nb_model, vectorizer)

    # Save state dict if best accuracy is achieved
    if phobert_accuracy > best_accuracy:
        best_accuracy = phobert_accuracy
        best_model_state = model.state_dict()
        if not os.path.exists('state_dict'):
            os.makedirs('state_dict', exist_ok=True)
        save_path = os.path.join('state_dict', f'phobert_val_acc_{round(best_accuracy, 4)}.pth')
        torch.save(best_model_state, save_path)
        print(f"Best model saved at: {save_path}")
