
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn
from torch.optim import AdamW
from data_utils import get_data_loaders
from evaluate_model import evaluate
from train_naive_bayes import train_naive_bayes, evaluate_naive_bayes
from evaluate_model import combined_evaluation
import os

# Định nghĩa Focal Loss với alpha cho từng lớp
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

# Khởi tạo PhoBERT và Naive Bayes
def initialize_models():
    # PhoBERT model
    model_phobert = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
    model_phobert.to(device)
    
    # Naive Bayes model
    model_nb = train_naive_bayes(X_train_nb, y_train_nb)
    
    return model_phobert, model_nb

# Huấn luyện PhoBERT với FocalLoss
def train_phobert(model, train_loader, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

# Huấn luyện mô hình kết hợp PhoBERT và Naive Bayes
def train_combined_model(train_loader, test_loader, X_test_nb, y_test_nb, epochs=10):
    model_phobert, model_nb = initialize_models()
    optimizer = AdamW(model_phobert.parameters(), lr=5e-6)
    criterion = FocalLoss(alpha=[0.5, 2.0, 1.0], gamma=2.5)

    best_accuracy_combined = 0
    model_name = model_phobert.config._name_or_path.split("/")[-1]
    
    for epoch in range(epochs):
        print(f"--- Epoch {epoch + 1} ---")
        
        # Huấn luyện PhoBERT
        train_phobert(model_phobert, train_loader, optimizer, criterion, epochs=1)
        
        # Đánh giá PhoBERT
        #accuracy_phobert = evaluate(model_phobert, test_loader)
        
        # Đánh giá mô hình kết hợp PhoBERT và Naive Bayes
        accuracy_combined = combined_evaluation(model_phobert, model_nb, test_loader, X_test_nb, y_test_nb)

        # Lưu mô hình nếu kết hợp có độ chính xác cao nhất
        if accuracy_combined > best_accuracy_combined:
            best_accuracy_combined = accuracy_combined
            best_model_state = model_phobert.state_dict()
            
            # Đảm bảo thư mục lưu trữ tồn tại
            if not os.path.exists('state_dict'):
                os.makedirs('state_dict', exist_ok=True)
                
            # Lưu PhoBERT với độ chính xác cao nhất trong mô hình kết hợp
            save_path = os.path.join('state_dict', f'combined_model_combined_acc_{round(best_accuracy_combined, 4)}.pth')
            torch.save(best_model_state, save_path)
            print(f"Best combined model saved at: {save_path}")

# Khởi tạo DataLoader và dữ liệu Naive Bayes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader, X_train_nb, X_test_nb, y_train_nb, y_test_nb = get_data_loaders()

# Khởi động quá trình huấn luyện tích hợp
train_combined_model(train_loader, test_loader, X_test_nb, y_test_nb, epochs=10)
