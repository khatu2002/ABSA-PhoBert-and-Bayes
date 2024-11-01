from sklearn.metrics import accuracy_score, classification_report
import torch

# Hàm đánh giá PhoBERT
def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = attention_mask.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Đánh giá kết quả
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["negative", "neutral", "positive"])
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(report)
    
    return accuracy  # Trả về độ chính xác để sử dụng trong huấn luyện

# Hàm đánh giá Naive Bayes
def evaluate_naive_bayes(model_nb, X_test, y_test):
    preds_nb = model_nb.predict(X_test)
    accuracy = accuracy_score(y_test, preds_nb)
    report = classification_report(y_test, preds_nb, target_names=["negative", "neutral", "positive"])

    print(f"Naive Bayes Accuracy: {accuracy * 100:.2f}%")
    print(report)

    return accuracy

# Hàm đánh giá kết hợp giữa PhoBERT và Naive Bayes
def combined_evaluation(model_phobert, model_nb, test_loader, X_test_nb, y_test_nb, threshold=0.7):
    model_phobert.eval()
    all_preds_combined = []
    all_labels = []

    with torch.no_grad():
        for idx, (input_ids, attention_mask, labels) in enumerate(test_loader):
            input_ids = input_ids.to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = attention_mask.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.cpu().numpy()
            all_labels.extend(labels)

            # PhoBERT dự đoán
            outputs = model_phobert(input_ids=input_ids, attention_mask=attention_mask)
            preds_phobert = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            # Naive Bayes dự đoán cho các mẫu tương ứng
            preds_nb = model_nb.predict(X_test_nb[idx * len(labels):(idx + 1) * len(labels)])

            # Kết hợp dự đoán
            for i in range(len(preds_phobert)):
                if outputs.logits[i, preds_phobert[i]] > threshold:  # Chọn PhoBERT nếu chắc chắn trên ngưỡng
                    all_preds_combined.append(preds_phobert[i])
                else:  # Ngược lại, dùng Naive Bayes
                    all_preds_combined.append(preds_nb[i])

    # Đánh giá kết quả kết hợp
    accuracy = accuracy_score(all_labels, all_preds_combined)
    report = classification_report(all_labels, all_preds_combined, target_names=["negative", "neutral", "positive"])

    print(f"Combined Model Accuracy: {accuracy * 100:.2f}%")
    print(report)

    return accuracy
