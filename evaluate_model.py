from sklearn.metrics import accuracy_score, classification_report
import torch
import numpy as np

# Hàm đánh giá PhoBERT
def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # Lưu trữ xác suất dự đoán
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            probs = torch.softmax(outputs.logits, dim=-1)  # Lấy xác suất từ logits

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Đánh giá kết quả
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["negative", "neutral", "positive"])

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(report)

    return accuracy, np.concatenate(all_probs)  # Trả về xác suất dự đoán

# Hàm đánh giá Naive Bayes
def evaluate_naive_bayes(model_nb, X_test, y_test):
    preds_nb = model_nb.predict(X_test)
    accuracy = accuracy_score(y_test, preds_nb)
    report = classification_report(y_test, preds_nb, target_names=["negative", "neutral", "positive"])

    print(f"Naive Bayes Accuracy: {accuracy * 100:.2f}%")
    print(report)

    return accuracy

# Hàm đánh giá kết hợp giữa PhoBERT và Naive Bayes
def combined_evaluation(model_phobert, model_nb, test_loader, X_test_nb, y_test_nb):
    model_phobert.eval()
    all_preds_combined = []
    all_labels = []
    all_probs_phobert = []  # Lưu xác suất dự đoán của PhoBERT
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for idx, (input_ids, attention_mask, labels) in enumerate(test_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.cpu().numpy()
            all_labels.extend(labels)

            # PhoBERT dự đoán
            outputs = model_phobert(input_ids=input_ids, attention_mask=attention_mask)
            preds_phobert = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            probs_phobert = torch.softmax(outputs.logits, dim=-1).cpu().numpy()  # Lấy xác suất

            # Naive Bayes dự đoán cho các mẫu tương ứng
            preds_nb = model_nb.predict(X_test_nb[idx * len(labels):(idx + 1) * len(labels)])
            probs_nb = model_nb.predict_proba(X_test_nb[idx * len(labels):(idx + 1) * len(labels)])  # Dự đoán xác suất

            # Kết hợp dự đoán bằng weighted voting
            for i in range(len(preds_phobert)):
                weight_phobert = np.max(probs_phobert[i])  # Lấy xác suất cao nhất từ PhoBERT
                weight_nb = np.max(probs_nb[i])  # Lấy xác suất cao nhất từ Naive Bayes

                if weight_phobert > weight_nb:  # Chọn dự đoán của PhoBERT
                    all_preds_combined.append(preds_phobert[i])
                else:  # Ngược lại, dùng Naive Bayes
                    all_preds_combined.append(preds_nb[i])

    # Đánh giá kết quả kết hợp
    accuracy = accuracy_score(all_labels, all_preds_combined)
    report = classification_report(all_labels, all_preds_combined, target_names=["negative", "neutral", "positive"])

    print(f"Combined Model Accuracy: {accuracy * 100:.2f}%")
    print(report)

    return accuracy
