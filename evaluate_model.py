from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import torch

# Đánh giá mô hình PhoBERT
def evaluate_phobert(model, test_loader):
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

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["negative", "neutral", "positive"])
    
    print(f"PhoBERT Accuracy: {accuracy * 100:.2f}%")
    print(report)
    
    return accuracy, all_preds  # Return predictions for use in weighted averaging

# Đánh giá bằng Naive Bayes
def evaluate_naive_bayes(texts, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    clf = MultinomialNB()
    clf.fit(X, labels)

    predictions = clf.predict(X)
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, target_names=["negative", "neutral", "positive"])
    
    print(f"Naive Bayes Accuracy: {accuracy * 100:.2f}%")
    print(report)
    
    return accuracy, predictions  # Return predictions for use in weighted averaging

# Đánh giá mô hình kết hợp
def evaluate_combined_model(phobert_model, nb_model, vectorizer, test_loader, texts, labels, weight_phobert=0.7, weight_nb=0.3):
    # Đánh giá PhoBERT
    phobert_accuracy, phobert_preds = evaluate_phobert(phobert_model, test_loader)

    # Đánh giá Naive Bayes
    nb_accuracy, nb_preds = evaluate_naive_bayes(texts, labels)

    # Weighted Averaging
    combined_preds = []
    for i in range(len(phobert_preds)):
        combined_pred = (weight_phobert * phobert_preds[i] + weight_nb * nb_preds[i])
        combined_preds.append(combined_pred)

    combined_preds = [1 if p > 0.5 else 0 for p in combined_preds]  # Chuyển đổi sang nhãn
    combined_accuracy = accuracy_score(labels, combined_preds)
    report = classification_report(labels, combined_preds, target_names=["negative", "neutral", "positive"])

    print(f"Combined Model Accuracy: {combined_accuracy * 100:.2f}%")
    print(report)

    return combined_accuracy
