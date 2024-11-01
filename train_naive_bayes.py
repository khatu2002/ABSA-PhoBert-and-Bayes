from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Hàm huấn luyện Naive Bayes
def train_naive_bayes(X_train, y_train):
    model_nb = MultinomialNB()
    model_nb.fit(X_train, y_train)
    return model_nb

# Hàm đánh giá Naive Bayes
def evaluate_naive_bayes(model, X_test, y_test):
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=["negative", "neutral", "positive"])
    
    print(f"Naive Bayes Accuracy: {accuracy * 100:.2f}%")
    print(report)
    
    return accuracy
