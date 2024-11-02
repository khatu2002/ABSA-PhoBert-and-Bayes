import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Hàm huấn luyện Naive Bayes
def train_naive_bayes(X_train, y_train):
    model_nb = MultinomialNB(alpha=1.0)  # Có thể thử điều chỉnh alpha
    model_nb.fit(X_train, y_train)
    return model_nb

# Hàm đánh giá Naive Bayes
def evaluate_naive_bayes(model, X_test, y_test):
    try:
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, target_names=["negative", "neutral", "positive"])
        
        print(f"Naive Bayes Accuracy: {accuracy * 100:.2f}%")
        print(report)
        
        return accuracy
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    try:
        # Đọc dữ liệu từ tệp CSV hoặc TSV
        train_data = pd.read_csv('train_final_cleaned.tsv', sep='\t', header=None, names=['sentence', 'aspect', 'sentiment'])
        test_data = pd.read_csv('test_final_cleaned.tsv', sep='\t', header=None, names=['sentence', 'aspect', 'sentiment'])
    except FileNotFoundError:
        print("Error: Data file not found. Please check the file paths.")
    
    # Chuyển đổi dữ liệu thành định dạng TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))  # Thử với bi-gram
    X_train = vectorizer.fit_transform(train_data['sentence'])
    X_test = vectorizer.transform(test_data['sentence'])
    y_train = train_data['sentiment']
    y_test = test_data['sentiment']

    # Huấn luyện mô hình Naive Bayes
    model_nb = train_naive_bayes(X_train, y_train)

    # Đánh giá mô hình
    evaluate_naive_bayes(model_nb, X_test, y_test)
