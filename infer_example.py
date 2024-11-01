import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# Đường dẫn tới mô hình và từ điển khía cạnh
MODEL_PATH = "state_dict/phobert-base_val_acc_0.8313.pth"
ASPECT_DICT_PATH = "aspect_dict.txt"

# Load từ điển khía cạnh
def load_aspect_dict(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        aspect_keywords = f.read().splitlines()
    return sorted(set(aspect.lower() for aspect in aspect_keywords), key=len, reverse=True)

# Load model và tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
model.load_state_dict(torch.load(MODEL_PATH))  # Tải trọng số từ file
model.eval()

# Hàm nhận diện và nhóm khía cạnh với phần mô tả liên quan (chỉ lấy khía cạnh đầu tiên trong mỗi phần)
def identify_aspects(sentence, aspect_dict):
    sentence = sentence.lower()
    for keyword in aspect_dict:
        # Tìm kiếm khía cạnh đầu tiên trong câu
        if re.search(rf'\b{keyword}\b', sentence):
            return {keyword: sentence}  # Trả về khía cạnh đầu tiên tìm thấy
    return {}

# Hàm dự đoán cảm xúc cho một khía cạnh
def predict_sentiment(sentence, aspect):
    inputs = tokenizer.encode_plus(
        f"Về khía cạnh {aspect}, {sentence}",
        add_special_tokens=True,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    sentiment = "Positive" if predicted_class == 2 else "Neutral" if predicted_class == 1 else "Negative"
    return sentiment

# Tách câu dựa trên các dấu phẩy và các từ nối để lấy phần mô tả từng khía cạnh
def split_sentence(sentence):
    splitters = [" nhưng ", " tuy nhiên ", " mặc dù ", " song ", ",", ";"]
    parts = [sentence]
    for splitter in splitters:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(splitter))
        parts = new_parts
    return [part.strip() for part in parts if part.strip()]

# Hàm chính để xử lý câu đầu vào
def main(sentence):
    aspect_dict = load_aspect_dict(ASPECT_DICT_PATH)
    split_words = [" nhưng ", " tuy nhiên ", " mặc dù ", " song ", ";", ","]
    
    # Kiểm tra xem câu có chứa các từ nối hay không
    if any(word in sentence.lower() for word in split_words):
        # Nếu có từ nối, tách câu thành các phần
        sentence_parts = split_sentence(sentence.lower())
        
        # Lấy khía cạnh đầu tiên của mỗi phần
        for part in sentence_parts:
            aspects = identify_aspects(part, aspect_dict)
            if aspects:
                main_aspect = list(aspects.keys())[0]
                main_phrase = aspects[main_aspect]
                sentiment = predict_sentiment(main_phrase, main_aspect)
                print(f"Extracted aspect: {main_aspect}")
                print(f"Combined phrases: {main_phrase}")
                print(f"Prediction for aspect ('{main_aspect}'): {sentiment}")

    else:
        # Nếu không có từ nối, chỉ lấy khía cạnh đầu tiên trong toàn bộ câu
        aspects = identify_aspects(sentence.lower(), aspect_dict)
        if aspects:
            main_aspect = list(aspects.keys())[0]
            main_phrase = aspects[main_aspect]
            sentiment = predict_sentiment(main_phrase, main_aspect)
            print(f"Extracted aspect: {main_aspect}")
            print(f"Combined phrases: {main_phrase}")
            print(f"Prediction for aspect ('{main_aspect}'): {sentiment}")

if __name__ == "__main__":
    sentence = " ".join(sys.argv[2:])
    main(sentence)
