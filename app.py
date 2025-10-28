from flask import Flask, request, render_template, jsonify
import pickle
from nltk.stem import SnowballStemmer
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

app = Flask(__name__)

# --- Load your saved model ---
with open('sentiment_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

negative_counts = model_data['negative_counts']
positive_counts = model_data['positive_counts']
prob_negative = model_data['prob_negative']
prob_positive = model_data['prob_positive']
negative_review_count = model_data['negative_review_count']
positive_review_count = model_data['positive_review_count']

# --- Preprocessing ---
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# --- Prediction Logic (same as notebook) ---
def make_class_prediction(text, word_counts, class_prob, total_count):
    words = text.split()
    word_probs = []
    for word in words:
        word_probs.append((word_counts.get(word, 0) + 1) / (total_count + len(word_counts)))
    final_prob = class_prob
    for p in word_probs:
        final_prob *= p
    return final_prob

def predict_sentiment(review_text):
    processed_text = preprocess_text(review_text)
    neg_pred = make_class_prediction(processed_text, negative_counts, prob_negative, negative_review_count)
    pos_pred = make_class_prediction(processed_text, positive_counts, prob_positive, positive_review_count)
    return "POSITIVE" if pos_pred > neg_pred else "NEGATIVE"

# --- Flask routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    sentiment = predict_sentiment(review)
    return render_template('index.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
