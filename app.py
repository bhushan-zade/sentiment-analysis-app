import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load the model and vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Preprocess text (same as in Jupyter)
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Streamlit app interface
st.title("Sentiment Analysis Web App")
user_input = st.text_area("Enter a review for sentiment analysis:")

if st.button("Analyze"):
    if user_input:
        processed_input = preprocess_text(user_input)
        input_vector = vectorizer.transform([processed_input])
        prediction = model.predict(input_vector)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.write(f"The sentiment of the review is: **{sentiment}**")
    else:
        st.write("Please enter a review.")
