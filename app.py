import streamlit as st
import numpy as np
import re
import pickle

st.title("ReviewRadar")
st.subheader("Classify user-entered film review as positive or negative")
# Load the model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

input_text = st.text_input("Enter input data:")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


def process_input(input_text):
    lemmatizer = WordNetLemmatizer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    
    input_text = re.sub('[^a-zA-Z]', ' ', input_text)
    input_text = input_text.lower()
    input_text = input_text.split()
    input_text = [lemmatizer.lemmatize(word) for word in input_text if word not in set(all_stopwords)]
    input_text = ' '.join(input_text)
    
    return [input_text]

input_text = process_input(input_text)
X_input = vectorizer.transform(input_text)
prediction = model.predict(X_input)


output_container = st.sidebar.empty()

if st.button("Classify Text"):
    # classify the text and update the output container
    output_container.text("Output: " + prediction)

