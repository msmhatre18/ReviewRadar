# ReviewRadar 

This code is for a Streamlit app that classifies user-entered movie review as positive or negative. It loads a model from a pickle file and a vectorizer from another pickle file (both in Proper_Classification.ipynb), processes the user's input using a function that removes stopwords, lemmatizes the input, and then vectorizes it using the TfidfVectorizer. The model then makes a prediction on the vectorized input and displays the result in an output container.

(NLP Learning App).
