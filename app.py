# Step 01: Import Required Libraries
import streamlit  # Streamlit for interactive web app
import tensorflow
import keras

from tensorflow.keras.datasets import imdb  # IMDB dataset for sentiment analysis
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Padding for uniform sequence length
from tensorflow.keras.models import load_model  # Loading pre-trained deep learning models
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense  # Neural network layers
from pathlib import Path


# Step 02: Load IMDB Word Index
# Retrieve dictionary mapping words to numerical indices from the IMDB dataset
word_index = imdb.get_word_index()

# Reverse the mapping to allow lookup from indices back to words
# using list comprehension then converts it to a dictionary
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# using dist comprehension to reverse the word index mapping
# reverse_word_index = {value: key for key, value in word_index.items()}

# Step 03: Load Pre-Trained Model
# Load the saved RNN model that has been trained on IMDB movie reviews
model_path = model_path = Path("model.keras").resolve()
model = keras.models.load_model(model_path)

# Step 04: Utility Functions
# '''
# def decode_review(encoded_review):
#     """
#     Convert encoded review (list of word indices) back into readable text.
#     If a word is not found in reverse_word_index, it replaces it with '?'.
#     """
#     return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])  # Adjusting for reserved tokens (offset by -3)
# '''

def pre_process_input(user_input):
    """
    Preprocesses user input for deep learning model.
    
    Steps:
    1. Convert text to lowercase and split into words.
    2. Convert words into numerical indices using 'word_index'.
    3. Apply padding to ensure the sequence length is consistent (600 tokens).
    """
    words = user_input.lower().split()  # Convert text to lowercase and tokenize
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # Convert words to indices (+3 for reserved tokens)
    padded_review = pad_sequences([encoded_review], maxlen=600)  # Ensure fixed-length input for deep learning model
    return padded_review

def predict_statement(review):
    """
    Takes a user review as input, processes it, and predicts sentiment using the trained RNN model.
    
    - If sentiment prediction > 0.5, it is classified as 'Positive'.
    - If sentiment prediction ≤ 0.5, it is classified as 'Negative'.
    """
    processed_input = pre_process_input(review)  # Convert raw text into numerical format
    prediction = model.predict(processed_input)  # Get model prediction probability (0-1)
    statement = 'Positive' if prediction > 0.5 else 'Negative'  # Threshold-based sentiment classification
    return statement, prediction[0][0]  # Return classification label and sentiment probability

# Step 05: Streamlit Web App
streamlit.title('Movie Review Sentiment Analysis using RNN')
streamlit.subheader('Predict A movie Review (Positive? / Negative?)')

# User Input Section
user_input = streamlit.text_area(label="Enter a movie review:")

# Prediction Button
if streamlit.button(label='Predict Sentiment'):
    # Ensure user input exists before making a prediction
    if user_input.strip():
        statement, score = predict_statement(user_input)  # Get prediction results
        
        # Display results in Streamlit app
        streamlit.write(f'**Review:** {user_input}')
        streamlit.write(f'**Predicted Sentiment:** {statement}')  # Positive or Negative classification
        streamlit.write(f'**Confidence Score:** {score:.4f}')  # Probability score formatted to 4 decimal places
    else:
        streamlit.write("⚠️ Please enter a review before predicting.")
