import streamlit as st
import pandas as pd
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model  # Changed this import

def text_prediction(m, model):
    message = [str(m)]
    num_classes = 5
    embed_num_dims = 300
    max_seq_len = 500
    class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(message)
    
    seq = tokenizer.texts_to_sequences(message)
    padded = pad_sequences(seq, maxlen=max_seq_len)
    
    pred = model.predict(padded)
    return class_names[np.argmax(pred)]

def main():
    st.title("©️CLASSIFICATION OF TEXT INTO EMOTION STATES😃")
    message = st.text_input("Give a Sentence")
    
    loaded_model = load_model(r'https://github.com/basavaadarsh5/nlp-/raw/main/cnn_w2v.h5') # Load the Keras model
    
    if st.button("THE EMOTION STATE IS "):
        result = text_prediction(message, loaded_model)
        st.success(result)

if __name__ == '__main__':
    main()
