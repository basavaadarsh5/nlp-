%%writefile app01.py
import pandas as pd
import numpy as np
import pickle
import time
import streamlit as st

# text preprocessing
from nltk.tokenize import word_tokenize
import re

# plots
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# preparing input to the  model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# keras layers
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

loaded_model= pickle.load(open('/content/drive/MyDrive/LSTM_train_model.sav','rb'))



def text_prediction(m):
  message=[str(m)]


  num_classes = 5
  embed_num_dims = 300
  max_seq_len = 500

  class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']

  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(message)

  seq = tokenizer.texts_to_sequences(message)
  padded = pad_sequences(seq, maxlen=max_seq_len)


  pred = loaded_model.predict(padded)
  return "{}".format(class_names[np.argmax(pred)])


def main():
  st.title("©️CLASSIFICATION OF TEXT INTO EMOTION STATES😃")
  message=st.text_input("Give an Sentence")


  if st.button("THE EMOTION STATE IS "):
    result=text_prediction(message)
    st.success(result)

if __name__ == '__main__':
  main()

pip install streamlit -q

!streamlit run app01.py & npx localtunnel --port 8501

!pip install ultralytics


