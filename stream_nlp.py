import streamlit as st
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load tokenizer dan
tokenizer_path = "tokenizer.pkl"
model_path = "CNN_model_trial4.h5"

try:
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    model = load_model(model_path, compile=False)
except Exception as e:
    st.error(f"Error saat memuat model/tokenizer: {e}")
    tokenizer, model = None, None

# Prediksi Sentimen
def predict_sentiment(text):
    if not model or not tokenizer:
        return "error", 0

    sequence = tokenizer.texts_to_sequences([text])
    if not sequence[0]:
        return "unknown", 0

    padded_sequence = pad_sequences(sequence, maxlen=50)
    prediction = model.predict(padded_sequence)
    sentiment = np.argmax(prediction)
    labels = ["bad", "good", "neutral"]
    
    return labels[sentiment], prediction[0][sentiment]

# Streamlit UI
st.title("Analisis Sentimen CNN")
user_input = st.text_area("Masukkan Kalimat:")

if st.button("Prediksi") and user_input.strip():
    sentiment, confidence = predict_sentiment(user_input)
    if sentiment == "error":
        st.error("Model tidak tersedia.")
    elif sentiment == "unknown":
        st.warning("Kata tidak dikenali oleh tokenizer.")
    else:
        st.write(f"**Predicted Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {round(confidence * 100, 2)}%")
