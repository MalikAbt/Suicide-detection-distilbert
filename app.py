import streamlit as st
import numpy as np
from keras.models import load_model
from keras.layers import TextVectorization
import tensorflow as tf

@st.cache_resource
def load_my_model():
    model = load_model("model/Suicide_detection_model.h5")
    return model

@st.cache_resource
def load_vocab(filepath="vocab.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        unique_vocab = [line.strip() for line in f.readlines()]
        print(f"Taille du vocab chargé : {len(unique_vocab)}")
    return unique_vocab

def load_vectorizer(vocab_filepath="vocab.txt", max_length=50):
    unique_vocab = load_vocab(vocab_filepath)
    vectorize_layer = TextVectorization(
        output_mode='int',
        output_sequence_length=max_length,
        vocabulary=unique_vocab
    )
    return vectorize_layer


model = load_my_model()
vectorize_layer = load_vectorizer()

st.title("Détection de texte suicidaire")

user_input = st.text_area("Entrez un texte à analyser", height=150)

if st.button("Analyser"):
    if user_input.strip() == "":
        st.warning("Veuillez entrer un texte.")
    else:
        # Transformer le texte en séquence
        input_seq = vectorize_layer(tf.constant([user_input]))
        # Faire la prédiction
        prediction = model.predict(input_seq, verbose=0)
        score = float(prediction[0][0])
        label = "suicidaire" if score > 0.5 else "non suicidaire"
        st.write(f"**Prédiction:** {label} (score: {score:.3f})")
