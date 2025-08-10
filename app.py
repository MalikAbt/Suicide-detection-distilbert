import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("model/Suicide_detection_model_Bert")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

st.title("Détection de texte suicidaire")

user_input = st.text_area("Entrez un texte à analyser", height=150)

if st.button("Analyser"):
    if user_input.strip() == "":
        st.warning("Veuillez entrer un texte.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            score = float(probs[0][1])  # probabilité de la classe suicidaire
            label = "suicidaire" if score > 0.5 else "non suicidaire"
            st.write(f"**Prédiction:** {label} (score: {score:.3f})")
