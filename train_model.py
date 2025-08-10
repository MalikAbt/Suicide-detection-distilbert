import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Embedding, LSTM, Dense, Flatten, Bidirectional, Dropout
from keras.layers import TextVectorization
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments,Trainer
import os
import re

# ==========================
# Paramètres
# ==========================
vocab_size = 15000
embedding_dim = 100
max_length = 50

# ==========================
# Chargement des données
# ==========================
texts = pd.read_csv('Data/Suicide_Detection.csv')
texts = texts.dropna(subset=['text'])
texts['labels'] = texts['class'].map({'non-suicide': 0, 'suicide': 1})

print(texts['labels'].value_counts(normalize=True))

# ==========================
# Nettoyage du texte
# ==========================
emoji_pattern = re.compile(
    "["
    u"\U0001F600-\U0001F64F"  
    u"\U0001F300-\U0001F5FF"  
    u"\U0001F680-\U0001F6FF"  
    u"\U0001F1E0-\U0001F1FF"  
    u"\U00002700-\U000027BF"  
    u"\U0001F900-\U0001F9FF"  
    u"\U00002600-\U000026FF"  
    "]+", flags=re.UNICODE
)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)           
    text = emoji_pattern.sub(r'', text)           
    text = re.sub(r'[^a-z\s]', '', text)          
    text = re.sub(r'\s+', ' ', text).strip()      
    return text

texts['text'] = texts['text'].apply(clean_text)

# ==========================
# Vectorisation
# ==========================
vectorize_layer = TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=max_length
)

vectorize_layer.adapt(texts['text'])

X = vectorize_layer(texts['text'])
y = texts['labels'].values

# ==========================
# Split train/val/test       F1_score
# ==========================
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# ==========================
# Poids de classe
# ==========================
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))


# ==========================
# Fine Tuning Bert  
# ==========================

model_path = 'Bert'

tokenizer = AutoTokenizer.from_pretrained(model_path)

id2label = {0: "suicide", 1: "non-suicide"}
label2id = {"suicide" :0 , "non-suicide": 1}




# ==========================
# Embedding GloVe
# ==========================
embedding_index = {}
with open('glove.6B.100d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

vocab = vectorize_layer.get_vocabulary()
embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))
for i, word in enumerate(vocab):
    if i == 0:
        continue
    if i <= vocab_size:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# ==========================
# Modèle
# ==========================
model = Sequential()
model.add(Embedding(input_dim=vocab_size + 1,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_length,
                    trainable=False))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ==========================
# Entraînement
# ==========================
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=5,
          batch_size=32,
          class_weight=class_weights_dict,
          callbacks=[early_stop])

# ==========================
# Évaluation sur le test set pour f1_score
# ==========================
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred_classes)
prec = precision_score(y_test, y_pred_classes)
rec = recall_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes)

print(f"Accuracy: {acc:.4f}")
print(f"Précision: {prec:.4f}")
print(f"Rappel: {rec:.4f}")
print(f"F1-score: {f1:.4f}")

# ==========================
# Sauvegarde du modèle
# ==========================
os.makedirs("model", exist_ok=True)
model.save("model/Suicide_detection_model.h5")
print("Modèle sauvegardé dans model/Suicide_detection_model.h5")
