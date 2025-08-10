# Suicide Detection with LSTM and GloVe Embeddings

Ce projet consiste à entraîner un modèle de deep learning (LSTM bidirectionnel) pour détecter des textes suicidaires à partir de données issues de posts Reddit et autres sources.

## Description

- Utilisation de **GloVe embeddings** (100 dimensions) pour représenter les mots.
- Pré-traitement du texte (nettoyage, suppression d’émojis, mise en minuscules).
- Modèle LSTM bidirectionnel avec dropout pour limiter l’overfitting.
- Dataset : Suicide Detection Dataset de Kaggle , envrion 1 millions de ligne .
- Gestion du déséquilibre de classes via pondération (class weights).
- Implémenté en Python avec TensorFlow / Keras.

## Objectif

Tester la capacité d’un modèle LSTM simple à détecter des posts potentiellement suicidaires, en exploitant des embeddings pré-entraînés.



