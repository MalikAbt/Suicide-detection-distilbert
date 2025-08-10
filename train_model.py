import numpy as np
import pandas as pd
from transformers import TrainingArguments, Trainer, BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification, EarlyStoppingCallback
from datasets import Dataset
import os

def main():
    # Chargement des données
    df = pd.read_csv('Data/Suicide_Detection.csv')
    print(df['class'].unique())
    df = df.dropna(subset=['text'])
    df = df.sample(n=10000, random_state=42)
    df['labels'] = df['class'].map({'non-suicide': 0, 'suicide': 1})
    df['labels'] = df['labels'].astype(int)
  

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    dataset = Dataset.from_pandas(df[['text', 'labels']])

    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=64                # 64 au lieu de 128 pour diminuer le temps de calcul. Correspond a 64 tokens. 
        )

    # Tokenize avec multiprocessing (OK sous Windows dans ce bloc)
    tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=4)

    # Split train/test
    split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = split['train']
    eval_dataset = split['test']

    # Charger modèle
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Arguments d'entraînement
    training_args = TrainingArguments(
        output_dir="./bert_finetuned",
        optim="adamw_torch",
        eval_strategy="steps",   
        eval_steps=500,
        save_steps=500,
        per_device_train_batch_size=4, # reduit à 4 pour optimiser le traitement
        per_device_eval_batch_size=4,  # reduit à 4 pour optimiser le traitement
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=False,                     # True pour gpu
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Démarrage de l'entraînement...")
    trainer.train(resume_from_checkpoint=False)
    print("Entraînement terminé.")

    # Évaluation
    metrics = trainer.evaluate()
    print(metrics)

    # Sauvegarde du modèle
    os.makedirs("model", exist_ok=True)
    trainer.save_model("model/Suicide_detection_model_Bert")
    print("Modèle sauvegardé dans model/Suicide_detection_model_Bert")

if __name__ == "__main__":
    main()
