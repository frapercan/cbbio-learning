import pandas as pd
import numpy as np
import re

from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, \
    TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset
from evaluate import load

import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Configuración del logging
logging_dir = "../logs"
writer = SummaryWriter(log_dir=logging_dir)

class ConfusionMatrixCallback(TrainerCallback):
    def __init__(self, log_dir, tokenizer, writer):
        self.log_dir = log_dir
        self.tokenizer = tokenizer
        self.writer = writer

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        eval_dataloader = kwargs['eval_dataloader']
        model = kwargs['model']
        model.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in eval_dataloader:
                inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(model.device)

                outputs = model(**inputs)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=['O', '1'], yticklabels=['O','1'])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        img = Image.open(buf)
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1)

        self.writer.add_image("Confusion Matrix", img_tensor, state.global_step)
        buf.close()

# Configuración del experimento
experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logging_dir = os.path.join("../logs", experiment_name)

# Cargar el DataFrame desde el archivo CSV proporcionado
df = pd.read_csv('./SF_concatenated_sequences_with_labels.csv')

# Usar las secuencias y etiquetas directamente desde el DataFrame cargado
sequences = df['concatenated_sequence'].tolist()
labels = [list(map(int, list(label_seq))) for label_seq in df['SF_label_sequence']]

print(f"Etiquetas generadas. Total de secuencias: {len(sequences)}")

print("Tokenizando secuencias...")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.25, shuffle=True)

train_tokenized = tokenizer(train_sequences)
test_tokenized = tokenizer(test_sequences)

train_dataset = Dataset.from_dict(train_tokenized)
test_dataset = Dataset.from_dict(test_tokenized)

# Agregar las etiquetas a los datasets
train_dataset = train_dataset.add_column("labels", train_labels)
test_dataset = test_dataset.add_column("labels", test_labels)

print("Secuencias tokenizadas y datasets creados.")

print("Cargando modelo preentrenado...")
model = AutoModelForTokenClassification.from_pretrained("facebook/esm2_t12_35M_UR50D", num_labels=2)

data_collator = DataCollatorForTokenClassification(tokenizer)

print("Configurando parámetros de entrenamiento...")

args = TrainingArguments(
    output_dir=f"./{experiment_name}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=50,
    weight_decay=0.001,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir=logging_dir,
    logging_strategy="epoch",
    report_to="tensorboard",
)

metric = load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    labels = labels.reshape((-1,))
    predictions = np.argmax(predictions, axis=2)
    predictions = predictions.reshape((-1,))
    predictions = predictions[labels != -100]
    labels = labels[labels != -100]
    return metric.compute(predictions=predictions, references=labels)

print("Iniciando entrenamiento del modelo...")

confusion_matrix_callback = ConfusionMatrixCallback(log_dir=logging_dir, tokenizer=tokenizer, writer=writer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[confusion_matrix_callback],
)

trainer.train()

writer.close()
