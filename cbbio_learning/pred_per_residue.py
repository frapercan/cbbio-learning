import sys
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, \
    TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset
from evaluate import load

import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter






experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logging_dir = os.path.join("../logs", experiment_name)

# Cargar el DataFrame desde el archivo CSV proporcionado
df = pd.read_csv('/home/bioxaxi/PycharmProjects/cbbio-learning/data/merged_sf_metamorphic_regions.csv')

# Función para construir etiquetas basadas en los rangos
def build_labels(sequence, regions):
    labels = np.zeros(len(sequence), dtype=np.int64)  # Inicializar tensor de etiquetas con ceros
    region_re = r"(\d+)-(\d+)"  # Expresión regular para extraer los rangos "start-end"

    if isinstance(regions, float) and np.isnan(regions):
        return labels

    found_regions = re.findall(region_re, regions)

    for start, end in found_regions:
        labels[int(start) - 1:int(end)] = 1  # Marcar con 1 las posiciones dentro del rango

    return labels

# Crear una lista para almacenar las filas desdobladas
rows_list = []

# Para cada fila, desdoblar la información relacionada con la cadena A y B
for _, row in df.iterrows():
    # Verificar si las secuencias no son NaN antes de proceder
    if pd.notna(row['resSF.pdb.pairA.sequence']):
        # Crear la fila para la cadena A
        row_chainA = {
            'retainedID': row['retainedID_x'],
            'resSF.pdb.pair': row['resSF.pdb.pairA_x'],
            'resSF.pdb.pair.sequence': row['resSF.pdb.pairA.sequence'],
            'Unip.chain.pair': row['Unip.chain.pairA'],
            'pdb_id': row['pdb_id'],
            'chain': row['chain'],
            'source': 'A'
        }
        # Generar etiquetas para la cadena A
        row_chainA['labels'] = build_labels(row_chainA['resSF.pdb.pair.sequence'], row_chainA['resSF.pdb.pair'])
        rows_list.append(row_chainA)

    if pd.notna(row['resSF.pdb.pairB.sequence']):
        # Crear la fila para la cadena B
        row_chainB = {
            'retainedID': row['retainedID_x'],
            'resSF.pdb.pair': row['resSF.pdb.pairB_x'],
            'resSF.pdb.pair.sequence': row['resSF.pdb.pairB.sequence'],
            'Unip.chain.pair': row['Unip.chain.pairB'],
            'pdb_id': row['pdb_idB'],
            'chain': row['chainB'],
            'source': 'B'
        }
        # Generar etiquetas para la cadena B
        row_chainB['labels'] = build_labels(row_chainB['resSF.pdb.pair.sequence'], row_chainB['resSF.pdb.pair'])
        rows_list.append(row_chainB)

# Crear un nuevo DataFrame a partir de la lista de filas
df_separated = pd.DataFrame(rows_list)


# Mostrar las primeras filas del nuevo DataFrame separado
print(df_separated.head())

# Proceder con el resto del código tal como estaba en pred_per_residue.py

print("Generando etiquetas para las secuencias...")

sequences = df_separated['resSF.pdb.pair.sequence'].tolist()
labels = df_separated['labels'].tolist()

df_separated.to_csv("./etiquetas.csv",index= False)



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
model = AutoModelForTokenClassification.from_pretrained("facebook/esm2_t12_35M_UR50D", num_labels=3)
data_collator = DataCollatorForTokenClassification(tokenizer)

print("Configurando parámetros de entrenamiento...")

args = TrainingArguments(
    output_dir=f"./{experiment_name}",  # Guarda los modelos y otros resultados en un directorio específico del experimento
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=50,
    weight_decay=0.001,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir=logging_dir,  # Directorio para los logs de TensorBoard
    logging_strategy="epoch",  # Registra las métricas en cada época
    report_to="tensorboard",  # Usa TensorBoard para logging
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
# Configurar el Trainer para usar el callback personalizado
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()
