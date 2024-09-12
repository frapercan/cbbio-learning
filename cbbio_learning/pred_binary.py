import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.optim as optim

# Verificación de CUDA
if not torch.cuda.is_available():
    raise Exception("CUDA no está disponible. Se requiere una GPU con CUDA.")
device = torch.device("cuda")

# Definición del clasificador binario
class ProteinClassifier(nn.Module):
    def __init__(self):
        super(ProteinClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        ).to(device)

    def forward(self, embeddings):
        logits = self.classifier(embeddings)
        return logits

# Definición del conjunto de datos
class ProteinDataset(Dataset):
    def __init__(self, dataframe, model_name='Rostlab/ProstT5', device='cpu', mode='seq'):
        self.dataframe = dataframe
        self.embeddings = []
        self.labels = []
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5EncoderModel.from_pretrained(model_name).to(device)
        model.eval()

        with torch.no_grad():
            i = 0
            for _, row in dataframe.iterrows():
                print(i)
                i+=1
                sequence = row[mode]
                try:
                    label = row['label']
                except KeyError:
                    label = False
                # Procesamiento y tokenización
                sequence_processed = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
                inputs = tokenizer(sequence_processed, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=True).to(device)

                # Generación de embeddings
                outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                embeddings = outputs.last_hidden_state.mean(dim=1)

                self.embeddings.append(embeddings.cpu().numpy())
                self.labels.append(label)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx]), torch.tensor(self.labels[idx], dtype=torch.float)

# Carga y división del conjunto de datos
df = pd.read_csv('sf_dataset.csv')
mode = "seq"
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_dataset = ProteinDataset(df_train, mode=mode, device=device)
test_dataset = ProteinDataset(df_test, mode=mode, device=device)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Configuración del modelo, optimizador y función de pérdida
def setup_model():
    model = ProteinClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    return model, optimizer, loss_fn, device

# Funciones de entrenamiento y validación
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct_predictions, total_predictions = 0, 0, 0
    for sequences, labels in dataloader:
        sequences, labels = sequences.to(device), labels.to(device).float().unsqueeze(-1)
        optimizer.zero_grad()
        logits = model(sequences).squeeze(-1)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.sigmoid(logits) >= 0.5
        correct_predictions += (preds == labels.bool()).sum().item()
        total_predictions += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, correct_predictions, total_predictions = 0, 0, 0
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device).float().unsqueeze(-1)
            logits = model(sequences).squeeze(-1)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            preds = torch.sigmoid(logits) >= 0.5
            correct_predictions += (preds == labels.bool()).sum().item()
            total_predictions += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

# Entrenamiento y validación del modelo
def train_and_validate(model, train_dataloader, test_dataloader, optimizer, loss_fn, device, num_epochs):
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_dataloader, optimizer, loss_fn, device)
        val_loss, val_accuracy = validate_one_epoch(model, test_dataloader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        print(f'Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, '
              f'Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    return train_losses, val_losses, train_accuracies, val_accuracies


def make_inferences(model, dataloader, device):
    model.eval()
    predictions = []
    i = 0
    with torch.no_grad():
        print(i)
        i+=1
        for sequences, _ in dataloader:  # Si no necesitas las etiquetas en inferencia
            sequences = sequences.to(device)
            logits = model(sequences).squeeze(-1)
            preds = torch.sigmoid(logits) >= 0.5
            predictions.extend(preds.cpu().numpy())  # Procesa todos los batch juntos
    return predictions



# Ejecución del entrenamiento
model, optimizer, loss_fn, device = setup_model()
num_epochs = 80
train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(model, train_dataloader, test_dataloader, optimizer, loss_fn, device, num_epochs)

# Evaluación y visualización de la matriz de confusión
def evaluate_and_plot_confusion_matrix(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device).float().unsqueeze(-1)
            logits = model(sequences).squeeze(-1)
            preds = torch.sigmoid(logits) >= 0.5

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

evaluate_and_plot_confusion_matrix(model, test_dataloader, device)

from Bio import SeqIO
import pandas as pd
import re
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5EncoderModel


# Función para procesar el encabezado y extraer la información relevante
def process_fasta_header(header):
    # Dividimos el encabezado por los delimitadores para extraer la información
    parts = header.split('|')
    protein_id = parts[1]  # Código de acceso de la proteína
    protein_name = parts[2]  # Nombre corto de la proteína

    description_parts = parts[2].split(' ')
    description = " ".join(description_parts[1:])  # Descripción de la proteína

    # Extraer otras partes del encabezado usando expresiones regulares
    organism_match = re.search(r'OS=([^ ]+)', header)
    organism = organism_match.group(1) if organism_match else None

    gene_match = re.search(r'GN=([^ ]+)', header)
    gene = gene_match.group(1) if gene_match else None

    taxon_match = re.search(r'OX=([^ ]+)', header)
    taxon = taxon_match.group(1) if taxon_match else None

    return protein_id, protein_name, description, organism, gene, taxon


# Leer el archivo FASTA y extraer secuencias junto con los encabezados
fasta_file = '/home/bioxaxi/PycharmProjects/cbbio-learning/cbbio_learning/uniprotkb_proteome_UP000000625_2024_09_11.fasta'
headers = []
sequences = []
protein_ids = []
protein_names = []
descriptions = []
organisms = []
genes = []
taxons = []

for record in SeqIO.parse(fasta_file, "fasta"):
    headers.append(record.id)  # Almacenar el encabezado completo
    sequences.append(str(record.seq))  # Almacenar la secuencia

    # Procesar el encabezado para extraer los campos individuales
    protein_id, protein_name, description, organism, gene, taxon = process_fasta_header(record.description)
    protein_ids.append(protein_id)
    protein_names.append(protein_name)
    descriptions.append(description)
    organisms.append(organism)
    genes.append(gene)
    taxons.append(taxon)

# Crear un DataFrame con toda la información extraída
df_proteome = pd.DataFrame({
    "header": headers,
    "seq": sequences,
    "protein_id": protein_ids,
    "protein_name": protein_names,
    "description": descriptions,
    "organism": organisms,
    "gene": genes,
    "taxon": taxons
})

# Reemplazar caracteres no válidos en las secuencias por "X"
df_proteome["seq"] = df_proteome["seq"].apply(lambda x: re.sub(r"[UZOB]", "X", x))

proteome_dataset = ProteinDataset(df_proteome, mode="seq", device=device)
proteome_dataloader = DataLoader(proteome_dataset, batch_size=1, shuffle=False)


predictions = make_inferences(model, proteome_dataloader, device)

# Añadir las predicciones al DataFrame
df_proteome['prediction'] = predictions

# Guardar el DataFrame con predicciones a un archivo CSV
df_proteome.to_csv('proteome_predictions.csv', index=False)

# Mostrar las primeras filas del DataFrame con predicciones
print(df_proteome.head())


# Mostrar las primeras filas del DataFrame
df_proteome.head()
